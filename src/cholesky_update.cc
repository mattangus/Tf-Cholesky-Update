#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/padding.h"

#include <iostream>
#include <cuda.h>

#include "cholesky_update.h"

using namespace tensorflow;
using namespace std;
using namespace shape_inference;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

//For now only accept 3 and 2 for rank
//i.e. batch of matrices and batch of vectors
Status ShapeFn(InferenceContext* c)
{
	//check input shape has 3 dimensions (batch, d, d)
	ShapeHandle r_shape;
	TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &r_shape));

	//check indices has 2 dimensions (batch, d)
	ShapeHandle x_shape;
	TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &x_shape));

	ShapeHandle m_shape;
	TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &m_shape));

	int r_rank = c->Rank(r_shape);
	int x_rank = c->Rank(x_shape);
	//R must be square
	if (c->Value(c->Dim(r_shape,r_rank - 1)) != c->Value(c->Dim(r_shape, r_rank - 2)))
			return errors::InvalidArgument("R must be square");
	
	if (c->Value(c->Dim(m_shape,0)) != c->Value(c->Dim(m_shape, 0)))
		return errors::InvalidArgument("Batch size must match mask size");

	//R must match shape of xx^T
	for(int i = 0; i < 2; i++)
    {
        DimensionHandle r_dim = c->Dim(r_shape,i);
        DimensionHandle x_dim = c->Dim(x_shape,i);

        if (c->Value(r_dim) != c->Value(x_dim))
            return errors::InvalidArgument(
                "R and x must have same dims");
    }

	//set output size
	c->set_output(0, c->input(0));

	return Status::OK();
}

/**
 * register the operation with necessary options
 */
REGISTER_OP("CholUpdate")
		.Input("r: Ref(T)")
		.Input("x: T")
		.Input("m: Tmask")
		.Output("c: Ref(T)")
        .Attr(GetConvnetDataFormatAttrString())
		.Attr("T: {float32, float64}")
		.Attr("Tmask: {bool,int32,float32,float64}")
		.Attr("use_locking: bool = true")
		.SetShapeFn(ShapeFn);

// int to_ind(int b, int h, int w, int batch, int dim)
// {
// 	return (b*dim + h)*dim + w;
// }

template <typename T, typename Tmask>
struct launchCholUpdateKernel<CPUDevice, T, Tmask> {
	void operator()(const CPUDevice& d,
			typename TTypes<T>::Flat R, typename TTypes<T>::Flat x,
            typename TTypes<T>::ConstFlat x_in, typename TTypes<Tmask>::ConstFlat m,
			int batch_size, int dim) {
		//based on https://stackoverflow.com/a/16160905/1097517
		//R.setZero();
		//R += R_in;
		x.setZero();
		x += x_in;

		for(int b = 0; b < batch_size; b++)
		{
			if(m(b) == 0)
				continue;
			for(int k = 0; k < dim; k++)
			{
				T Rkk = R(RIND(b,k,k));
				T xk = x(XIND(b,k));

				T r = sqrt(Rkk*Rkk + xk*xk);
				T c = r/Rkk;
				T s = xk/Rkk;
				R(RIND(b,k,k)) = r;
				for(int i = k+1; i < dim; i++)
				{
					R(RIND(b,i,k)) = (R(RIND(b,i,k)) + s*x(XIND(b,i)))/c;
					x(XIND(b,i)) = c*x(XIND(b,i)) - s*R(RIND(b,i,k));
				}
			}
		}
	}
};

template <typename Device, typename T, typename Tmask>
class CholUpdateOp : public OpKernel {
public:

	explicit CholUpdateOp(OpKernelConstruction* context)
		: OpKernel(context)
	{
		string data_format;
        OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
        OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                    errors::InvalidArgument("Invalid data format"));

		//only nhwc supported
        OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("CholUpdate only supports NHWC ",
                                "on device type ",
                                DeviceTypeString(context->device_type())));

		OP_REQUIRES_OK(context,
                   context->GetAttr("use_locking", &use_exclusive_lock_));

	}

	void Compute(OpKernelContext* context) override {
		// We always return the input ref.
    	context->forward_ref_input_to_ref_output(0, 0);

		if (use_exclusive_lock_) {
			mutex_lock l(*context->input_ref_mutex(0));
			DoUpdate(context);
		} else {
			DoUpdate(context);
		}
	}
private:
	void DoUpdate(OpKernelContext* context) {
		// Grab the input tensor
		Tensor r_tensor = context->mutable_input(0, use_exclusive_lock_);
		OP_REQUIRES(context, r_tensor.IsInitialized(),
					errors::FailedPrecondition("Attempting to use uninitialized "
											"parameters: ",
											requested_input(0)));
		const Tensor& x_tensor = context->input(1);
		const Tensor& m_tensor = context->input(2);
	

		Tensor x_workspace;
    	OP_REQUIRES_OK(context, context->allocate_temp(
							DataTypeToEnum<T>::v(),
							x_tensor.shape(), &x_workspace));

		int batch_size = GetTensorDim(r_tensor.shape(), data_format_, 'N');
		int dim = GetTensorDim(r_tensor.shape(), data_format_, 'H');

		//flatten tensors
		auto x_work_flat = x_workspace.flat<T>();
		auto r_flat = r_tensor.flat<T>();
		auto x_flat = x_tensor.flat<T>();
		auto m_flat = m_tensor.flat<Tmask>();

		// Call the cuda kernel launcher
		launchCholUpdateKernel<Device, T, Tmask>()(
			context->eigen_device<Device>(),
			r_flat,
			x_work_flat,
			x_flat,
			m_flat,
			batch_size, dim);
	}
    TensorFormat data_format_;
	bool use_exclusive_lock_;
};

//register kernel with types needed
#define REGISTER_GPU(type, mtype) \
	REGISTER_KERNEL_BUILDER( \
		Name("CholUpdate") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T") \
		.TypeConstraint<mtype>("Tmask"), \
		CholUpdateOp<GPUDevice, type, mtype>) \

REGISTER_GPU(float, bool);
REGISTER_GPU(double, bool);
REGISTER_GPU(float, int);
REGISTER_GPU(double, int);
REGISTER_GPU(float, float);
REGISTER_GPU(double, float);
REGISTER_GPU(float, double);
REGISTER_GPU(double, double);

#undef REGISTER_GPU

#define REGISTER_CPU(type, mtype) \
	REGISTER_KERNEL_BUILDER( \
		Name("CholUpdate") \
		.Device(DEVICE_CPU) \
		.TypeConstraint<type>("T") \
		.TypeConstraint<mtype>("Tmask"), \
		CholUpdateOp<CPUDevice, type, mtype>) \

REGISTER_CPU(float, bool);
REGISTER_CPU(double, bool);
REGISTER_CPU(float, int);
REGISTER_CPU(double, int);
REGISTER_CPU(float, float);
REGISTER_CPU(double, float);
REGISTER_CPU(float, double);
REGISTER_CPU(double, double);

#undef REGISTER_CPU