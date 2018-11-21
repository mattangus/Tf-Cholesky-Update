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

	int r_rank = c->Rank(r_shape);
	int x_rank = c->Rank(x_shape);
	//R must be square
	if (c->Value(c->Dim(r_shape,r_rank - 1)) != c->Value(c->Dim(r_shape, r_rank - 2)))
			return errors::InvalidArgument("a must be square");

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
		.Input("r: T")
		.Input("x: T")
		.Output("c: T")
        .Attr(GetConvnetDataFormatAttrString())
		.Attr("T: {int32, float32, float64}")
		.Attr("use_locking: bool = true")
		.SetShapeFn(ShapeFn);

// int to_ind(int b, int h, int w, int batch, int dim)
// {
// 	return (b*dim + h)*dim + w;
// }

template <typename dtype>
struct launchCholUpdateKernel<CPUDevice, dtype> {
	void operator()(const CPUDevice& d,
			typename TTypes<dtype>::Flat R, typename TTypes<dtype>::Flat x,
            typename TTypes<dtype>::ConstFlat R_in, typename TTypes<dtype>::ConstFlat x_in,
			int batch_size, int dim) {
		//based on https://stackoverflow.com/a/16160905/1097517
		R.setZero();
		R += R_in;
		x.setZero();
		x += x_in;

		for(int b = 0; b < batch_size; b++)
		{
			for(int k = 0; k < dim; k++)
			{
				dtype Rkk = R(RIND(b,k,k));
				dtype xk = x(XIND(b,k));

				dtype r = sqrt(Rkk*Rkk + xk*xk);
				dtype c = r/Rkk;
				dtype s = xk/Rkk;
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

template <typename Device, typename dtype>
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

		// const DataType dt = DataTypeToEnum<dtype>::v();
		// OP_REQUIRES_OK(context,
		// 			context->MatchSignature({MakeRefType(dt), dt},
		// 									{MakeRefType(dt)}));

		// OP_REQUIRES_OK(context,
        //            context->GetAttr("use_locking", &use_exclusive_lock_));

	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& r_tensor = context->input(0);
		OP_REQUIRES(context, r_tensor.IsInitialized(),
					errors::FailedPrecondition("Attempting to use uninitialized "
											"parameters: ",
											requested_input(0)));
		const Tensor& x_tensor = context->input(1);

		Tensor x_workspace;
    	OP_REQUIRES_OK(context, context->allocate_temp(
							DataTypeToEnum<dtype>::v(),
							x_tensor.shape(), &x_workspace));

		int batch_size = GetTensorDim(r_tensor.shape(), data_format_, 'N');
		int dim = GetTensorDim(r_tensor.shape(), data_format_, 'H');

		//flatten tensors
		auto x_work_flat = x_workspace.flat<dtype>();
		auto r_flat = r_tensor.flat<dtype>();
		auto x_flat = x_tensor.flat<dtype>();

		// std::cout << "here" << __LINE__ << std::endl;

		// x_work_flat.device(context->eigen_device<Device>()).setZero();
		// std::cout << "here" << __LINE__ << std::endl;
		// x_work_flat += x_flat; //how do you just copy?
		// std::cout << "here" << __LINE__ << std::endl;

		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0,
			r_tensor.shape(),&output_tensor));

		//auto output = context->output(0);
		//out->template flat<Scalar>()
		auto output_flat = output_tensor->flat<dtype>();
		// std::cout << "here" << __LINE__ << std::endl;
		// output.setZero();
		// std::cout << "here" << __LINE__ << std::endl;
		// output = r_flat; //how do you just copy?
		// //const int N = output.size();
		// std::cout << "here" << __LINE__ << std::endl;

		// Call the cuda kernel launcher
		launchCholUpdateKernel<Device, dtype>()(
			context->eigen_device<Device>(),
			output_flat,
			x_work_flat,
			r_flat,
			x_flat,
			batch_size, dim);
	}
private:
    TensorFormat data_format_;
	bool use_exclusive_lock_;
};

//register kernel with types needed
#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("CholUpdate") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		CholUpdateOp<GPUDevice, type>) \

REGISTER_GPU(float);
// REGISTER_GPU(double);

#undef REGISTER_GPU

#define REGISTER_CPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("CholUpdate") \
		.Device(DEVICE_CPU) \
		.TypeConstraint<type>("T"), \
		CholUpdateOp<CPUDevice, type>) \

REGISTER_CPU(float);
// REGISTER_CPU(double);

#undef REGISTER_CPU