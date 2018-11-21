#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>
#include "tensorflow/core/util/cuda_device_functions.h"

#include "cholesky_update.h"

using GPUDevice = Eigen::GpuDevice;

template <typename dtype>
__global__ void CholUpdateKernel(dtype* R, dtype* x, int batch_size, int dim){
	for(int b : tensorflow::CudaGridRangeX(batch_size))
	{
		for(int k = 0; k < dim; k++)
		{
			dtype Rkk = R[RIND(b,k,k)];
			dtype xk = x[XIND(b,k)];

			dtype r = sqrt(Rkk*Rkk + xk*xk);
			dtype c = r/Rkk;
			dtype s = xk/Rkk;
			R[RIND(b,k,k)] = r;
			for(int i = k+1; i < dim; i++)
			{
				R[RIND(b,i,k)] = (R[RIND(b,i,k)] + s*x[XIND(b,i)])/c;
				x[XIND(b,i)] = c*x[XIND(b,i)] - s*R[RIND(b,i,k)];
			}
		}
	}
}

template <typename dtype>
struct launchCholUpdateKernel<GPUDevice, dtype> {
	void operator()(const GPUDevice& d, 
			typename TTypes<dtype>::Flat R, typename TTypes<dtype>::Flat x,
			typename TTypes<dtype>::ConstFlat x_in,
			int batch_size, int dim) {
		
		//To32Bit(R).device(d) = To32Bit(R_in);
		To32Bit(x).device(d) = To32Bit(x_in);

		const int kThreadsPerBlock = 1024;
		
		CholUpdateKernel<dtype><<<(dim*batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
						kThreadsPerBlock, 0, d.stream()>>>(
							R.data(), x.data(), batch_size, dim);

		cudaError_t cudaerr = cudaDeviceSynchronize();
		if (cudaerr != cudaSuccess)
			printf("kernel launch failed with error \"%s\".\n",
				cudaGetErrorString(cudaerr));
	}
};

//forward declaration for all the types needed
typedef Eigen::GpuDevice GPUDevice;
#define ADD_KERNEL_TYPE(type)							\
	template struct launchCholUpdateKernel<GPUDevice, type>;	\

ADD_KERNEL_TYPE(float);
ADD_KERNEL_TYPE(double);

#undef ADD_KERNEL_TYPE