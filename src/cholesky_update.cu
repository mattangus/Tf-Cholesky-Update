#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>
#include "tensorflow/core/util/cuda_device_functions.h"

#include "cholesky_update.h"

using GPUDevice = Eigen::GpuDevice;

template <typename T, typename Tmask>
__global__ void CholUpdateKernel(T* R, T* x, const Tmask* m, int batch_size, int dim){
	for(int b : tensorflow::CudaGridRangeX(batch_size))
	{
		if(m[b] == 0)
			continue;
		for(int k = 0; k < dim; k++)
		{
			T Rkk = R[RIND(b,k,k)];
			T xk = x[XIND(b,k)];

			T r = sqrt(Rkk*Rkk + xk*xk);
			T c = r/Rkk;
			T s = xk/Rkk;
			R[RIND(b,k,k)] = r;
			for(int i = k+1; i < dim; i++)
			{
				R[RIND(b,i,k)] = (R[RIND(b,i,k)] + s*x[XIND(b,i)])/c;
				x[XIND(b,i)] = c*x[XIND(b,i)] - s*R[RIND(b,i,k)];
			}
		}
	}
}

template <typename T, typename Tmask>
struct launchCholUpdateKernel<GPUDevice, T, Tmask> {
	void operator()(const GPUDevice& d, 
			typename TTypes<T>::Flat R, typename TTypes<T>::Flat x,
			typename TTypes<T>::ConstFlat x_in, typename TTypes<Tmask>::ConstFlat m,
			int batch_size, int dim) {
		
		//To32Bit(R).device(d) = To32Bit(R_in);
		To32Bit(x).device(d) = To32Bit(x_in);

		const int kThreadsPerBlock = 1024;
		
		CholUpdateKernel<T, Tmask><<<(dim*batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
						kThreadsPerBlock, 0, d.stream()>>>(
							R.data(), x.data(), m.data(), batch_size, dim);

		cudaError_t cudaerr = cudaDeviceSynchronize();
		if (cudaerr != cudaSuccess)
			printf("kernel launch failed with error \"%s\".\n",
				cudaGetErrorString(cudaerr));
	}
};

//forward declaration for all the types needed
typedef Eigen::GpuDevice GPUDevice;
#define ADD_KERNEL_TYPE(type,mtype)							\
	template struct launchCholUpdateKernel<GPUDevice, type, mtype>;	\

ADD_KERNEL_TYPE(float, bool);
ADD_KERNEL_TYPE(double, bool);
ADD_KERNEL_TYPE(float, int);
ADD_KERNEL_TYPE(double, int);
ADD_KERNEL_TYPE(float, float);
ADD_KERNEL_TYPE(double, float);
ADD_KERNEL_TYPE(float, double);
ADD_KERNEL_TYPE(double, double);

#undef ADD_KERNEL_TYPE