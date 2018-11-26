#pragma once
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

#define RIND(b,x,y) ((b)*dim + (y))*dim + (x)
#define XIND(b,x) (b)*dim + (x)

using namespace tensorflow;

template <typename Device, typename T, typename Tmask>
struct launchCholUpdateKernel {
  void operator()(const Device& d,
            typename TTypes<T>::Flat output, typename TTypes<T>::Flat x_workspace,
            typename TTypes<T>::ConstFlat x, typename TTypes<Tmask>::ConstFlat m,
            int batch_size, int dim);
};