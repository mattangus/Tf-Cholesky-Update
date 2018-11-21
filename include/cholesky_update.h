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

template <typename Device, typename dtype>
struct launchCholUpdateKernel {
  void operator()(const Device& d,
            typename TTypes<dtype>::Flat output, typename TTypes<dtype>::Flat x_workspace,
            const typename TTypes<dtype>::ConstFlat R, const typename TTypes<dtype>::ConstFlat x,
            int batch_size, int dim);
};