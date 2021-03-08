/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef _OCM_ONNX_OPS_LIST_H_
#define _OCM_ONNX_OPS_LIST_H_

#include <iostream>
#include <set>

namespace ocm{
  
std::set<std::string> common_supported_ops = {
      "Add",
      "And",
      "AveragePool",
      "BatchNormalization",
      "Cast",
      "Clip",
      "Concat",
      "Constant",
      "ConstantOfShape",
      "Conv",
      "ConvTranspose",
      "DepthToSpace",
      "Div",
      "Dropout",
      "Elu",
      "Equal",
      "Erf",
      "Exp",
      "Flatten",
      "Floor",
      "Gather",
      "Gemm",
      "GlobalAveragePool",
      "Greater",
      "Identity",
      "InstanceNormalization",
      "LeakyRelu",
      "Less",
      "Log",
      "LRN",
      "LSTM",
      "MatMul",
      "Max",
      "MaxPool",
      "Mean",
      "Min",
      "Mul",
      "Neg",
      "OneHot",
      "Pad",
      "Pow",
      "PRelu",
      "Reciprocal",
      "ReduceMax",
      "ReduceMean",
      "ReduceMin",
      "ReduceSum",
      "Relu",
      "Reshape",
      "Shape",
      "Sigmoid",
      "Slice",
      "Softmax",
      "SpaceToDepth",
      "Split",
      "Sqrt",
      "Squeeze",
      "Sub",
      "Sum",
      "Tanh",
      "TopK",
      "Transpose",
      "Unsqueeze",
  };

  std::set<std::string> supported_ops_cpu = {
    "Abs",
    "Acos",
    "Acosh",
    "ArgMax",
    "ArgMin",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "Cos",
    "Cosh",
    "GlobalLpPool",
    "HardSigmoid",
    "Not",
    "ReduceLogSum",
    "ReduceProd",
    "ReduceSumSquare",
    "Resize",
    "Selu",
    "Sign",
    "Sinh",
    "Softsign",
    "Tan"
  };


  std::set<std::string> supported_ops_gpu = {
    "Abs",
    "Asin",
    "Asinh",
    "Atan",
    "Ceil",
    "GlobalLpPool",
    "HardSigmoid",
    "Not",
    "Selu",
    "Tan",
  };
  std::set<std::string> supported_ops_vpu = {
    "ReduceLogSum",
    "ReduceSumSquare",
    "SinFloat",
  };

}//namespace ocm 

#endif //_OCM_ONNX_OPS_LIST_H_