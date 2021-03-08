/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef _OCM_TF_OPS_LIST_H_
#define _OCM_TF_OPS_LIST_H_

#include <iostream>
#include <set>

namespace ocm{

/**
 *  OpenVINO 2021.1 supported TF ops, Refer following page:
 *  https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html
 */
std::set<std::string> common_supported_ops = {
    // "Abs", // Unittest - cwise_math  // Not supported by OV
    "Add",
    "AddN",
    "AddV2",  // implemented using ngraph ADD
    "All", // the TF api for this is tf.math.reduce_all
    "ArgMax",
    "AvgPool",
    //"BatchToSpaceND", // New models...Commented as Not supported in Bridge
    "BiasAdd",
    "Cast", // Unittest
    "ConcatV2",
    "Const",
    "Conv2D",
    "Conv2DBackpropInput",
    //"CropAndResize", // Commented as Not supported in Bridge
    "DepthwiseConv2dNative",
    "DepthToSpace", // Unittest
    "Equal",
    "Exp",
    "ExpandDims",
    "Fill", // Unittest
    "FloorMod",
    // "FloorDiv", //Unit test - cwise_math
    "FusedBatchNorm",
    "Gather",
    "GatherV2",
    "Greater", // Unittest - cwise_math
    "GreaterEqual",
    "Identity",
    "LRN",
    "Less", //Unit test - Softmax
    "LogSoftmax",//Unit test - Softmax
    "LogicalAnd",
    "MatMul",
    "Max",
    "Maximum",
    "MaxPool",
    "Mean",
    "Minimum",
    "MirrorPad",
    "Mul",
    "OneHot", // Unittest
    "Pack",
    "Pad",
    "PadV2",
    "Placeholder",
    "Range",
    "RealDiv",
    "Relu",
    "Relu6",
    "Reshape",
    //"ResizeBilinear", //Sprint-3 ...Commented as Not supported in Bridge
    //"ResizeNearestNeighbor", // New models...Commented as Not supported in Bridge
    //"Round", // New models...Commented as Not supported in Bridge
    "Rsqrt", // Unittest
    "Shape",
    //"Sign", // Unittest - cwise_math // Not supported by OV
    "Size", // Unittest
    "Sigmoid", // Unittest
    "Slice",
    "Softmax",
    //"SpaceToBatchND", // New models...Commented as Not supported in Bridge
    "SpaceToDepth",
    "Split",
    "SplitV",
    "Squeeze",
    "StridedSlice",
    "Sub",
    "Sum", // Unittest - cwise_math
    "Tile",
    "TopKV2",
    "Transpose",
    "Unpack",
    //"Where",  // Commented as it introduces dynamic shape error
    "ZerosLike" // Unittest
};

/**
 *  TF OPs supported through composite ops i.e. translated using multiple other available Ngraph OPs
 *  and are not supported by MO route
 */
std::set<std::string> composite_ops = {
    "ArgMin",
    "FusedBatchNormV3",
    "_FusedConv2D",
    "_FusedMatMul",
    "NonMaxSuppressionV2",
    "NoOp"
};

//Op supported only on CPU and not supported on VPU
std::set<std::string> cpu_only_ops = {
    "Acos", // Unittest - cwise_math
    "Acosh", // Unittest - cwise_math
    "Asin", // Unittest - cwise_math
    "Asinh", // Unittest - cwise_math
    "Atan", // Unittest - cwise_math
    "Atanh", // Unittest - cwise_math
    "Bucketize",
    "ExperimentalSparseWeightedSum",
    "Neg", // Unittest - cwise_math    
    "Sinh", // Unittest - cwise_math
    "SparseToDense",
    "Tanh" 
};

std::set<std::string> gpu_only_ops = {
    "Acos", // Unittest - cwise_math
    "Acosh", // Unittest - cwise_math
    "Asin", // Unittest - cwise_math
    "Asinh", // Unittest - cwise_math
    "Atan", // Unittest - cwise_math
    "Atanh", // Unittest - cwise_math
    "Neg", // Unittest - cwise_math 
    "Sinh", // Unittest - cwise_math       
    "Tanh" 
};

std::set<std::string> vpu_only_ops = {
};

const std::map<std::string, std::set<string>> ov_2021_2_op_update_cpu = {
  {"add", {}},    //Ops newly added by OpenVINO in this version 
  {"remove", {}}, //Ops removed by OpenVINO in this version
  {"update", {"Abs","FloorDiv", "Sign", "Prod", "Softplus", "LeakyRelu"}}  // Ops for which OCM has enabled support.
};

const std::map<std::string, std::set<string>> ov_2021_2_op_update_gpu = {
  {"add", {}},    //Ops newly added by OpenVINO in this version 
  {"remove", {}}, //Ops removed by OpenVINO in this version
  {"update", {"Prod", "Softplus", "LeakyRelu"}}  // Ops for which OCM has enabled support.
};

const std::map<std::string, std::set<string>> ov_2021_2_op_update_vpu = {
  {"add", {}},    //Ops newly added by OpenVINO in this version 
  {"remove", {}}, //Ops removed by OpenVINO in this version
  {"update", {"FloorDiv", "Prod", "Softplus", "LeakyRelu"}}  // Ops for which OCM has enabled support.
};

} //namespace ocm 

#endif //_OCM_TF_OPS_LIST_H_