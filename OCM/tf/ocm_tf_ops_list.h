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
    "Abs", // Unittest - cwise_math
    "Add",
    "AddN",
    "AddV2",  // implemented using ngraph ADD
    "ArgMax",
    "AvgPool",
    "BiasAdd",
    "Cast", // Unittest
    "ConcatV2",
    "Const",
    "Conv2D",
    "ExpandDims", // Unittest
    "Fill", // Unittest
    "FloorMod", //Unit test - Softmax
    "FloorDiv", //Unit test - cwise_math
    "FusedBatchNorm",
    "GatherV2", // Unittest
    "Greater", // Unittest - cwise_math
    "Identity",
    "Less", //Unit test - Softmax
    "LogSoftmax",//Unit test - Softmax
    "MatMul",
    "MaxPool",
    "MirrorPad",
    "Mul",
    "OneHot", // Unittest
    "Pack",
    "Pad",
    "PadV2",
    "Placeholder",
    "Range", // Unittest - Softmax
    "RealDiv", // Unittest - cwise_math
    "Relu",
    "Relu6", // Unittest - Relu
    "Reshape",
    "Rsqrt", // Unittest
    "Shape",
    "Sign", // Unittest - cwise_math
    "Size", // Unittest
    "Slice", // Unittest 
    "Softmax",
    "SpaceToDepth", // Unittest
    "Split", // Unittest 
    "SplitV", // Unittest 
    "Squeeze",
    "StridedSlice",
    "Sub",
    "Sum", // Unittest - cwise_math
    "Tile", // Unittest 
    "Transpose", // Unittest - Softmax
    "Unpack", // Unittest 
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
    "Mean",
    "Neg", // Unittest - cwise_math    
    "Sinh", // Unittest - cwise_math
    "SparseToDense"
    "Tanh", // Unittest - cwise_math   
};

std::set<std::string> gpu_only_ops = {
    "Acos", // Unittest - cwise_math
    "Acosh", // Unittest - cwise_math
    "Asin", // Unittest - cwise_math
    "Asinh", // Unittest - cwise_math
    "Atan", // Unittest - cwise_math
    "Atanh", // Unittest - cwise_math
    "Mean",
    "Neg", // Unittest - cwise_math 
    "Sinh", // Unittest - cwise_math       
    "Tanh", // Unittest - cwise_math
};

std::set<std::string> vpu_only_ops = {
};

} //namespace ocm 

#endif //_OCM_TF_OPS_LIST_H_