#ifndef _OCM_TF_OPS_LIST_H_
#define _OCM_TF_OPS_LIST_H_

#include <iostream>
#include <set>

namespace ocm{

// OpenVINO 2021.1 supported TF ops, Refer following page:
// https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html
std::set<std::string> common_supported_ops = {
    "Add",
    "AddN",
    "AddV2",  // implemented using ngraph ADD
    "ArgMax",
    "AvgPool",
    "BiasAdd",
    "ConcatV2",
    "Const",
    "Conv2D",
    "FloorMod", //Unit test - Softmax
    "FusedBatchNorm",
    "Identity",
    "Less", //Unit test - Softmax
    "LogSoftmax",//Unit test - Softmax
    "MatMul",
    "MaxPool",
    "Mean",
    "Mul",
    "Pack",
    "Pad",
    "Placeholder",
    "Range", // Unittest - Softmax
    "Relu",
    "Reshape",
    "Shape",
    "Slice", // Unittest 
    "Softmax",
    "Split", // Unittest 
    "SplitV", // Unittest 
    "Squeeze",
    "StridedSlice",
    "Sub",
    "Tile" // Unittest 
    "Transpose" // Unittest - Softmax
    "Unpack" // Unittest 
    "ZerosLike" // Unittest
};

//TF OPs supported through composite ops i.e. translated using multiple other available Ngaph OPs
std::set<std::string> composite_ops = {
    "FusedBatchNormV3",
    "_FusedConv2D",
    "_FusedMatMul",
};

//Op supported only on CPU and not supported on VPU
std::set<std::string> cpu_only_ops = {
    "Bucketize",
    "ExperimentalSparseWeightedSum",
    "SparseToDense"
};

std::set<std::string> gpu_only_ops = {
};

std::set<std::string> vpu_only_ops = {
};

} //namespace ocm 

#endif //_OCM_TF_OPS_LIST_H_