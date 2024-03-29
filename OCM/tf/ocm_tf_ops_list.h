/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef _OCM_TF_OPS_LIST_H_
#define _OCM_TF_OPS_LIST_H_

#include <iostream>
#include <set>

namespace ocm {

/**
 *  OpenVINO 2021.1 supported TF ops, Refer following page:
 *  https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html
 */
std::set<std::string> common_supported_ops = {
    "Add", "AddN",
    "AddV2", // implemented using ngraph ADD
    "All",   // the TF api for this is tf.math.reduce_all
    "ArgMax", "AvgPool",
    "BiasAdd",
    "Cast", 
    "ConcatV2", "Const", "Conv2D", "Conv2DBackpropInput",
    "DepthwiseConv2dNative",
    "DepthToSpace", 
    "Equal", "Exp", "ExpandDims",
    "Fill", 
    "FloorMod",
    "FusedBatchNorm", "Gather", "GatherV2",
    "Greater", 
    "GreaterEqual", "Identity", "LRN",
    "Less",       
    "LogSoftmax", 
    "LogicalAnd", "MatMul", "Max", "Maximum", "MaxPool", "Mean", "Minimum",
    "MirrorPad", "Mul",
    "OneHot", 
    "Pack", "Pad", "PadV2", "Placeholder", "Range", "RealDiv", "Relu", "Relu6",
    "Reshape",
    "Rsqrt", 
    "Shape",
    "Size",    
    "Sigmoid", 
    "Slice", "Softmax",
    "SpaceToDepth", "Split", "SplitV", "Square", "Squeeze", "StridedSlice",
    "Sub",
    "Sum", 
    "Tile", "TopKV2", "Transpose", "Unpack",
    // "Where",  // Commented as it introduces dynamic shape error
    "ZerosLike" 
};

/**
 *  TF OPs supported through composite ops i.e. translated using multiple other
 * available Ngraph OPs
 *  and are not supported by MO route
 */
std::set<std::string> composite_ops = {"ArgMin",
                                       "FusedBatchNormV2",
                                       "FusedBatchNormV3",
                                       "_FusedBatchNormEx",
                                       "_FusedConv2D",
                                       "_FusedDepthwiseConv2dNative",
                                       "_FusedMatMul",
                                       "NonMaxSuppressionV2",
                                       "NoOp"};

// Op supported only on CPU and not supported on VPU
std::set<std::string> cpu_only_ops = {
    "Acos",  // Unittest - cwise_math
    "Acosh", // Unittest - cwise_math
    "Asin",  // Unittest - cwise_math
    "Asinh", // Unittest - cwise_math
    "Atan",  // Unittest - cwise_math
    "Atanh", // Unittest - cwise_math
    "Bucketize",     "ExperimentalSparseWeightedSum",
    "Neg",  // Unittest - cwise_math
    "Sinh", // Unittest - cwise_math
    "SparseToDense", 
    "Tanh"};

std::set<std::string> gpu_only_ops = {"Acos",  // Unittest - cwise_math
                                      "Acosh", // Unittest - cwise_math
                                      "Asin",  // Unittest - cwise_math
                                      "Asinh", // Unittest - cwise_math
                                      "Atan",  // Unittest - cwise_math
                                      "Atanh", // Unittest - cwise_math
                                      "Neg",   // Unittest - cwise_math
                                      "Sinh",  // Unittest - cwise_math
                                      "Tanh"};

std::set<std::string> vpu_only_ops = {};

const std::map<std::string, std::set<string>> ov_2021_2_op_update_cpu = {
    {"add", {}},    // Ops newly added by OpenVINO in this version
    {"remove", {}}, // Ops removed by OpenVINO in this version
    {"update", {"Abs", "FloorDiv", "Sign", "Prod", "Softplus",
                "LeakyRelu"}} // Ops for which OCM has enabled support.
};

const std::map<std::string, std::set<string>> ov_2021_2_op_update_gpu = {
    {"add", {}},    // Ops newly added by OpenVINO in this version
    {"remove", {}}, // Ops removed by OpenVINO in this version
    {"update", {"Prod", "Softplus","LeakyRelu"}} // Ops for which OCM has enabled support.
};

const std::map<std::string, std::set<string>> ov_2021_2_op_update_vpu = {
    {"add", {}},    // Ops newly added by OpenVINO in this version
    {"remove", {}}, // Ops removed by OpenVINO in this version
    {"update", {"FloorDiv", "Prod", "Softplus", "LeakyRelu",
                "Tanh"}} // Ops for which OCM has enabled support.
};

const std::map<std::string, std::set<string>> ov_2021_3_op_update_cpu = {
    {"update", {"Log", "MaxPoolV2", "Sqrt"}} // Ops for which OCM has enabled support.
};

const std::map<std::string, std::set<string>> ov_2021_3_op_update_gpu = {
    {"update", {"Log","MaxPoolV2"}} // Ops for which OCM has enabled support.
};

const std::map<std::string, std::set<string>> ov_2021_3_op_update_vpu = {
    {"update", {"Neg", "Log", "MaxPoolV2"}} // Ops for which OCM has enabled support.
};

const std::map<std::string, std::set<string>> ov_2021_4_op_update_cpu = {
    {"update",
     {"ResizeBilinear", "ResizeNearestNeighbor", "Round",
      "GatherNd", "CropAndResize", "Reverse", "Reciprocal", "BatchToSpaceND", 
      "SpaceToBatchND", "Elu", "FakeQuantWithMinMaxVars",
      "Cos", "Cosh", "Sin", "Tan", "Conv3D", "MaxPool3D", "Floor", "ScatterNd",
      "AvgPool3D", "Conv3DBackpropInputV2"}} // Ops for which OCM has enabled support.
};

const std::map<std::string, std::set<string>> ov_2021_4_op_update_gpu = {
    {"update",
     {"ResizeBilinear", "ResizeNearestNeighbor", "Round",
      "GatherNd", "CropAndResize", "Reverse",
      "Reciprocal", "BatchToSpaceND", "SpaceToBatchND", "Elu", "FakeQuantWithMinMaxVars",
      "Conv3D", "MaxPool3D", "Floor", "ScatterNd", "AvgPool3D"}} // Ops for which OCM has enabled support.
};

const std::map<std::string, std::set<string>> ov_2021_4_op_update_vpu = {
    {"update", {"ResizeBilinear", "ResizeNearestNeighbor", "Round",
                "GatherNd", "CropAndResize", "Reverse",
                "Reciprocal", "BatchToSpaceND", "SpaceToBatchND", "Elu", 
                "FakeQuantWithMinMaxVars", "MaxPool3D", "Floor", "ScatterNd",
                "AvgPool3D", "Conv3DBackpropInputV2"}} // Ops for which OCM has enabled support.
};

const std::map<std::string, std::set<string>> ov_2022_1_0_op_update_cpu = {
    {"update", {"SquaredDifference", "LessEqual", "NotEqual", "Cumsum", "NonMaxSuppression", "NonMaxSuppressionV2", "NonMaxSuppressionV3", 
    "NonMaxSuppressionV4", "NonMaxSuppressionV5", "CTCGreedyDecoder", "BatchMatMulV2", "BatchMatMul", "_MklSwish"
    }}
};

const std::map<std::string, std::set<string>> ov_2022_1_0_op_update_gpu = {
    {"update", {"SquaredDifference", "LessEqual", "NotEqual", "Cumsum", "NonMaxSuppression", "NonMaxSuppressionV2", "NonMaxSuppressionV3", 
    "NonMaxSuppressionV4", "NonMaxSuppressionV5", "BatchMatMulV2"
    }}
};

const std::map<std::string, std::set<string>> ov_2022_1_0_op_update_vpu = {
    {"update", {"SquaredDifference", "NonMaxSuppression", "NonMaxSuppressionV2", "NonMaxSuppressionV3", 
    "NonMaxSuppressionV4", "NonMaxSuppressionV5", "BatchMatMulV2", "Conv3D"
    }},    
    // Disabling "Range" op as OV is throwing Dynamic to Staitc error for it
    // Even in the case when all the inputs to it are constant
    {"remove", {"Range"}} // Ops removed by OpenVINO in this version
};

const std::map<std::string, std::set<string>> ov_2022_2_0_op_update_cpu = {
    {"update", {"Select","SegmentSum", "ParallelDynamicStitch", "DynamicPartition", "Erf","Einsum", "BroadcastGradientArgs","Concat","ExtractImagePatches","LogicalNot","LogicalOr","LogicalXor","Mod","RandomUniform","Roll","SelectV2","Swish"}}
};

} // namespace ocm

#endif //_OCM_TF_OPS_LIST_H_