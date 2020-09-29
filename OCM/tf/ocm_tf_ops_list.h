#ifndef _OCM_TF_OPS_LIST_H_
#define _OCM_TF_OPS_LIST_H_

#include <iostream>
#include <set>

namespace ocm{

//OpenVINO 2020.4 supported TF ops
std::set<std::string> common_supported_ops = {
	"Add",
	"AddN",
	"ArgMax",
	"AvgPool",
	"BatchToSpaceND",
	"BiasAdd",
	"Bucketize",
	"Cast",
	"Ceil",
	"Concat",
	"ConcatV2",
	"Const",
	"Conv2D",
	"Conv2DBackpropInput",
	"Cos",
	"Cosh",
	"CropAndResize",
	"CumSum",
	"DepthToSpace",
	"DepthwiseConv2dNative",
	"Enter",
	"Equal",
	"Exit",
	"Exp",
	"ExpandDims",
	"ExperimentalSparseWeightedSum",
	"ExtractImagePatches",
	"Fill",
	"Floor",
	"FusedBatchNorm",
	"Gather",
	"GatherNd",
	"GatherV2",
	"Greater",
	"GreaterEqual",
	"Identity",
	"LRN",
	"Less",
	"Log",
	"Log1p",
	"LogicalAnd",
	"LogicalOr",
	"LogicalNot",
	"LogSoftmax",
	"LoopCond",
	"MatMul",
	"Max",
	"MaxPool",
	"Maximum",
	"Mean",
	"Merge",
	"Min",
	"Minimum",
	"MirrorPad",
	"Mul",
	"Neg",
	"NextIteration",
	"NonMaxSuppressionV3",
	"NonMaxSuppressionV4",
	"NonMaxSuppressionV5",
	"NoOp",
	"OneHot",
	"Pack",
	"Pad",
	"PadV2",
	"Placeholder",
	"PlaceholderWithDefault",
	"Prod",
	"Range",
	"Rank",
	"RealDiv",
	"Relu",
	"Relu6",
	"Reshape",
	"ResizeBilinear",
	"ResizeNearestNeighbor",
	"ResourceGather",
	"ReverseSequence",
	"Round",
	"Rsqrt",
	"Shape",
	"Sigmoid",
	"Sin",
	"Sinh",
	"Size",
	"Slice",
	"Softmax",
	"Softplus",
	"Softsign",
	"SpaceToBatchND",
	"SparseToDense",
	"Split",
	"SplitV",
	"Sqrt",
	"Square",
	"SquaredDifference",
	"Square",
	"Squeeze",
	"StopGradient",
	"StridedSlice",
	"Sub",
	"Sum",
	"Swish",
	"Switch",
	"Tan",
	"Tanh",
	"TensorArrayGatherV3",
	"TensorArrayReadV3",
	"TensorArrayScatterV3",
	"TensorArraySizeV3",
	"TensorArrayV3",
	"TensorArrayWriteV3",
	"Tile",
	"TopKV2",
	"Transpose",
	"Unpack",
	"Where",
	"ZerosLike"
};

//Layers supported through high level ops transformed by Model Optimizer
std::set<std::string> other_ops = {
"All",
"Assert",
"NonMaxSuppressionV2",
"FusedBatchNormV3",
"LeakyRelu",
"SpaceToDepth",
"OneShotIterator",
"IteratorGetNext",
"swish_f32",
"AddV2",
"MaxPool3D",
"Reciprocal"
};

//Op supported only on CPU and not supported on VPU
std::set<std::string> cpu_only_ops = {
  "Bucketize",
  "ExperimentalSparseWeightedSum",
  "SparseToDense"
};

std::set<std::string> gpu_only_ops = {
"Abs",
};

std::set<std::string> vpu_only_ops = {
"ReduceLogSum",
};

} //namespace ocm 

#endif //_OCM_TF_OPS_LIST_H_