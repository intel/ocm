#include "ocm_tf_checker.h"
#include "ocm_tf_ops_list.h"

namespace ocm{
  
const std::set<DataType> SupportedTypes(const std::string device_id="CPU"){
  
  const std::set<DataType> cpuSupportedInputTypes = {DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64,
      DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BFLOAT16};
  
  if(device_id=="CPU"){
    return cpuSupportedInputTypes;
  }

  return cpuSupportedInputTypes;
}

const std::set<DataType> SupportedTypesIdx(const std::string device_id="CPU"){

  const std::set<DataType> cpuSupportedInputTypes= {DT_INT32, DT_INT64};

  if(device_id=="CPU"){
    return cpuSupportedInputTypes;
  }
  
  return cpuSupportedInputTypes;
}

const TypeConstraintMap& GetTypeConstraintMap() {
  //
  // A map of op types (e.g. "Add") to type constraint maps. For (fake)
  // example:
  //
  //  type_constraint_map["Cast"]["SrcT"] = {DT_FLOAT, DT_BOOL};
  //  type_constraint_map["Cast"]["DstT"] = {DT_DOUBLE, DT_INT16};
  //
  // ...would mean that for the "Cast" op, the "SrcT" type variable can be
  // DT_FLOAT or DT_BOOL, and the "DstT" type variable can be DT_DOUBLE or
  // DT_INT16.
  //
  static bool initialized = false;
  static TypeConstraintMap type_constraint_map;
  if (!initialized) {
    //
    // Initialize type constraint map.
    //
    type_constraint_map["Add"]["T"] = SupportedTypes();
    type_constraint_map["AddN"]["T"] = SupportedTypes();
    type_constraint_map["AddV2"]["T"] = SupportedTypes();
    type_constraint_map["ArgMax"]["T"] = SupportedTypes();
    type_constraint_map["ArgMax"]["Tidx"] = SupportedTypesIdx();
    type_constraint_map["AvgPool"]["T"] = SupportedTypes();
    type_constraint_map["BiasAdd"]["T"] = SupportedTypes();
    type_constraint_map["ConcatV2"]["T"] = SupportedTypes();
    type_constraint_map["ConcatV2"]["Tidx"] = SupportedTypesIdx();    
    type_constraint_map["Const"]["dtype"] = SupportedTypes();
    type_constraint_map["Conv2D"]["T"] = SupportedTypes();
    type_constraint_map["FusedBatchNorm"]["T"] = SupportedTypes();
    type_constraint_map["FusedBatchNormV3"]["T"] = {DT_FLOAT};
    type_constraint_map["_FusedConv2D"]["T"] = SupportedTypes(); // formed after TF optimization pass, not in original graph
    type_constraint_map["_FusedMatMul"]["T"] = SupportedTypes(); // formed after TF optimization pass, not in original graph
    type_constraint_map["Identity"]["T"] = SupportedTypes();
    type_constraint_map["MatMul"]["T"] = SupportedTypes();
    type_constraint_map["MaxPool"]["T"] = SupportedTypes();
    type_constraint_map["Mean"]["T"] = SupportedTypes();
    type_constraint_map["Mean"]["Tidx"] = SupportedTypesIdx();    
    type_constraint_map["Mul"]["T"] = SupportedTypes();
    type_constraint_map["Pack"]["T"] = SupportedTypes();
    type_constraint_map["Pad"]["Tpaddings"] = SupportedTypes();
    // type_constraint_map['Placeholder']['T'] = SupportedTypes();
    type_constraint_map["Relu"]["T"] = SupportedTypes();
    type_constraint_map["Reshape"]["T"] = SupportedTypes();
    type_constraint_map["Shape"]["T"] = SupportedTypes();
    type_constraint_map["Shape"]["out_type"] = SupportedTypesIdx();    
    type_constraint_map["Softmax"]["T"] = SupportedTypes();
    type_constraint_map["Sub"]["T"] = SupportedTypes();
    type_constraint_map["Squeeze"]["T"] = SupportedTypes();
    type_constraint_map["StridedSlice"]["T"] = SupportedTypes();
    type_constraint_map["StridedSlice"]["Index"] = SupportedTypesIdx();  
    type_constraint_map["Sub"]["T"] = SupportedTypes();  
  }
  return type_constraint_map;
}

std::set<std::string> GetTFSupportedOPs(std::string device_id, std::string ov_version){
	
	std::set<std::string> supported_ops = {};

  if (device_id == "CPU") {
    std::merge(common_supported_ops.begin(), common_supported_ops.end(),
               cpu_only_ops.begin(), cpu_only_ops.end(),
               std::inserter(supported_ops,supported_ops.begin()));
  } else if (device_id == "GPU") {
    std::merge(common_supported_ops.begin(), common_supported_ops.end(),
               gpu_only_ops.begin(), gpu_only_ops.end(),
               std::inserter(supported_ops, supported_ops.begin()));
  } else if (device_id == "MYRIAD" || device_id == "HDDL") {
    std::merge(common_supported_ops.begin(), common_supported_ops.end(),
               vpu_only_ops.begin(), vpu_only_ops.end(),
               std::inserter(supported_ops, supported_ops.begin()));
  }
    return supported_ops;
}
// Checks if the node meets the confirmation constraints
static Status ConfirmationOk(
        tensorflow::Node* node,
        std::map<std::string, ConfirmationFunction>& confirmation_function_map,
        bool& confirmation_ok) {
    auto it = confirmation_function_map.find(node->type_string());
    if (it != confirmation_function_map.end()) {
        TF_RETURN_IF_ERROR(it->second(node, &confirmation_ok));
    }
    return Status::OK();
}

static ConfirmationFunction FusedBatchNormConfirmationFunction() {
  auto cf = [](Node* n, bool* result) {
    bool tf_is_training;
    if (GetNodeAttr(n->attrs(), "is_training", &tf_is_training) !=
        Status::OK()) {
      tf_is_training = true;
    }
    *result = !tf_is_training;
    return Status::OK();
  };
  return cf;
};

// Generates a "simple" confirmation function which always returns true,
static ConfirmationFunction SimpleConfirmationFunction() {
  auto cf = [](tensorflow::Node *, bool* result) {
    *result = true;
    return tensorflow::Status::OK();
  };
  return cf;
};

const std::map<std::string, ConfirmationFunction>& GetConfirmationMap() {
  //
  // A map of op types (e.g. "Add") to confirmation functions. These can be
  // used to check arbitrary constraints. For example:
  //
  //    confirmation_function_map["MyOp"] = [](Node* n, bool* confirmed) {
  //      int dummy;
  //      if (GetAttr(n->attrs(),"my_unsupported_attr",&dummy).ok()) {
  //        *confirmed = false;
  //        return Status::OK();
  //      }
  //      *confirmed = true;
  //      return Status::OK();
  //    };
  //
  // The foregoing function checks every "MyOp" node to make sure that it does
  // not have the attribute "my_unsupported_attr", and rejects placement if it
  // does.
  static std::map<std::string, ConfirmationFunction> confirmation_function_map;
  static bool initialized = false;
  if (!initialized) {
    //
    // Initialize confirmation function map.
    //
    // Please keep these in alphabetical order by op name.
    //
    confirmation_function_map["Abs"] = SimpleConfirmationFunction();
    confirmation_function_map["Acos"] = SimpleConfirmationFunction();
    confirmation_function_map["Add"] = SimpleConfirmationFunction();
    confirmation_function_map["AddN"] = SimpleConfirmationFunction();
    confirmation_function_map["AddV2"] = SimpleConfirmationFunction();
    confirmation_function_map["Any"] = SimpleConfirmationFunction();
    confirmation_function_map["All"] = SimpleConfirmationFunction();
    confirmation_function_map["ArgMax"] = SimpleConfirmationFunction();
    confirmation_function_map["ArgMin"] = SimpleConfirmationFunction();
    confirmation_function_map["Asin"] = SimpleConfirmationFunction();
    confirmation_function_map["Atan"] = SimpleConfirmationFunction();
    confirmation_function_map["AvgPool"] = SimpleConfirmationFunction();
    confirmation_function_map["BatchMatMul"] = SimpleConfirmationFunction();
    confirmation_function_map["BatchMatMulV2"] = SimpleConfirmationFunction();
    confirmation_function_map["BiasAdd"] = SimpleConfirmationFunction();
    confirmation_function_map["Cast"] = SimpleConfirmationFunction();
    confirmation_function_map["Ceil"] = SimpleConfirmationFunction();
    confirmation_function_map["ConcatV2"] = SimpleConfirmationFunction();
    confirmation_function_map["Const"] = SimpleConfirmationFunction();
    confirmation_function_map["Conv2D"] = SimpleConfirmationFunction();
    confirmation_function_map["Conv2DBackpropInput"] = SimpleConfirmationFunction();
    confirmation_function_map["Conv3D"] = SimpleConfirmationFunction();
    confirmation_function_map["CropAndResize"] = SimpleConfirmationFunction();
    confirmation_function_map["Cos"] = SimpleConfirmationFunction();
    confirmation_function_map["Cosh"] = SimpleConfirmationFunction();
    confirmation_function_map["Cumsum"] = SimpleConfirmationFunction();
    confirmation_function_map["DepthwiseConv2dNative"] = SimpleConfirmationFunction();
    confirmation_function_map["DepthToSpace"] = [](Node* n, bool* result) {
      std::string tf_data_format;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(n->attrs(), "data_format", &tf_data_format));
      *result = tf_data_format != "NCHW_VECT_C";
      return Status::OK();
    };
    confirmation_function_map["Equal"] = SimpleConfirmationFunction();
    confirmation_function_map["Exp"] = SimpleConfirmationFunction();
    confirmation_function_map["ExpandDims"] = SimpleConfirmationFunction();
    confirmation_function_map["Fill"] = SimpleConfirmationFunction();
    confirmation_function_map["Floor"] = SimpleConfirmationFunction();
    confirmation_function_map["FloorDiv"] = SimpleConfirmationFunction();
    confirmation_function_map["FusedBatchNorm"] = FusedBatchNormConfirmationFunction();
    confirmation_function_map["FusedBatchNormV2"] = FusedBatchNormConfirmationFunction();
    confirmation_function_map["FusedBatchNormV3"] = FusedBatchNormConfirmationFunction();
    confirmation_function_map["_FusedConv2D"] = SimpleConfirmationFunction();
    confirmation_function_map["GatherNd"] = SimpleConfirmationFunction();
    confirmation_function_map["GatherV2"] = SimpleConfirmationFunction();
    confirmation_function_map["_FusedMatMul"] =
        SimpleConfirmationFunction();  // TODO accept under all conditions?
                                       // check?
    confirmation_function_map["Greater"] = SimpleConfirmationFunction();
    confirmation_function_map["GreaterEqual"] = SimpleConfirmationFunction();
    confirmation_function_map["Identity"] = SimpleConfirmationFunction();
    confirmation_function_map["IsFinite"] = SimpleConfirmationFunction();
    confirmation_function_map["L2Loss"] = SimpleConfirmationFunction();
    confirmation_function_map["LogSoftmax"] = SimpleConfirmationFunction();
    confirmation_function_map["Less"] = SimpleConfirmationFunction();
    confirmation_function_map["LessEqual"] = SimpleConfirmationFunction();
    confirmation_function_map["Log"] = SimpleConfirmationFunction();
    confirmation_function_map["Log1p"] = SimpleConfirmationFunction();
    confirmation_function_map["LogicalAnd"] = SimpleConfirmationFunction();
    confirmation_function_map["LogicalNot"] = SimpleConfirmationFunction();
    confirmation_function_map["LogicalOr"] = SimpleConfirmationFunction();
    confirmation_function_map["MatMul"] = SimpleConfirmationFunction();
    confirmation_function_map["Max"] = SimpleConfirmationFunction();
    confirmation_function_map["Maximum"] = SimpleConfirmationFunction();
    confirmation_function_map["MaxPool"] = SimpleConfirmationFunction();
    confirmation_function_map["MaxPool3D"] = SimpleConfirmationFunction();
    confirmation_function_map["Mean"] = SimpleConfirmationFunction();
    confirmation_function_map["Min"] = SimpleConfirmationFunction();
    confirmation_function_map["Minimum"] = SimpleConfirmationFunction();
    confirmation_function_map["MirrorPad"] = SimpleConfirmationFunction();
    confirmation_function_map["Mul"] = SimpleConfirmationFunction();
    confirmation_function_map["Mod"] = SimpleConfirmationFunction();
    confirmation_function_map["Neg"] = SimpleConfirmationFunction();
    confirmation_function_map["NotEqual"] = SimpleConfirmationFunction();
    confirmation_function_map["NonMaxSuppressionV4"] = SimpleConfirmationFunction();
    confirmation_function_map["NoOp"] = SimpleConfirmationFunction();
    confirmation_function_map["OneHot"] = SimpleConfirmationFunction();
    confirmation_function_map["Pad"] = SimpleConfirmationFunction();
    confirmation_function_map["PadV2"] = SimpleConfirmationFunction();
    confirmation_function_map["Pow"] = SimpleConfirmationFunction();
    confirmation_function_map["PreventGradient"] = SimpleConfirmationFunction();
    confirmation_function_map["Prod"] = SimpleConfirmationFunction();
    confirmation_function_map["Rank"] = SimpleConfirmationFunction();
    confirmation_function_map["RealDiv"] = SimpleConfirmationFunction();
    confirmation_function_map["Reciprocal"] = SimpleConfirmationFunction();
    confirmation_function_map["Relu"] = SimpleConfirmationFunction();
    confirmation_function_map["Relu6"] = SimpleConfirmationFunction();
    confirmation_function_map["Reshape"] = SimpleConfirmationFunction();
    confirmation_function_map["Rsqrt"] = SimpleConfirmationFunction();
    confirmation_function_map["ScatterNd"] = SimpleConfirmationFunction();
    confirmation_function_map["Select"] = SimpleConfirmationFunction();
    confirmation_function_map["Shape"] = SimpleConfirmationFunction();
    confirmation_function_map["Sigmoid"] = SimpleConfirmationFunction();
    confirmation_function_map["Sign"] = SimpleConfirmationFunction();
    confirmation_function_map["Sin"] = SimpleConfirmationFunction();
    confirmation_function_map["Sinh"] = SimpleConfirmationFunction();
    confirmation_function_map["Size"] = SimpleConfirmationFunction();
    confirmation_function_map["Slice"] = SimpleConfirmationFunction();
    confirmation_function_map["Snapshot"] = SimpleConfirmationFunction();
    confirmation_function_map["Softmax"] = SimpleConfirmationFunction();
    confirmation_function_map["Softplus"] = SimpleConfirmationFunction();
    confirmation_function_map["SpaceToDepth"] = confirmation_function_map["DepthToSpace"];
    confirmation_function_map["Split"] = SimpleConfirmationFunction();
    confirmation_function_map["SplitV"] = SimpleConfirmationFunction();
    confirmation_function_map["Sqrt"] = SimpleConfirmationFunction();
    confirmation_function_map["Square"] = SimpleConfirmationFunction();
    confirmation_function_map["SquaredDifference"] = SimpleConfirmationFunction();
    confirmation_function_map["Squeeze"] = SimpleConfirmationFunction();
    confirmation_function_map["StridedSlice"] = SimpleConfirmationFunction();
    confirmation_function_map["Pack"] = SimpleConfirmationFunction();
    confirmation_function_map["Sub"] = SimpleConfirmationFunction();
    confirmation_function_map["Sum"] = SimpleConfirmationFunction();
    confirmation_function_map["Tan"] = SimpleConfirmationFunction();
    confirmation_function_map["Tanh"] = SimpleConfirmationFunction();
    confirmation_function_map["Tile"] = SimpleConfirmationFunction();
    confirmation_function_map["TopKV2"] = [](Node* n, bool* result) {
      bool sorted = true;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "sorted", &sorted));

      // sorted = false is not supported right now, it falls back to TF if set
      // to false.
      *result = sorted;
      return Status::OK();
    };
    confirmation_function_map["Transpose"] = SimpleConfirmationFunction();
    confirmation_function_map["Unpack"] = SimpleConfirmationFunction();
    confirmation_function_map["UnsortedSegmentSum"] = SimpleConfirmationFunction();
    confirmation_function_map["Xdivy"] = SimpleConfirmationFunction();
    confirmation_function_map["ZerosLike"] = SimpleConfirmationFunction();
    initialized = true;
  }
  return confirmation_function_map;
}

static bool IsOpModeSupportedTF(Node* node, std::map<std::string, ConfirmationFunction>& confirmation_function_map){
    // check node's confirmation constraints
    bool confirmation_constraint_ok = false;
    ConfirmationOk(node, confirmation_function_map,
                                    confirmation_constraint_ok);
    if (!confirmation_constraint_ok) {
        std::cout << "Node does not meet confirmation constraints: "
                << node->type_string() << std::endl;
    }
    return confirmation_constraint_ok;
}

static bool IsTypeSupported(tensorflow::Node* node, const TypeConstraintMap& type_constraint_map){

  bool type_constraints_ok=true;
  const auto& itr = type_constraint_map.find(node->type_string());
  if (itr != type_constraint_map.end()) {
    for (const auto& name_and_set : itr->second) {
      auto& type_attr_name = name_and_set.first;
      auto& allowed_types = name_and_set.second;

      DataType dt;

      if (GetNodeAttr(node->attrs(), type_attr_name, &dt) != Status::OK() ||
          std::find(allowed_types.begin(), allowed_types.end(), dt) ==
              allowed_types.end()) {
        type_constraints_ok = false;
        break;
      }
    }
  }

  return type_constraints_ok;
}

std::vector<void *> TFNodesChecker::PrepareSupportedNodesList(){
	std::vector<void *> node_list;

  // Get OV supported ops list for TF
	supported_ops = GetTFSupportedOPs(device_id, ov_version);
  std::cout <<"TF OPS list generated" <<"\n";
  
  // Get the op type map based in the input device_id
  const TypeConstraintMap& type_constraint_map = GetTypeConstraintMap();

  // Get the op mode confirmation map
  static std::map<std::string, ConfirmationFunction> confirmation_function_map = GetConfirmationMap();

	for (auto node : graph->op_nodes()){
		bool is_node_supported = true;
		// check if the optype supported
		do{
			// CHECK_1: if the op is supported
			is_node_supported &= IsOpSupported(node->type_string());
			if(is_node_supported == false){
			    std::cout << "Op: " << node->type_string() << " is not supported " << std::endl;
				break;
			}

			// CHECK_2: OP Type and Dimensions Check...
      is_node_supported &= IsTypeSupported(node, type_constraint_map);
      if(is_node_supported == false){
			    std::cout << "Op Type:" << node->type_string() << " is not supported " << std::endl;
  				break;
			}

			// CHECK_3: OP mode check based on attributes
      is_node_supported &= IsOpModeSupportedTF(node, confirmation_function_map);
      if(is_node_supported == false){
			    std::cout << "Op Mode:" << node->type_string() << " is not supported " << std::endl;
  				break;
			}
			// CHECK_4: TBD

		} while(false);
		if(is_node_supported){
			node_list.push_back((void *)node);
		}

	}
	return node_list;
}

} //namespace ocm 