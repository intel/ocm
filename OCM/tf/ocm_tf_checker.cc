#include "ocm_tf_checker.h"
#include "ocm_tf_ops_list.h"

namespace ocm{

// Refer following page for type support: 
// https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html  
const std::set<DataType> SupportedTypes(const std::string device_id="CPU"){
  
  const std::set<DataType> cpu_supported_inputTypes = {
    DT_BFLOAT16,
    DT_HALF,
    DT_FLOAT,
    DT_INT8, 
    DT_INT16, 
    DT_INT32, 
    DT_INT64, 
    DT_UINT8, 
    };

  const std::set<DataType> gpu_supported_inputTypes = {
    DT_BFLOAT16,
    DT_HALF,  
    DT_FLOAT, 
    DT_INT8,
    };

  const std::set<DataType> myriad_supported_inputTypes = {
    DT_BFLOAT16, 
    DT_HALF, 
    DT_FLOAT
    };
  
  const std::set<DataType> hddl_supported_inputTypes = {
    DT_BFLOAT16,
    DT_HALF,  
    DT_FLOAT
    };
  
  if(device_id=="CPU"){
    return cpu_supported_inputTypes;
  } else if(device_id=="GPU"){
    return gpu_supported_inputTypes;
  } else if(device_id=="MYRIAD"){
    return myriad_supported_inputTypes;
  } else if(device_id=="HDDL"){
    return hddl_supported_inputTypes;
  }

  return cpu_supported_inputTypes;
}

const std::set<DataType> SupportedTypesIdx(const std::string device_id="CPU"){

  const std::set<DataType> cpu_supported_inputTypes= {DT_INT32, DT_INT64};

  if(device_id=="CPU"){
    return cpu_supported_inputTypes;
  }
  
  return cpu_supported_inputTypes;
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
    type_constraint_map["FloorMod"]["T"] = SupportedTypes();
    type_constraint_map["FusedBatchNorm"]["T"] = SupportedTypes();
    type_constraint_map["FusedBatchNormV3"]["T"] = SupportedTypes();
    type_constraint_map["_FusedConv2D"]["T"] = SupportedTypes(); // formed after TF optimization pass, not in original graph
    type_constraint_map["_FusedMatMul"]["T"] = SupportedTypes(); // formed after TF optimization pass, not in original graph
    type_constraint_map["Identity"]["T"] = SupportedTypes();
    type_constraint_map["Less"]["T"] = SupportedTypes();
    type_constraint_map["LogSoftmax"]["T"] = SupportedTypes();
    type_constraint_map["MatMul"]["T"] = SupportedTypes();
    type_constraint_map["MaxPool"]["T"] = SupportedTypes();
    type_constraint_map["Mean"]["T"] = SupportedTypes();
    type_constraint_map["Mean"]["Tidx"] = SupportedTypesIdx();    
    type_constraint_map["Mul"]["T"] = SupportedTypes();
    type_constraint_map["Pack"]["T"] = SupportedTypes();
    type_constraint_map["Pad"]["Tpaddings"] = SupportedTypes();
    type_constraint_map["Placeholder"]["dtype"] = SupportedTypes();
    type_constraint_map["Range"]["Tidx"] = SupportedTypesIdx();
    type_constraint_map["Relu"]["T"] = SupportedTypes();
    type_constraint_map["Reshape"]["T"] = SupportedTypes();
    type_constraint_map["Shape"]["T"] = SupportedTypes();
    type_constraint_map["Shape"]["out_type"] = SupportedTypesIdx(); 
    type_constraint_map["Slice"]["T"] = SupportedTypes(); // Added for unit tests
    type_constraint_map["Softmax"]["T"] = SupportedTypes();
    type_constraint_map["Split"]["T"] = SupportedTypes(); // For unit tests
    type_constraint_map["SplitV"]["T"] = SupportedTypes(); // For unit tests
    type_constraint_map["Sub"]["T"] = SupportedTypes();
    type_constraint_map["Squeeze"]["T"] = SupportedTypes();
    type_constraint_map["StridedSlice"]["T"] = SupportedTypes();
    type_constraint_map["StridedSlice"]["Index"] = SupportedTypesIdx();  
    type_constraint_map["Sub"]["T"] = SupportedTypes();  
    type_constraint_map["Tile"]["T"] = SupportedTypes(); // For unit tests
    type_constraint_map["Transpose"]["T"] = SupportedTypes();
    type_constraint_map["Transpose"]["Tperm"] = SupportedTypesIdx();
    type_constraint_map["Unpack"]["T"] = SupportedTypes(); // For unit tests
    type_constraint_map["ZerosLike"]["T"] = SupportedTypes(); // For unit tests
  }
  return type_constraint_map;
}

std::set<std::string> GetTFSupportedOPs(std::string device_id, std::string ov_version){
	
	std::set<std::string> supported_ops = {};

  if (device_id == "CPU") {
    // std::merge(common_supported_ops.begin(), common_supported_ops.end(),
    //            cpu_only_ops.begin(), cpu_only_ops.end(),
    //            std::inserter(supported_ops,supported_ops.begin()));
    supported_ops.insert(common_supported_ops.begin(), common_supported_ops.end());
    supported_ops.insert(cpu_only_ops.begin(), cpu_only_ops.end());
    supported_ops.insert(composite_ops.begin(), composite_ops.end());
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

static Status ValidateInputCountMin(const Node* op, tensorflow::int32 count, bool* result) {
  if (op->num_inputs() < count) {
    *result = false;
    std::cout<<"\""<< op->name()<< "\" requires at least "<<
                                   count<< " input(s), got "<< op->num_inputs()<<
                                   " instead";
  }
  *result = true;
  return Status::OK();
}

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
    confirmation_function_map["Add"] = SimpleConfirmationFunction();
    confirmation_function_map["AddN"] = SimpleConfirmationFunction();
    confirmation_function_map["AddV2"] = SimpleConfirmationFunction();
    confirmation_function_map["All"] = SimpleConfirmationFunction();
    confirmation_function_map["ArgMax"] = SimpleConfirmationFunction();
    confirmation_function_map["AvgPool"] = SimpleConfirmationFunction();
    confirmation_function_map["BiasAdd"] = SimpleConfirmationFunction();
    confirmation_function_map["Ceil"] = SimpleConfirmationFunction();
    confirmation_function_map["ConcatV2"] = SimpleConfirmationFunction();
    confirmation_function_map["Const"] = SimpleConfirmationFunction();
    confirmation_function_map["Conv2D"] = SimpleConfirmationFunction();
    confirmation_function_map["FloorMod"] = SimpleConfirmationFunction();
    confirmation_function_map["FusedBatchNorm"] = FusedBatchNormConfirmationFunction();
    confirmation_function_map["FusedBatchNormV3"] = FusedBatchNormConfirmationFunction();
    confirmation_function_map["_FusedConv2D"] = SimpleConfirmationFunction();
    confirmation_function_map["_FusedMatMul"] = SimpleConfirmationFunction();  
    confirmation_function_map["Identity"] = SimpleConfirmationFunction();
    confirmation_function_map["Less"] = SimpleConfirmationFunction();
    confirmation_function_map["LogSoftmax"] = SimpleConfirmationFunction();
    confirmation_function_map["MatMul"] = SimpleConfirmationFunction();
    confirmation_function_map["MaxPool"] = SimpleConfirmationFunction();
    confirmation_function_map["Mean"] = SimpleConfirmationFunction();
    confirmation_function_map["Mul"] = SimpleConfirmationFunction();
    confirmation_function_map["Pack"] = [](Node* n, bool* result) {
      // num of inputs
      tensorflow::int32 count = 1;
      TF_RETURN_IF_ERROR(ValidateInputCountMin(n, count, result));
      return tensorflow::Status::OK();
    };
    confirmation_function_map["Pad"] = SimpleConfirmationFunction();
    confirmation_function_map["Placeholder"] = SimpleConfirmationFunction();
    confirmation_function_map["Relu"] = SimpleConfirmationFunction();
    confirmation_function_map["Range"] = SimpleConfirmationFunction();
    confirmation_function_map["Reshape"] = SimpleConfirmationFunction();
    confirmation_function_map["Shape"] = SimpleConfirmationFunction();
    confirmation_function_map["Slice"] = SimpleConfirmationFunction(); // For unit tests
    confirmation_function_map["Softmax"] = SimpleConfirmationFunction();
    confirmation_function_map["Split"] = SimpleConfirmationFunction(); // For unit tests
    confirmation_function_map["SplitV"] = SimpleConfirmationFunction(); // For unit tests
    confirmation_function_map["Squeeze"] = SimpleConfirmationFunction();
    confirmation_function_map["StridedSlice"] = SimpleConfirmationFunction();
    confirmation_function_map["Sub"] = SimpleConfirmationFunction();
    confirmation_function_map["Tile"] = SimpleConfirmationFunction(); // For unit tests
    confirmation_function_map["Transpose"] = SimpleConfirmationFunction();
    confirmation_function_map["Unpack"] = SimpleConfirmationFunction(); // For unit tests
    confirmation_function_map["ZerosLike"] = SimpleConfirmationFunction(); // For unit tests
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
  //std::cout <<"TF OPS list generated" <<"\n";
  
  // Get the op type map based in the input device_id
  const TypeConstraintMap& type_constraint_map = GetTypeConstraintMap();

  // Get the op mode confirmation map
  static std::map<std::string, ConfirmationFunction> confirmation_function_map = GetConfirmationMap();

	for (auto node : graph->op_nodes()){
		bool is_node_supported = true;
		// check if the optype supported
		do{
      // CHECK for the static Ops _Arg and _Retval
      if(node->type_string()=="_Arg" || node->type_string()=="_Retval"){
        // no further checks, as these are ops related to input and output
        break;
      }

			// CHECK_1: if the op is supported
			is_node_supported &= IsOpSupported(node->type_string());
			if(is_node_supported == false){
			    std::cout << "Op: " << node->type_string() << " is not supported " << std::endl;
				break;
			}

			// CHECK_2: OP Type and Dimensions Check...
      is_node_supported &= IsTypeSupported(node, type_constraint_map);
      if(is_node_supported == false){
			    std::cout << "Op Type: " << node->type_string() << " is not supported " << std::endl;
  				break;
			}

			// CHECK_3: OP mode check based on attributes
      is_node_supported &= IsOpModeSupportedTF(node, confirmation_function_map);
      if(is_node_supported == false){
			    std::cout << "Op Mode: " << node->type_string() << " is not supported " << std::endl;
  				break;
			}

		} while(false);
		if(is_node_supported){
			node_list.push_back((void *)node);
		}
         
	}
	return node_list;
}

} //namespace ocm 