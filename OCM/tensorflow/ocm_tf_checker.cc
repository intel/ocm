#include "ocm_tf_checker.h"
#include "ocm_tf_ops_list.h"

std::set<std::string> getTFSupportedOPs(std::string device_id, std::string ov_version){
	
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

static bool isOpModeSupportedTF(Node* node){
    static std::map<std::string, ConfirmationFunction> confirmation_function_map =
                                                        GetConfirmationMap();
                                                                // check node's confirmation constraints
    bool confirmation_constraint_ok = false;
    ConfirmationOk(node, confirmation_function_map,
                                    confirmation_constraint_ok);
    if (!confirmation_constraint_ok) {
        std::cout << "Node does not meet confirmation constraints: "
                << node->type_string() << std::endl;
        if (confirmation_function_map.find(node->type_string()) ==
            confirmation_function_map.end()) {
        // not found
        // no_support_histogram[node->type_string()]++;
        } else {
        // found
        // fail_confirmation_histogram[node->type_string()]++;
        }
    }
    return confirmation_constraint_ok;
}

std::vector<void *> TFNodesChecker::PrepareSupportedNodesList(){
	std::vector<void *> nodeList;

    // get OV supported ops list for TF
	supported_ops = getTFSupportedOPs(device_id, ov_version);
    std::cout <<"TF OPS list generated" <<"\n";

	for (auto node : graph->op_nodes()){
		bool isNodeSupported = true;
		// check if the optype supported
		do{
			// CHECK_1: if the op is supported
			isNodeSupported &= isOpSupported(node->type_string());
			if(isNodeSupported == false){
			    std::cout << "Op: " << node->type_string() << " is not supported " << std::endl;
				break;
			}

			// CHECK_2: OP Type and Dimensions Check...
			// confirmation_function_map is non-const unlike the other maps

			// CHECK_3: OP mode check based on attributes
            isNodeSupported &= isOpModeSupportedTF(node);
            if(isNodeSupported == false){
			    std::cout << "Op Mode:" << node->type_string() << " is not supported " << std::endl;
				break;
			}
			// CHECK_4: TBD

		} while(false);
		if(isNodeSupported){
			nodeList.push_back((void *)node);
		}
	}
	return nodeList;
}
/*
std::vector<const void *> TFNodesChecker::PrepareSupportedNodesList(){

	std::vector<const void *> nodeList;

	// get OV supported ops list for TF
	supported_ops = getTFSupportedOPs(device_id, ov_version);
    std::cout <<"TF OPS list generated" <<"\n";

	for (const auto node : graph->op_nodes()){
		bool isNodeSupported = true;
		// check if the optype supported
		do{
			// CHECK_1: if the op is supported
			isNodeSupported &= isOpSupported(node->type_string());
            std::cout << "Checkin for node: "<<node->type_string() << " , Supported: "<<isNodeSupported<<std::endl;
			if(isNodeSupported == false){break;}

			// CHECK_2: OP Type and Dimensions Check...
			
			// CHECK_3: OP mode check based on attributes

			// CHECK_4: TBD

		} while(false);
	}

    std::cout<< "All the OPS are supported "<<std::endl;
	return nodeList;
}
*/
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
        confirmation_function_map["BiasAdd"] = SimpleConfirmationFunction();
        confirmation_function_map["ConcatV2"] = SimpleConfirmationFunction();
        confirmation_function_map["Const"] = SimpleConfirmationFunction();
#if 0
        confirmation_function_map["Conv2D"] = [](Node* n, bool* result) {

                            std::string tf_data_format;
                           GetNodeAttr(n->attrs(), "data_format", &tf_data_format);
                            *result = tf_data_format == "NCHW"; //NHWC
                            std::cout << tf_data_format << "  and result is " << *result << std::endl;
                            return Status::OK();
                        };
#else
        confirmation_function_map["Conv2D"] = SimpleConfirmationFunction();
#endif   
        confirmation_function_map["Identity"] = SimpleConfirmationFunction();
        confirmation_function_map["LRN"] = SimpleConfirmationFunction();
        confirmation_function_map["MatMul"] = SimpleConfirmationFunction();
        confirmation_function_map["MaxPool"] = SimpleConfirmationFunction();
        confirmation_function_map["Relu"] = SimpleConfirmationFunction();
        confirmation_function_map["Reshape"] = SimpleConfirmationFunction();
        confirmation_function_map["Shape"] = SimpleConfirmationFunction();
        confirmation_function_map["Softmax"] = SimpleConfirmationFunction();    
        confirmation_function_map["Split"] = SimpleConfirmationFunction();
    }
    return confirmation_function_map;
}
