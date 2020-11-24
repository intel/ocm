#include "ocm_tf_checker.h"
#include "ocm_tf_ops_list.h"

namespace ocm{

/**
 *  Refer following page for type support: 
 *  https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html  
 *  @return supported datatypes set based on device_id
 */
const std::set<DataType> SupportedTypes(const std::string device_id="CPU"){
  
  const std::set<DataType> cpu_supported_inputTypes = {
    DT_FLOAT,
    //DT_INT8, 
    DT_INT16,
    DT_INT32, 
    DT_INT64, 
    DT_UINT8, 
    DT_UINT16, 
    };

  const std::set<DataType> gpu_supported_inputTypes = {
    DT_BFLOAT16,
    DT_HALF,  
    DT_FLOAT, 
    DT_UINT8
    //DT_INT8,
    };

  const std::set<DataType> myriad_supported_inputTypes = {
    DT_BFLOAT16, 
    DT_HALF, 
    DT_FLOAT,
    DT_UINT8
    };
  
  const std::set<DataType> hddl_supported_inputTypes = {
    DT_BFLOAT16,
    DT_HALF,  
    DT_FLOAT,
    DT_UINT8
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

/**
 *  Refer following page for type support: 
 *  https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html  
 *  @return supported datatypes set for the Indices attributes based on device_id
 */
const std::set<DataType> SupportedTypesIdx(const std::string device_id="CPU"){

  const std::set<DataType> cpu_supported_inputTypes= {DT_INT32, DT_INT64};

  if(device_id=="CPU"){
    return cpu_supported_inputTypes;
  }
  
  return cpu_supported_inputTypes;
}

/**
 * Generates Type (Data Type) constraints map for all the Tensorflow variables
 * @return a map with key as "opname" string and value is again a map with key as
 * TF data type notation string and value as a set of tensorflow datatypes
 */
const TypeConstraintMap& GetTypeConstraintMap(std::string device_id) {
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
    type_constraint_map["Abs"]["T"] = SupportedTypes(); //cwise_math
    type_constraint_map["Acos"]["T"] = SupportedTypes(); //cwise_math
    type_constraint_map["Acosh"]["T"] = SupportedTypes(); //cwise_math
    type_constraint_map["Add"]["T"] = SupportedTypes();
    type_constraint_map["AddN"]["T"] = SupportedTypes();
    type_constraint_map["AddV2"]["T"] = SupportedTypes();
    type_constraint_map["ArgMax"]["T"] = SupportedTypes();
    type_constraint_map["ArgMax"]["Tidx"] = SupportedTypesIdx();
    type_constraint_map["Asin"]["T"] = SupportedTypes(); //cwise_math
    type_constraint_map["Asinh"]["T"] = SupportedTypes(); //cwise_math
    type_constraint_map["Atan"]["T"] = SupportedTypes(); //cwise_math
    type_constraint_map["Atanh"]["T"] = SupportedTypes(); //cwise_math
    type_constraint_map["ArgMin"]["T"] = SupportedTypes();
    type_constraint_map["ArgMin"]["Tidx"] = SupportedTypesIdx();
    type_constraint_map["AvgPool"]["T"] = SupportedTypes();
    type_constraint_map["BiasAdd"]["T"] = SupportedTypes();
    type_constraint_map["Cast"]["SrcT"] = SupportedTypes();
    type_constraint_map["Cast"]["DstT"] = SupportedTypes();
    type_constraint_map["ConcatV2"]["T"] = SupportedTypes();
    type_constraint_map["ConcatV2"]["Tidx"] = SupportedTypesIdx();    
    type_constraint_map["Const"]["dtype"] = SupportedTypes();
    type_constraint_map["Conv2D"]["T"] = SupportedTypes();
    type_constraint_map["Conv2DBackpropInput"]["T"] = SupportedTypes();
    type_constraint_map["CropAndResize"]["T"] = SupportedTypes();
    type_constraint_map["CropAndResize"]["extrapolation_value"] = {DT_FLOAT};
    type_constraint_map["ExpandDims"]["T"] = SupportedTypes();
    type_constraint_map["Fill"]["T"] = SupportedTypes();
    type_constraint_map["Fill"]["index_type"] = SupportedTypesIdx();
    type_constraint_map["FloorMod"]["T"] = SupportedTypes();
    type_constraint_map["FloorDiv"]["T"] = SupportedTypes();
    type_constraint_map["FusedBatchNorm"]["T"] = SupportedTypes();
    type_constraint_map["FusedBatchNormV3"]["T"] = SupportedTypes();
    type_constraint_map["_FusedConv2D"]["T"] = SupportedTypes(); // formed after TF optimization pass, not in original graph
    type_constraint_map["_FusedMatMul"]["T"] = SupportedTypes(); // formed after TF optimization pass, not in original graph
    type_constraint_map["Gather"]["Tparams"] = SupportedTypes();
    type_constraint_map["Gather"]["Tindices"] = SupportedTypesIdx();
    type_constraint_map["Gather"]["Taxis"] = SupportedTypesIdx();
    type_constraint_map["GatherV2"]["Tparams"] = SupportedTypes();
    type_constraint_map["GatherV2"]["Tindices"] = SupportedTypesIdx();
    type_constraint_map["GatherV2"]["Taxis"] = SupportedTypesIdx();
    type_constraint_map["Greater"]["T"] = SupportedTypes(); //cwise_math    
    type_constraint_map["GreaterEqual"]["T"] = SupportedTypes(); 
    type_constraint_map["Identity"]["T"] = SupportedTypes();
    // LRN: If input is of type other then the mentioned types, TF itself throws an error
    // For other attributes TF automatically typecasts them to required types
    type_constraint_map["LRN"]["T"] = {DT_BFLOAT16, DT_HALF, DT_FLOAT};
    type_constraint_map["Less"]["T"] = SupportedTypes();
    type_constraint_map["LogSoftmax"]["T"] = SupportedTypes();
    type_constraint_map["MatMul"]["T"] = SupportedTypes();
    type_constraint_map["MaxPool"]["T"] = SupportedTypes();
    type_constraint_map["Mean"]["T"] = SupportedTypes();
    type_constraint_map["Mean"]["Tidx"] = SupportedTypesIdx();    
    type_constraint_map["MirrorPad"]["T"] = SupportedTypes();  // For unit tests  
    type_constraint_map["MirrorPad"]["Tpaddings"] = SupportedTypesIdx();  // For unit tests   
    type_constraint_map["Mul"]["T"] = SupportedTypes();
    type_constraint_map["Neg"]["T"] = SupportedTypes(); //cwise_math    
    type_constraint_map["OneHot"]["axis"] = {DT_INT32, DT_INT64};
    type_constraint_map["OneHot"]["T"] = SupportedTypes();
    type_constraint_map["OneHot"]["TI"] = SupportedTypes();
    type_constraint_map["Pack"]["T"] = SupportedTypes();
    type_constraint_map["Pad"]["Tpaddings"] = SupportedTypesIdx();
    type_constraint_map["PadV2"]["T"] = SupportedTypes();
    type_constraint_map["PadV2"]["Tpaddings"] = SupportedTypesIdx();
    type_constraint_map["Placeholder"]["dtype"] = SupportedTypes();
    type_constraint_map["Range"]["Tidx"] = SupportedTypesIdx();
    type_constraint_map["RealDiv"]["T"] = SupportedTypes(); //cwise_math    
    type_constraint_map["Relu"]["T"] = SupportedTypes();
    type_constraint_map["Relu6"]["T"] = SupportedTypes();
    type_constraint_map["Reshape"]["T"] = SupportedTypes();
    type_constraint_map["ResizeBilinear"]["T"] = SupportedTypes();
    type_constraint_map["Rsqrt"]["T"] = SupportedTypes();
    type_constraint_map["Shape"]["T"] = SupportedTypes();
    type_constraint_map["Shape"]["out_type"] = SupportedTypesIdx(); 
    type_constraint_map["Sigmoid"]["T"] = SupportedTypes(); //cwise_math
    type_constraint_map["Sign"]["T"] = SupportedTypes(); //cwise_math
    type_constraint_map["Sinh"]["T"] = SupportedTypes(); //cwise_math
    type_constraint_map["Size"]["T"] = SupportedTypes(); 
    type_constraint_map["Size"]["out_type"] = SupportedTypesIdx(); 
    type_constraint_map["Slice"]["T"] = SupportedTypes();
    type_constraint_map["Softmax"]["T"] = SupportedTypes();
    type_constraint_map["SpaceToDepth"]["T"] = SupportedTypes();
    type_constraint_map["Split"]["T"] = SupportedTypes();
    type_constraint_map["SplitV"]["T"] = SupportedTypes(); // For unit tests
    type_constraint_map["Sub"]["T"] = SupportedTypes();
    type_constraint_map["Squeeze"]["T"] = SupportedTypes();
    type_constraint_map["StridedSlice"]["T"] = SupportedTypes();
    type_constraint_map["StridedSlice"]["Index"] = SupportedTypesIdx();  
    type_constraint_map["Sub"]["T"] = SupportedTypes();  
    type_constraint_map["Sum"]["T"] = SupportedTypes(); //cwise_math    
    type_constraint_map["Tanh"]["T"] = SupportedTypes(); //cwise_math    
    type_constraint_map["Tile"]["T"] = SupportedTypes(); 
    type_constraint_map["TopKV2"]["T"] = SupportedTypes(); 
    type_constraint_map["Transpose"]["T"] = SupportedTypes();
    type_constraint_map["Transpose"]["Tperm"] = SupportedTypesIdx();
    type_constraint_map["Unpack"]["T"] = SupportedTypes();
    type_constraint_map["ZerosLike"]["T"] = SupportedTypes();
  }
  return type_constraint_map;
}

std::set<std::string> GetTFSupportedOPs(std::string device_id, std::string ov_version){
	
	std::set<std::string> supported_ops = {};

  if (device_id == "CPU") {
    supported_ops.insert(common_supported_ops.begin(), common_supported_ops.end());
    supported_ops.insert(cpu_only_ops.begin(), cpu_only_ops.end());
    supported_ops.insert(composite_ops.begin(), composite_ops.end());
  } else if (device_id == "GPU") {
    supported_ops.insert(common_supported_ops.begin(), common_supported_ops.end());
    supported_ops.insert(gpu_only_ops.begin(), gpu_only_ops.end());
    supported_ops.insert(composite_ops.begin(), composite_ops.end());
  } else if (device_id == "MYRIAD" || device_id == "HDDL") {
    supported_ops.insert(common_supported_ops.begin(), common_supported_ops.end());
    supported_ops.insert(vpu_only_ops.begin(), vpu_only_ops.end());
    supported_ops.insert(composite_ops.begin(), composite_ops.end());
  }
  return supported_ops;
}

// Checks if the node meets the confirmation constraints
static Status ConfirmationOk( tensorflow::Node* node,
                              std::map<std::string, ConfirmationFunction>& confirmation_function_map,
                              bool& confirmation_ok) {
    auto it = confirmation_function_map.find(node->type_string());
    if (it != confirmation_function_map.end()) {
        TF_RETURN_IF_ERROR(it->second(node, &confirmation_ok));
    }
    return Status::OK();
}

// Implements a specific constraint on the input ops count
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

// Generates confirmation function for fused BN as it requires separate checks
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

/**
 * Generates constraints map for all the Tensorflow Ops which check
 * all the attributes
 * @return a map with key as opname string and value as confirmation function 
 */
const std::map<std::string, ConfirmationFunction>& GetConfirmationMap(std::string device_id) {
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
    confirmation_function_map["Abs"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Acos"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Acosh"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Add"] = SimpleConfirmationFunction();
    confirmation_function_map["AddN"] = SimpleConfirmationFunction();
    confirmation_function_map["AddV2"] = SimpleConfirmationFunction();
    confirmation_function_map["All"] = SimpleConfirmationFunction();
    confirmation_function_map["ArgMax"] = [device_id](Node* n, bool* result) {
      *result = true;
      // only Float32 input type is supported
      DataType dt;
      std::string type_attr_name = "T";
      if (GetNodeAttr(n->attrs(), type_attr_name, &dt)!= Status::OK() || dt!=DT_FLOAT){
        *result = false;
        return tensorflow::Status::OK();
      };
      return tensorflow::Status::OK();
    };
    
    confirmation_function_map["ArgMin"] = confirmation_function_map["ArgMax"];
    confirmation_function_map["Asin"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Asinh"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Atan"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Atanh"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["AvgPool"] = SimpleConfirmationFunction();
    confirmation_function_map["BiasAdd"] = SimpleConfirmationFunction();
    confirmation_function_map["Cast"] = SimpleConfirmationFunction();
    confirmation_function_map["Ceil"] = SimpleConfirmationFunction();
    confirmation_function_map["ConcatV2"] = SimpleConfirmationFunction();
    confirmation_function_map["Const"] = SimpleConfirmationFunction();
    confirmation_function_map["Conv2D"] = SimpleConfirmationFunction();
    confirmation_function_map["Conv2DBackpropInput"] = SimpleConfirmationFunction();
    confirmation_function_map["CropAndResize"] = [](Node* n, bool* result) {
      // Currently OpenVINO supports on "bilinear" method for CropAndResize
      *result = true;
      std::string resize_method;
      std::string type_attr_name = "method";
      if (GetNodeAttr(n->attrs(), type_attr_name, &resize_method)!= Status::OK() || resize_method!="bilinear"){
        *result = false;
      };
      return tensorflow::Status::OK();
    };
    confirmation_function_map["ExpandDims"] = SimpleConfirmationFunction();
    confirmation_function_map["Fill"] = SimpleConfirmationFunction();
    confirmation_function_map["FloorMod"] = SimpleConfirmationFunction();
    confirmation_function_map["FloorDiv"] = SimpleConfirmationFunction();
    confirmation_function_map["FusedBatchNorm"] = FusedBatchNormConfirmationFunction();
    confirmation_function_map["FusedBatchNormV3"] = FusedBatchNormConfirmationFunction();
    confirmation_function_map["_FusedConv2D"] = SimpleConfirmationFunction();
    confirmation_function_map["_FusedMatMul"] = SimpleConfirmationFunction();  
    confirmation_function_map["Gather"] = SimpleConfirmationFunction();
    confirmation_function_map["GatherV2"] = SimpleConfirmationFunction();  
    confirmation_function_map["Greater"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["GreaterEqual"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Identity"] = SimpleConfirmationFunction();
    confirmation_function_map["LRN"] = SimpleConfirmationFunction();
    confirmation_function_map["Less"] = SimpleConfirmationFunction();
    confirmation_function_map["LogSoftmax"] = SimpleConfirmationFunction();
    confirmation_function_map["MatMul"] = SimpleConfirmationFunction();
    confirmation_function_map["MaxPool"] = SimpleConfirmationFunction();
    confirmation_function_map["Mean"] = SimpleConfirmationFunction();
    confirmation_function_map["MirrorPad"] = SimpleConfirmationFunction(); // For unit tests
    confirmation_function_map["Mul"] = SimpleConfirmationFunction();
    confirmation_function_map["Neg"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["OneHot"] = SimpleConfirmationFunction();
    confirmation_function_map["Pack"] = [](Node* n, bool* result) {
      // num of inputs
      tensorflow::int32 count = 1;
      TF_RETURN_IF_ERROR(ValidateInputCountMin(n, count, result));
      return tensorflow::Status::OK();
    };
    confirmation_function_map["Pad"] = SimpleConfirmationFunction();
    confirmation_function_map["PadV2"] = SimpleConfirmationFunction();
    confirmation_function_map["Placeholder"] = SimpleConfirmationFunction();
    confirmation_function_map["Range"] = SimpleConfirmationFunction();
    confirmation_function_map["RealDiv"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Relu"] = SimpleConfirmationFunction();
    confirmation_function_map["Relu6"] = SimpleConfirmationFunction();
    confirmation_function_map["Reshape"] = SimpleConfirmationFunction();
    confirmation_function_map["ResizeBilinear"] = SimpleConfirmationFunction();
    confirmation_function_map["Rsqrt"] = SimpleConfirmationFunction();
    confirmation_function_map["Sigmoid"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Sign"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Sinh"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Shape"] = SimpleConfirmationFunction();
    confirmation_function_map["Size"] = SimpleConfirmationFunction();
    confirmation_function_map["Slice"] = SimpleConfirmationFunction();
    confirmation_function_map["Softmax"] = SimpleConfirmationFunction();
    confirmation_function_map["SpaceToDepth"] = SimpleConfirmationFunction();
    // TF itself throws an error if the num of dimensions at "split_dim" axis is not completely 
    // divisible by "num_split" value 
    confirmation_function_map["Split"] = SimpleConfirmationFunction();
    confirmation_function_map["SplitV"] = SimpleConfirmationFunction();
    confirmation_function_map["Squeeze"] = [](Node* n, bool* result) {
        std::vector<int32> tf_axis;
        GetNodeAttr(n->attrs(), "squeeze_dims", &tf_axis);
//        std::cout << "sqeeze_dim size " <<  tf_axis.size() << std::endl;
        *result = true;
        //If Squeeze_dim is not provided do additional chhecks. 
        if(tf_axis.size() == 0){
            Node* tf_input;
            int input_idx = 0;
            TF_RETURN_IF_ERROR(n->input_node(input_idx, &tf_input));
            TensorShape t;
            *result = false;
            TF_RETURN_IF_ERROR(GetNodeAttr(tf_input->attrs(), "shape", &t));
            for (int i=0; i < t.dims(); i++){
                if(t.dim_size(i) == 1)
                    *result = true;
            }
        }
        return tensorflow::Status::OK();
    };
    confirmation_function_map["StridedSlice"] =[device_id](Node* n, bool* result) {
        *result = true;
        // First dimension of the input cannot be zero
        Node* tf_input_node;
        int input_idx = 3;
        
        TF_RETURN_IF_ERROR(n->input_node(input_idx, &tf_input_node));
        if(tf_input_node->type_string() ==  "Const"){
            // get stride  values
            Tensor values;
            TF_RETURN_IF_ERROR(GetNodeAttr(tf_input_node->attrs(), "value", &values));

            // Stride values are not specified exit
            for (int i=0; i < values.dims(); i++){
                if(values.dim_size(i) == 0){
                    *result = false;
                    std::cout << " ERROR : " << n->type_string() << " Op has empty Stride values." << std::endl;
                    return tensorflow::Status::OK();
                }
            }
            //From MC. Need to check/create a test case for this.
            auto array = values.data();
            int* int_array = static_cast<int*>(array);
            for(int i=0; i< values.NumElements() ;i++){
                if(*(int_array+i) < 0){
                    *result = false;
                    std::cout << " ERROR : " << n->type_string() << " Op has negative Stride value." << std::endl;
                    return tensorflow::Status::OK();
                }
            }
        }
        return tensorflow::Status::OK();
    };
    confirmation_function_map["Sub"] = SimpleConfirmationFunction();
    confirmation_function_map["Sum"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Tanh"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Tile"] = SimpleConfirmationFunction();
    confirmation_function_map["TopKV2"] = SimpleConfirmationFunction();
    confirmation_function_map["Transpose"] = SimpleConfirmationFunction();
    confirmation_function_map["Unpack"] = SimpleConfirmationFunction();
    confirmation_function_map["ZerosLike"] = SimpleConfirmationFunction();
    initialized = true;
  }
  return confirmation_function_map;
}

// Check node's confirmation constraints
static bool IsOpModeSupportedTF(Node* node, std::map<std::string, ConfirmationFunction>& confirmation_function_map){
    bool confirmation_constraint_ok = false;
    ConfirmationOk(node, confirmation_function_map,
                                    confirmation_constraint_ok);
    if (!confirmation_constraint_ok) {
        std::cout << " ERROR : Node does not meet confirmation constraints: "
                << node->type_string() << std::endl;
    }
    return confirmation_constraint_ok;
}

// Check node's Type (data type) constraints
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
        std::cout<<node->type_string()<<" "<<type_attr_name<<": "<<dt<<"\n";
        type_constraints_ok = false;
        break;
      }
    }
  } else{
    type_constraints_ok = false;
  }

  return type_constraints_ok;
}

static bool IsOpInputDimZeroTF(tensorflow::Node* node){
    bool is_input_dim_zero = true;
    int num_ips = node->num_inputs();
    for(int input_idx=0; input_idx < num_ips; input_idx++){
        Node* tf_input_node;
        if(node->input_node(input_idx, &tf_input_node) == Status::OK()){
            if(node->type_string() == "Max" || node->type_string() == "Mean" || 
               node->type_string() == "Sum"  || node->type_string() == "EuclideanNorm"){
                Tensor t;
                if(GetNodeAttr(tf_input_node->attrs(), "value", &t) == Status::OK()){
                    std::cout << " Dim is " << t.dims() << " DimSize(0) " << t.dim_size(0) << std::endl;
                    // if(t.dims() == 0){
                    //     is_input_dim_zero &= false;
                    //     return is_input_dim_zero;
                    // }
                    // Check dim of any of the input is ZERO.
                    for (int index=0; index<t.dims(); index++){
                        if (t.dim_size(index)==0){
                            is_input_dim_zero &= false;
                            // no further checks required, return from the function
                            return is_input_dim_zero;
                        }
                    }
                }
            }
            // ToDo: Added Placeholder for now. Not needed
            if(tf_input_node->type_string() != "Const" && 
                    tf_input_node->type_string() != "ConcatV2"){ 
                TensorShape t;

                if(GetNodeAttr(tf_input_node->attrs(), "shape", &t) == Status::OK()){
                    if(t.dims() == 0){
                        is_input_dim_zero &= false;
                        return is_input_dim_zero;
                    }
                    // Check dim of any of the input is ZERO.
                    for (int index=0; index<t.dims(); index++){
                        if (t.dim_size(index)==0){
                            is_input_dim_zero &= false;
                            // no further checks required, return from the function
                            return is_input_dim_zero;
                        }
                    }
                }
            }
        }
    }
    return is_input_dim_zero;
}
std::vector<void *> TFNodesChecker::PrepareSupportedNodesList(){

  std::vector<void *> node_list;

  // Get OV supported ops list for TF
  supported_ops = GetTFSupportedOPs(device_id, ov_version);
  //std::cout <<"TF OPS list generated" <<"\n";
  
  // Get the op type map based in the input device_id
  const TypeConstraintMap& type_constraint_map = GetTypeConstraintMap(device_id);

  // Get the op mode confirmation map
  static std::map<std::string, ConfirmationFunction> confirmation_function_map = GetConfirmationMap(device_id);

    for (auto node : graph->op_nodes()){
        bool is_node_supported = true;
        // check if the optype supported
        do{
            // CHECK_1: if the op is supported
            is_node_supported &= IsOpSupported(node->type_string());
            if(is_node_supported == false){
                std::cout << " ERROR : " << node->type_string() << " Op is not supported " << std::endl;
                break;
            }

                    // CHECK_2: OP Type and Dimensions Check...
            is_node_supported &= IsTypeSupported(node, type_constraint_map);
            if(is_node_supported == false){
                std::cout << " ERROR : " << node->type_string() << " Op Type is not supported " << std::endl;
                break;
            }

            // CHECK_3: OP mode check based on attributes
            is_node_supported &= IsOpModeSupportedTF(node, confirmation_function_map);
            if(is_node_supported == false){
                std::cout << " ERROR : " << node->type_string() << " Op Mode is not supported " << std::endl;
                break;
            }

            // Input dimension check
            is_node_supported &= IsOpInputDimZeroTF(node);
            if(is_node_supported == false){
                std::cout << " ERROR : " << node->type_string() << " Op - Input node Dim is ZERO " << std::endl;
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