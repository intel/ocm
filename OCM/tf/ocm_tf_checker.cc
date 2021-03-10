/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include "ocm_tf_checker.h"
#include "ocm_tf_ops_list.h"
#include "ocm_logging.h"

namespace ocm{

/**
 *  Refer following page for type support: 
 *  https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html  
 *  @return supported datatypes set based on device_id
 */
const std::set<DataType> SupportedTypes(const std::string device_id="CPU"){
  
  const std::set<DataType> cpu_supported_inputTypes = {
    DT_FLOAT,
    DT_INT16,
    DT_INT32, 
    DT_INT64, 
    DT_UINT8, 
    DT_UINT16, 
    };

  const std::set<DataType> gpu_supported_inputTypes = {
    DT_BFLOAT16,
#ifdef ENABLE_DT_HALF    
    DT_HALF, 
#endif
    DT_INT32,
    DT_FLOAT,
    DT_UINT8
    };

  const std::set<DataType> myriad_supported_inputTypes = {
    DT_BFLOAT16, 
#ifdef ENABLE_DT_HALF    
    DT_HALF, 
#endif
    DT_FLOAT,
    DT_INT32,
    DT_UINT8
    };
  
  const std::set<DataType> hddl_supported_inputTypes = {
    DT_BFLOAT16, 
#ifdef ENABLE_DT_HALF    
    DT_HALF, 
#endif
    DT_FLOAT,
    DT_INT32,  
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
const TypeConstraintMap& GetTypeConstraintMap(std::string device_id, std::string ov_version) {
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
    type_constraint_map["Abs"]["T"] = [device_id, ov_version](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
        if(ov_version == "2021.1"){
          supported_types.erase(DT_UINT32);
          supported_types.erase(DT_UINT64);
        }
      }
      return supported_types;
    }();
    type_constraint_map["Acos"]["T"] = SupportedTypes(device_id); //cwise_math
    type_constraint_map["Acosh"]["T"] = SupportedTypes(device_id); //cwise_math
    type_constraint_map["Add"]["T"] = SupportedTypes(device_id);
    type_constraint_map["AddN"]["T"] = SupportedTypes(device_id);
    type_constraint_map["AddV2"]["T"] = SupportedTypes(device_id);
    // need not to put input any constraint on input tensor, TF by default make sure the
    // the input tensor is of type bool, otherwise throws an error
    type_constraint_map["All"]["Tidx"] = SupportedTypesIdx(device_id);
    type_constraint_map["ArgMax"]["T"] = [device_id]() {
      // only Float32 input type is supported
      std::set<DataType> supported_types = {DT_FLOAT};
      if (device_id=="GPU"){
        supported_types.insert(DT_INT32);
      }
      return supported_types;
    }();
    type_constraint_map["ArgMax"]["Tidx"] = SupportedTypesIdx(device_id);
    type_constraint_map["ArgMin"]["T"] = [device_id]() {
      // only Float32 input type is supported
      std::set<DataType> supported_types = {DT_FLOAT};
      if (device_id=="GPU"){
        supported_types.insert(DT_INT32);
      }
      return supported_types;
    }();
    type_constraint_map["ArgMin"]["Tidx"] = SupportedTypesIdx(device_id);
    type_constraint_map["Asin"]["T"] = SupportedTypes(device_id); //cwise_math
    type_constraint_map["Asinh"]["T"] = SupportedTypes(device_id); //cwise_math
    type_constraint_map["Atan"]["T"] = SupportedTypes(device_id); //cwise_math
    type_constraint_map["Atanh"]["T"] = SupportedTypes(device_id); //cwise_math
    type_constraint_map["AvgPool"]["T"] =  [device_id, ov_version](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
#ifdef ENABLE_DT_HALF    
        supported_types.insert(DT_HALF);
#endif
      }
      return supported_types;
    }();
    type_constraint_map["BiasAdd"]["T"] = SupportedTypes(device_id);
    type_constraint_map["BatchToSpaceND"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Cast"]["SrcT"] =  [device_id, ov_version](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
#ifdef ENABLE_DT_HALF    
        supported_types.insert(DT_HALF);
#endif
      }
      else if (device_id=="MYRIAD" || device_id=="HDDL"){
        // checked using bridge code, it's working 
        supported_types.insert(DT_UINT16);
      }
      return supported_types;
    }();
    type_constraint_map["Cast"]["DstT"] =  [device_id, ov_version](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
#if ENABLE_DT_HALF      
        supported_types.insert(DT_HALF);
#endif
    }
      //DT_INT64 is supported by HDDL inferencing
      else if (device_id=="HDDL"){
        supported_types.insert(DT_INT64);
      }
      return supported_types;
    }();
    type_constraint_map["ConcatV2"]["T"] = SupportedTypes(device_id);
    type_constraint_map["ConcatV2"]["Tidx"] = SupportedTypesIdx(device_id);    
    type_constraint_map["Const"]["dtype"] = [device_id](){
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
        //modified as test cases with bridge were failing, though CPU
        //supports DT_STRING, so could be a data type issue on the bridge side too
        // supported_types={DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_BOOL, DT_STRING}; 
        supported_types={DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_BOOL}; 
      }
      else if (device_id=="MYRIAD" || device_id=="HDDL"){
        supported_types.insert(DT_INT64);
        // checked using bridge code, it's working 
        supported_types.insert(DT_UINT16);
      }
      else if (device_id=="GPU"){
        supported_types.insert(DT_INT64);
      }   

      return supported_types;
    }();
    type_constraint_map["Conv2D"]["T"] = [device_id](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="MYRIAD" || device_id=="HDDL"){
        supported_types.erase(DT_INT32);    
      }
      return supported_types;
    }();
    type_constraint_map["Conv2DBackpropInput"]["T"] = SupportedTypes(device_id);
    type_constraint_map["CropAndResize"]["T"] = SupportedTypes(device_id);
    type_constraint_map["DepthwiseConv2dNative"]["T"] = SupportedTypes(device_id);
    type_constraint_map["DepthToSpace"]["T"] = [device_id](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      return supported_types;
    }();
    type_constraint_map["Equal"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Exp"]["T"] = SupportedTypes(device_id);
    type_constraint_map["ExpandDims"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Fill"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Fill"]["index_type"] = SupportedTypesIdx(device_id);
    type_constraint_map["FloorMod"]["T"] = [device_id, ov_version](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
        // Floor_mod supports only I32 precision of inputs for CPU
        supported_types = {DT_INT32,DT_FLOAT};
        if(ov_version == "2021.1"){
          supported_types.erase(DT_FLOAT);
        }
      }
      else if (device_id=="MYRIAD" || device_id=="HDDL"){
        supported_types.erase(DT_INT32); 
      }
      else if (device_id=="GPU"){
        supported_types.erase(DT_INT32);
        supported_types.erase(DT_UINT8);
      }
      return supported_types;
    }();
    type_constraint_map["FloorDiv"]["T"] = [device_id, ov_version](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
        if(ov_version == "2021.1"){
          supported_types.erase(DT_UINT32);
          supported_types.erase(DT_UINT64);
        }
        supported_types.erase(DT_UINT16);
      }
      else if (device_id=="MYRIAD" || device_id=="HDDL"){
        supported_types.erase(DT_INT32);
        supported_types.erase(DT_UINT8);
      }
      return supported_types;
    }();
    type_constraint_map["FusedBatchNorm"]["T"] = SupportedTypes(device_id);
    type_constraint_map["FusedBatchNormV3"]["T"] = [device_id, ov_version](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
#ifdef ENABLE_DT_HALF    
        supported_types.insert(DT_HALF);
#endif
      }
      return supported_types;
    }();
    type_constraint_map["_FusedConv2D"]["T"] = SupportedTypes(device_id); // formed after TF optimization pass, not in original graph
    type_constraint_map["_FusedMatMul"]["T"] = SupportedTypes(device_id); // formed after TF optimization pass, not in original graph
    type_constraint_map["Gather"]["Tparams"] = SupportedTypes(device_id);
    type_constraint_map["Gather"]["Tindices"] = SupportedTypesIdx(device_id);
    type_constraint_map["GatherV2"]["Tparams"] = SupportedTypes(device_id);
    type_constraint_map["GatherV2"]["Tindices"] = SupportedTypesIdx(device_id);
    type_constraint_map["GatherV2"]["Taxis"] = SupportedTypesIdx(device_id);
    type_constraint_map["Greater"]["T"] = SupportedTypes(device_id); //cwise_math    
    type_constraint_map["GreaterEqual"]["T"] = SupportedTypes(device_id); 
    type_constraint_map["Identity"]["T"] = {DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_BOOL};//SupportedTypes(device_id);
    // LRN: If input is of type other then the mentioned types, TF itself throws an error
    // For other attributes TF automatically typecasts them to required types
    type_constraint_map["LRN"]["T"] = {DT_BFLOAT16, DT_HALF, DT_FLOAT};
    type_constraint_map["LeakyRelu"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Less"]["T"] = SupportedTypes(device_id);
    type_constraint_map["LogicalAnd"]["T"] = SupportedTypes(device_id);
    type_constraint_map["LogSoftmax"]["T"] = [device_id](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
#ifdef ENABLE_DT_HALF    
        supported_types.insert(DT_HALF);
#endif
      }
      return supported_types;
    }();
    type_constraint_map["MatMul"]["T"] = SupportedTypes(device_id);
    type_constraint_map["MaxPool"]["T"] = [device_id, ov_version](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
#ifdef ENABLE_DT_HALF    
        supported_types.insert(DT_HALF);
#endif
      }
      return supported_types;
    }();
    type_constraint_map["Max"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Maximum"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Mean"]["T"] = [device_id](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="MYRIAD"){
        supported_types.erase(DT_INT32);
      }
      else if (device_id=="HDDL"){
          supported_types.erase(DT_INT32);
      }
      else if (device_id=="CPU"){
        supported_types.erase(DT_INT16);
        supported_types.erase(DT_UINT16);
      }
      return supported_types;
    }();
    type_constraint_map["Mean"]["Tidx"] = SupportedTypesIdx(device_id);    
    type_constraint_map["Minimum"]["T"] = SupportedTypes(device_id);
    type_constraint_map["MirrorPad"]["T"] = SupportedTypes(device_id);  // For unit tests  
    type_constraint_map["MirrorPad"]["Tpaddings"] = SupportedTypesIdx(device_id);  // For unit tests   
    type_constraint_map["Mul"]["T"] = [device_id](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
          supported_types.erase(DT_UINT16);
      }
      return supported_types;
    }();
    type_constraint_map["Neg"]["T"] = SupportedTypes(device_id); //cwise_math  
    type_constraint_map["NonMaxSuppressionV2"]["T"] = SupportedTypes(device_id); // formed after TF optimization pass, not in original graph  
    type_constraint_map["OneHot"]["axis"] = SupportedTypesIdx(device_id);
    type_constraint_map["OneHot"]["T"] = SupportedTypes(device_id);
    type_constraint_map["OneHot"]["TI"] = SupportedTypes(device_id);
    type_constraint_map["Pack"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Pad"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Pad"]["Tpaddings"] = SupportedTypesIdx(device_id);
    type_constraint_map["PadV2"]["T"] = [device_id](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
        supported_types.erase(DT_UINT8); 
      }
      return supported_types;
    }();
    type_constraint_map["PadV2"]["Tpaddings"] = SupportedTypesIdx(device_id);
    //Additonal DT_HALF is needed. Need to handle this at common place.
    type_constraint_map["Placeholder"]["dtype"] = { DT_FLOAT,DT_HALF, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16};
    type_constraint_map["Prod"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Prod"]["Tidx"] = SupportedTypesIdx(device_id);
    type_constraint_map["Range"]["Tidx"] = SupportedTypesIdx(device_id);
    type_constraint_map["RealDiv"]["T"] = SupportedTypes(device_id); //cwise_math    
    type_constraint_map["Relu"]["T"] = [device_id](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="MYRIAD" || device_id=="HDDL"){
        supported_types.erase(DT_INT32);  
      }
      return supported_types;
    }();
    type_constraint_map["Relu6"]["T"] = [device_id](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
#ifdef ENABLE_DT_HALF    
        supported_types.insert(DT_HALF);
#endif
      }
      else if (device_id=="MYRIAD" || device_id=="HDDL"){
        supported_types.erase(DT_INT32);
      }
      return supported_types;
    }();
    type_constraint_map["Reshape"]["T"] = [device_id](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
        supported_types.insert(DT_BOOL);
      }
      return supported_types;
    }();
    type_constraint_map["ResizeBilinear"]["T"] = SupportedTypes(device_id);
    type_constraint_map["ResizeNearestNeighbor"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Round"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Rsqrt"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Shape"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Shape"]["out_type"] = SupportedTypesIdx(device_id); 
    type_constraint_map["Sigmoid"]["T"] = SupportedTypes(device_id); //cwise_math
    type_constraint_map["Sign"]["T"] = [device_id, ov_version](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
        if(ov_version == "2021.1"){
          supported_types.erase(DT_UINT32);
          supported_types.erase(DT_UINT64);
        }
      }
      return supported_types;
    }();
    type_constraint_map["Sinh"]["T"] = SupportedTypes(device_id); //cwise_math
    type_constraint_map["Size"]["T"] = SupportedTypes(device_id); 
    type_constraint_map["Size"]["out_type"] = SupportedTypesIdx(device_id); 
    type_constraint_map["Slice"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Softmax"]["T"] = [device_id](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
#ifdef ENABLE_DT_HALF    
        supported_types.insert(DT_HALF);
#endif
      }
      return supported_types;
    }();
    type_constraint_map["Softplus"]["T"] = SupportedTypes(device_id);
    type_constraint_map["SpaceToBatchND"]["T"] = SupportedTypes(device_id);
    type_constraint_map["SpaceToDepth"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Split"]["T"] = SupportedTypes(device_id);
    type_constraint_map["SplitV"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Sub"]["T"] = SupportedTypes(device_id);
    type_constraint_map["Squeeze"]["T"] = SupportedTypes(device_id);
    type_constraint_map["StridedSlice"]["T"] = SupportedTypes(device_id);
    type_constraint_map["StridedSlice"]["Index"] = SupportedTypesIdx(device_id);  
    type_constraint_map["Sub"]["T"] = SupportedTypes(device_id);  
    type_constraint_map["Sum"]["T"] = SupportedTypes(device_id); //cwise_math    
    type_constraint_map["Tanh"]["T"] = SupportedTypes(device_id); //cwise_math    
    type_constraint_map["Tile"]["T"] = [device_id](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="MYRIAD" || device_id=="HDDL"){
        supported_types.erase(DT_INT32);  
      }
      return supported_types;
    }(); 
    type_constraint_map["TopKV2"]["T"] = [device_id](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){ 
        supported_types.erase(DT_INT32);  
        supported_types.erase(DT_INT64);  
      }
      else if (device_id=="MYRIAD" || device_id=="HDDL"){
        supported_types.erase(DT_INT32);  
      }
      else if (device_id=="GPU"){
        //modified as test cases with bridge were failing, though GPU
        //supports DT_HALF, so could be a data type issue on the bridge side too
        supported_types.erase(DT_HALF);
      }      
      return supported_types;
    }(); 
    
    type_constraint_map["Transpose"]["T"] = [device_id](){ 
      //TODO: Additonal DT_HALF is needed for CPU. Need to handle this at common place.
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
#ifdef ENABLE_DT_HALF    
        supported_types.insert(DT_HALF);
#endif
      }
      else if (device_id=="MYRIAD" || device_id=="HDDL"){
        supported_types = {DT_FLOAT,DT_HALF, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16};
      }
      else if (device_id=="GPU"){
        //modified as test cases with bridge were failing, though GPU
        //supports DT_HALF, so could be a data type issue on the bridge side too
        supported_types.erase(DT_HALF);
      }
      return supported_types;
    }(); 
    type_constraint_map["Transpose"]["Tperm"] = SupportedTypesIdx(device_id);
    type_constraint_map["Where"]["T"] = [device_id](){ 
      std::set<DataType> supported_types = SupportedTypes(device_id);
      if (device_id=="CPU"){
        supported_types.insert(DT_BOOL);  
      }
      return supported_types;
    }();
    type_constraint_map["Unpack"]["T"] = SupportedTypes(device_id);
    type_constraint_map["ZerosLike"]["T"] = SupportedTypes(device_id);
  }
  return type_constraint_map;
}

std::set<std::string> GetTFSupportedOPs(std::string device_id, std::string ov_version){
  
  std::set<std::string> supported_ops = {};
  std::map<std::string, std::set<string>>  ov_based_op_list ={};
  if (device_id == "CPU") {
    // For default OpenVINO 2021.1 version
    supported_ops.insert(common_supported_ops.begin(), common_supported_ops.end());
    supported_ops.insert(cpu_only_ops.begin(), cpu_only_ops.end());
    supported_ops.insert(composite_ops.begin(), composite_ops.end());
    if(ov_version == "2021.2"){
      ov_based_op_list = ov_2021_2_op_update_cpu;
    }
  } else if (device_id == "GPU") {
    supported_ops.insert(common_supported_ops.begin(), common_supported_ops.end());
    supported_ops.insert(gpu_only_ops.begin(), gpu_only_ops.end());
    supported_ops.insert(composite_ops.begin(), composite_ops.end());
    if(ov_version == "2021.2"){
      ov_based_op_list = ov_2021_2_op_update_gpu;
    }
  } else if (device_id == "MYRIAD" || device_id == "HDDL") {
    supported_ops.insert(common_supported_ops.begin(), common_supported_ops.end());
    supported_ops.insert(vpu_only_ops.begin(), vpu_only_ops.end());
    supported_ops.insert(composite_ops.begin(), composite_ops.end());
    if(ov_version == "2021.2"){
      ov_based_op_list = ov_2021_2_op_update_vpu;
    }
  }
  if(!ov_based_op_list.empty()){
    for(auto it=ov_based_op_list.begin(); it!=ov_based_op_list.end(); ++it){
      if ((it->first == "add") || ((it->first == "update"))){
        if(!it->second.empty()){
          supported_ops.insert(it->second.begin(), it->second.end());
        }
      } else if (it->first == "remove"){
        if(!it->second.empty()){
          supported_ops.erase(it->second.begin(), it->second.end());
        }
      }
    }
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
    OCM_LOG(0) <<"\""<< op->name()<< "\" requires at least "<<
                                   count<< " input(s), got "<< op->num_inputs()<<
                                   " instead";
  }
  *result = true;
  return tensorflow::Status::OK();
}

// Validate the dimension of the input tensor of the node
static Status ValidateNodeInputDim(const Node* n, tensorflow::int32 count, bool* result){
  Node* tf_input_node;
  int input_idx = 0;
  TF_RETURN_IF_ERROR(n->input_node(input_idx, &tf_input_node));
  // get input shape
  TensorShape t;
  TF_RETURN_IF_ERROR(GetNodeAttr(tf_input_node->attrs(), "shape", &t));
  // check the first dimension
  if(t.dims()>count){
      *result = false;
      OCM_LOG(0)<<" ERROR : "<< n->name()<< "\" supports max  "<<
                                count<< " input dims, got "<< t.dims()<<
                                " instead" << std::endl;
  }
  return tensorflow::Status::OK();
}

// Generates a "simple" confirmation function which always returns true,
static ConfirmationFunction SimpleConfirmationFunction() {
  auto cf = [](tensorflow::Node *, bool* result) {
    *result = true;
    return tensorflow::Status::OK();
  };
  return cf;
};

/**
 * Generates constraints map for all the Tensorflow Ops which check
 * all the attributes
 * @return a map with key as opname string and value as confirmation function 
 */
const std::map<std::string, ConfirmationFunction>& GetConfirmationMap(std::string device_id, std::string ov_version) {
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
      *result=true;
      if(device_id=="HDDL")
      {  
        tensorflow::int32 count = 5;
        TF_RETURN_IF_ERROR(ValidateNodeInputDim(n, count, result));
      }
      return tensorflow::Status::OK();
    };
    confirmation_function_map["ArgMin"] = [device_id](Node* n, bool* result) {
      *result=true;
      if(device_id=="HDDL")
      {  
        tensorflow::int32 count = 5;
        TF_RETURN_IF_ERROR(ValidateNodeInputDim(n, count, result));
      }
      return tensorflow::Status::OK();
    };
    confirmation_function_map["Asin"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Asinh"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Atan"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Atanh"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["AvgPool"] = SimpleConfirmationFunction();
    confirmation_function_map["BatchToSpaceND"] = SimpleConfirmationFunction();
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
    confirmation_function_map["DepthwiseConv2dNative"] = SimpleConfirmationFunction();
    confirmation_function_map["DepthToSpace"] = SimpleConfirmationFunction();
    confirmation_function_map["ExpandDims"] = SimpleConfirmationFunction();
    confirmation_function_map["Equal"] = SimpleConfirmationFunction();
    confirmation_function_map["Exp"] = SimpleConfirmationFunction();
    confirmation_function_map["Fill"] = SimpleConfirmationFunction();
    confirmation_function_map["FloorMod"] = SimpleConfirmationFunction();
    confirmation_function_map["FloorDiv"] = SimpleConfirmationFunction();
    confirmation_function_map["FusedBatchNorm"] = [device_id](Node* n, bool* result) {
      bool tf_is_training;
      if (GetNodeAttr(n->attrs(), "is_training", &tf_is_training) !=
          Status::OK()) {
        tf_is_training = true;
      }
      *result = !tf_is_training;
      return tensorflow::Status::OK();
    };
    confirmation_function_map["FusedBatchNormV3"] = confirmation_function_map["FusedBatchNorm"];
    confirmation_function_map["_FusedConv2D"] = SimpleConfirmationFunction();
    confirmation_function_map["_FusedMatMul"] = SimpleConfirmationFunction();  
    confirmation_function_map["Gather"] = SimpleConfirmationFunction();
    confirmation_function_map["GatherV2"] = [device_id](Node* n, bool* result) {
      *result = true;
      // First dimension of the input cannot be zero
      Node* tf_input_node;
      int input_idx = 1;
      TF_RETURN_IF_ERROR(n->input_node(input_idx, &tf_input_node));
      if(tf_input_node->type_string() ==  "Const"){
        // get size_splits  values
        Tensor values;
        TF_RETURN_IF_ERROR(GetNodeAttr(tf_input_node->attrs(), "value", &values));
        for(int i=0; i< values.dims() ;i++) { /// `TensorShape` in `tensor_shape.h`.
          if(values.dim_size(i)==0){ /// Convenience accessor for the tensor shape.
            *result = false;
            OCM_LOG(0) << " ERROR : " << n->type_string() << " Op has dimension size " << values.dim_size(i) << std::endl;
            return tensorflow::Status::OK();
          }
        }
      }
      return tensorflow::Status::OK();
    };
    confirmation_function_map["Greater"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["GreaterEqual"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Identity"] = SimpleConfirmationFunction();
    confirmation_function_map["LRN"] = SimpleConfirmationFunction();
    confirmation_function_map["LeakyRelu"] = SimpleConfirmationFunction();
    confirmation_function_map["Less"] = SimpleConfirmationFunction();
    confirmation_function_map["LogicalAnd"] = SimpleConfirmationFunction();
    confirmation_function_map["LogSoftmax"] = SimpleConfirmationFunction();
    confirmation_function_map["MatMul"] = SimpleConfirmationFunction();
    confirmation_function_map["MaxPool"] = SimpleConfirmationFunction();
    confirmation_function_map["Max"] = SimpleConfirmationFunction();
    confirmation_function_map["Maximum"] = SimpleConfirmationFunction();
    confirmation_function_map["Mean"] = SimpleConfirmationFunction();
    confirmation_function_map["Minimum"] = SimpleConfirmationFunction();
    confirmation_function_map["MirrorPad"] = [device_id](Node* n, bool* result) {
      *result = true;
      // for VPU num of padding dimension has to be 4, otherwise getting following
      // error with OV, AssertionFailed: layer->pads_begin.size() == 4
      if (device_id=="MYRIAD" || device_id=="HDDL"){
        Node* tf_pad_paddings_node;
        int input_idx = 1;
          
        TF_RETURN_IF_ERROR(n->input_node(input_idx, &tf_pad_paddings_node));
        if(tf_pad_paddings_node->type_string() ==  "Const"){
          // get pad values
          Tensor values;
          TF_RETURN_IF_ERROR(GetNodeAttr(tf_pad_paddings_node->attrs(), "value", &values));
          // check the first dimension
          if(values.dim_size(0) != 4){
              *result = false;
          }
        }
      }

      // GPU doesn't supports input dimension greater than 5
      if (device_id=="GPU"){
        tensorflow::int32 count = 5;
        TF_RETURN_IF_ERROR(ValidateNodeInputDim(n, count, result));
      }
      return tensorflow::Status::OK();
    };
    confirmation_function_map["Mul"] = SimpleConfirmationFunction();
    confirmation_function_map["Neg"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["NonMaxSuppressionV2"] = SimpleConfirmationFunction();
    confirmation_function_map["NoOp"] = SimpleConfirmationFunction();
    confirmation_function_map["OneHot"] = [device_id](Node* n, bool* result) {
      *result = true;
      // GPU doesn't supports input dimension greater than 5
      if (device_id=="GPU"){
        tensorflow::int32 count = 5;
        TF_RETURN_IF_ERROR(ValidateNodeInputDim(n, count, result));
      }
      return tensorflow::Status::OK();
    };
    confirmation_function_map["Pack"] = [](Node* n, bool* result) {
      // num of inputs
      tensorflow::int32 count = 1;
      TF_RETURN_IF_ERROR(ValidateInputCountMin(n, count, result));
      return tensorflow::Status::OK();
    };
    confirmation_function_map["Pad"] = confirmation_function_map["MirrorPad"];
    confirmation_function_map["PadV2"] = confirmation_function_map["MirrorPad"];
    confirmation_function_map["Placeholder"] = SimpleConfirmationFunction();
    confirmation_function_map["Prod"] = SimpleConfirmationFunction();
    confirmation_function_map["Range"] = SimpleConfirmationFunction();
    confirmation_function_map["RealDiv"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Relu"] = SimpleConfirmationFunction();
    confirmation_function_map["Relu6"] = SimpleConfirmationFunction();
    confirmation_function_map["Reshape"] = SimpleConfirmationFunction();
    confirmation_function_map["ResizeBilinear"] = SimpleConfirmationFunction();
    confirmation_function_map["ResizeNearestNeighbor"] = SimpleConfirmationFunction();
    confirmation_function_map["Round"] = SimpleConfirmationFunction();
    confirmation_function_map["Rsqrt"] = SimpleConfirmationFunction();
    confirmation_function_map["Sigmoid"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Sign"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Sinh"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Shape"] = [device_id](Node* n, bool* result) {
      *result = true;
      // Myriad and HDDL doesn't supports input dimension greater than 5
      if (device_id=="MYRIAD" || device_id=="HDDL"){
        tensorflow::int32 count = 5;
        TF_RETURN_IF_ERROR(ValidateNodeInputDim(n, count, result));
      }
      return tensorflow::Status::OK();
    };
    confirmation_function_map["Size"] = SimpleConfirmationFunction();
    confirmation_function_map["Slice"] = SimpleConfirmationFunction();
    confirmation_function_map["Softmax"] = [device_id](Node* n, bool* result) {
      *result = true;
      // GPU doesn't supports input dimension greater than 5
      if (device_id=="GPU"){
        tensorflow::int32 count = 5;
        TF_RETURN_IF_ERROR(ValidateNodeInputDim(n, count, result));
      }
      return tensorflow::Status::OK();
    };
    confirmation_function_map["Softplus"] = SimpleConfirmationFunction();
    confirmation_function_map["SpaceToBatchND"] = SimpleConfirmationFunction();
    confirmation_function_map["SpaceToDepth"] = SimpleConfirmationFunction();
    // TF itself throws an error if the num of dimensions at "split_dim" axis is not completely 
    // divisible by "num_split" value 
    confirmation_function_map["Split"] = SimpleConfirmationFunction();
    confirmation_function_map["SplitV"] = [device_id](Node* n, bool* result) {
      *result = true;
      Node* tf_input_node;
      int input_idx = 1;
        
      TF_RETURN_IF_ERROR(n->input_node(input_idx, &tf_input_node));
      if(tf_input_node->type_string() ==  "Const"){
        // get size_splits  values
        Tensor values;
        TF_RETURN_IF_ERROR(GetNodeAttr(tf_input_node->attrs(), "value", &values));

        //From Bridge translation. Need to check/create a test case for this.
        #if TF_VERSION < 2
          auto array = (void*)DMAHelper::base(&values);
        #else
          auto array = values.data();
        #endif

        int* int_array = static_cast<int*>(array);
        bool found_neg_val = false;
        for(int i=0; i< values.NumElements() ;i++){
          if(*(int_array+i) < 0){
            if(found_neg_val){
              *result = false;
              OCM_LOG(0) << " ERROR : " << n->type_string() << " Op has multiple negatve value in size_splits." << std::endl;
              return tensorflow::Status::OK();
            }
            found_neg_val = true;
          }
        }
      }
      return tensorflow::Status::OK();
    };
    confirmation_function_map["Squeeze"] = [device_id](Node* n, bool* result) {
      std::vector<int32> tf_axis;
      GetNodeAttr(n->attrs(), "squeeze_dims", &tf_axis);
      *result = true;
      // If Squeeze_dim is not provided check if atleast one of the
      // dimension value is 1 (and that would be squeezed out)
      if(tf_axis.size() == 0){
        Node* tf_input;
        int input_idx = 0;
        TF_RETURN_IF_ERROR(n->input_node(input_idx, &tf_input));
        if (tf_input->type_string()=="Const" || tf_input->type_string()=="Placeholder"){
          TensorShape t;
          *result = false;
          TF_RETURN_IF_ERROR(GetNodeAttr(tf_input->attrs(), "shape", &t));
          for (int i=0; i < t.dims(); i++){
            if(t.dim_size(i) == 1)
              *result = true;
          }
        }
      }
      return tensorflow::Status::OK();
    };
    confirmation_function_map["StridedSlice"] =[device_id,ov_version](Node* n, bool* result) {
      *result = true;

      // GPU doesn't supports input dimension greater than 5
      if (device_id=="GPU"){
        tensorflow::int32 count = 5;
        TF_RETURN_IF_ERROR(ValidateNodeInputDim(n, count, result));
      }

      // Check on stride values
      Node* tf_input_stride_node;
      int input_idx = 3;
      TF_RETURN_IF_ERROR(n->input_node(input_idx, &tf_input_stride_node));
      if(tf_input_stride_node->type_string()=="Const"){
        // get stride  values
        Tensor values;
        TF_RETURN_IF_ERROR(GetNodeAttr(tf_input_stride_node->attrs(), "value", &values));

        // Stride values are not specified exit
        for (int i=0; i < values.dims(); i++){
          if(values.dim_size(i) == 0){
            *result = false;
            OCM_LOG(0) << " ERROR : " << n->type_string() << " Op has empty Stride values." << std::endl;
            return tensorflow::Status::OK();
          }
        }

        // Check: Negative stride values are not supported
        if (device_id=="MYRIAD" || device_id=="HDDL" || (device_id=="GPU" && ov_version == "2021.1")){
        #if TF_VERSION < 2
          auto array = (void*)DMAHelper::base(&values);
        #else
          auto array = values.data();
        #endif
          int* int_array = static_cast<int*>(array);
          for(int i=0; i< values.NumElements() ;i++){
            if(*(int_array+i) < 0){
              *result = false;
              OCM_LOG(0) << " ERROR : " << n->type_string() << " Op has negative Stride value." << std::endl;
              return tensorflow::Status::OK();
            }
          }
        }
      }

      // shrink_axis_mask attribute is not supported for MYRIAD and HDDL
      if (device_id=="MYRIAD" || device_id=="HDDL"){
        int shrink_axis_mask;
        int new_axis_mask;
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "shrink_axis_mask", &shrink_axis_mask));
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "new_axis_mask", &new_axis_mask));
        if (shrink_axis_mask){
          *result = false;
          OCM_LOG(0) << " ERROR : " << n->type_string() << " shrink_axis_mask is set ." << std::endl;
        }
        if (new_axis_mask){
          *result = false;
          OCM_LOG(0) << " ERROR : " << n->type_string() << " new_axis_mask is set ." << std::endl;
        }
      }
      return tensorflow::Status::OK();
    };
    confirmation_function_map["Sub"] = SimpleConfirmationFunction();
    confirmation_function_map["Sum"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Tanh"] = SimpleConfirmationFunction(); //cwise_math
    confirmation_function_map["Tile"] = [device_id](Node* n, bool* result) {
      *result = true;
      Node* tf_input_node;
      int input_idx = 1;
      
      TF_RETURN_IF_ERROR(n->input_node(input_idx, &tf_input_node));
      if(tf_input_node->type_string() ==  "Const"){
        // get multiple  values
        Tensor tensor_values;
        TF_RETURN_IF_ERROR(GetNodeAttr(tf_input_node->attrs(), "value", &tensor_values));

        // For Myriad/HDDL/GPU, the number of dimensions in the values cannot be greater than 8/6
        if (device_id=="MYRIAD" || device_id=="HDDL"){
          tensorflow::int32 count = 8;
          TF_RETURN_IF_ERROR(ValidateNodeInputDim(n, count, result));
        }
        else if(device_id=="GPU"){
          tensorflow::int32 count = 6;
          TF_RETURN_IF_ERROR(ValidateNodeInputDim(n, count, result));
        }
        if(*result == false)
          return tensorflow::Status::OK();
        
        DataType multiples_type = tensor_values.dtype();
        switch(multiples_type){
          case DataType::DT_INT32: {
            CheckTensorValues<tensorflow::int32>(tensor_values, result);
            break;
          }
          case DataType::DT_INT64:{
            CheckTensorValues<tensorflow::int64>(tensor_values, result);
            break;
          }
          default:
            OCM_LOG(2)<<"Error: "<<n->type_string()<<" Unsupported datatype"<<"\n";
            break;
        }

        if(!(*result)){
          OCM_LOG(0) << " ERROR : " << n->type_string() << " Op has invalid value of param-multple" << std::endl;
          return tensorflow::Status::OK();
        }
      }
      return tensorflow::Status::OK();
    };
    // Adapted from Bridge Translation, "sorted" parameter doesn't supports
    // false value as of now
    confirmation_function_map["TopKV2"] = [](Node* n, bool* result) {
      *result = true;
      bool sorted_value;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "sorted", &sorted_value));
      if (!sorted_value){
        *result = false;
      }
      return tensorflow::Status::OK();
    };
    confirmation_function_map["Transpose"] = [device_id](Node* n, bool* result) {
      *result = true;
      if(device_id=="GPU")
      {
        tensorflow::int32 count = 6;
        TF_RETURN_IF_ERROR(ValidateNodeInputDim(n, count, result));
      }
      if(device_id=="MYRIAD")
      {
        tensorflow::int32 count = 8;
        TF_RETURN_IF_ERROR(ValidateNodeInputDim(n, count, result));
      }
      if(device_id=="HDDL")
      {
        tensorflow::int32 count = 5;
        TF_RETURN_IF_ERROR(ValidateNodeInputDim(n, count, result));
      }
      return tensorflow::Status::OK();
    };
    confirmation_function_map["Where"] = SimpleConfirmationFunction();
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
    OCM_LOG(0) << " ERROR : Node does not meet confirmation constraints: "
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

      DataType dt=DT_INVALID;

      if (GetNodeAttr(node->attrs(), type_attr_name, &dt) != Status::OK() ||
        std::find(allowed_types.begin(), allowed_types.end(), dt) ==
            allowed_types.end()) {
        OCM_LOG(0)<<node->type_string()<<" "<<type_attr_name<<": "<<dt<<"\n";
        type_constraints_ok = false;
        break;
      }
    }
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

  // Get the op type map based in the input device_id
  const TypeConstraintMap& type_constraint_map = GetTypeConstraintMap(device_id, ov_version);

  // Get the op mode confirmation map
  static std::map<std::string, ConfirmationFunction> confirmation_function_map = GetConfirmationMap(device_id, ov_version);

  // remove the support for the disabled ops
  if(!disabled_ops.empty()){
    for (auto itr : disabled_ops){
      auto supported_ops_itr = supported_ops.find(itr);
      if(supported_ops_itr!=supported_ops.end()){
        OCM_LOG(0)<<"INFO: Removing "<<itr<<" from the supported ops set \n";
        supported_ops.erase(supported_ops_itr);
      } else{
        OCM_LOG(2)<<"Error: Cannot disable unsupported OP - "<< itr; 
        break;
      }
    }
  }

  for (auto node : graph->op_nodes()){
    bool is_node_supported = true;
    // check if the optype supported
    do{
      // CHECK_1: if the op is supported
      is_node_supported &= IsOpSupported(node->type_string());
      if(is_node_supported == false){
        OCM_LOG(1) << " ERROR : " << node->type_string() << " Op is not supported " << std::endl;
        break;
      }

      // CHECK_2: OP Type and Dimensions Check...
      is_node_supported &= IsTypeSupported(node, type_constraint_map);
      if(is_node_supported == false){
        OCM_LOG(1) << " ERROR : " << node->type_string() << " Op Type is not supported " << std::endl;
        break;
      }

      // CHECK_3: OP mode check based on attributes
      is_node_supported &= IsOpModeSupportedTF(node, confirmation_function_map);
      if(is_node_supported == false){
        OCM_LOG(1) << " ERROR : " << node->type_string() << " Op Mode is not supported " << std::endl;
        break;
      }

      // Input dimension check
      is_node_supported &= IsOpInputDimZeroTF(node);
      if(is_node_supported == false){
        OCM_LOG(1) << " ERROR : " << node->type_string() << " Op - Input node Dim is ZERO " << std::endl;
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
