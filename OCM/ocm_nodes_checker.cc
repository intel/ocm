/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include "ocm_nodes_checker.h"
#include "tf/ocm_tf_checker.h"
#include "onnx/ocm_onnx_checker.h"
#include "ocm_logging.h"

namespace ocm{

const char *OCMStatusMsg[] =
{
  "SUCCESS",
  "FAILURE",
  "INVALID_GRAPH",
  "INVALID_DEVICE",
  "INVALID_OV_VERSION",
  "INVALID_FW",
  "OCM_INIT_FAILED"
};

FrameworkNodesChecker::FrameworkNodesChecker(Framework_Names fw, std::string device_id, std::string ov_version, void* graph){

  switch (fw)
  {
    case  Framework_Names::TF:
      //std::cout << "Inside FrameworkNodesChecker::TF" << std::endl;
      ocmFrameworkObj = std::unique_ptr<TFNodesChecker>(new TFNodesChecker);
      break;
    case  Framework_Names::ONNX:
      ocmFrameworkObj = std::unique_ptr<ONNXRTNodesChecker>(new ONNXRTNodesChecker);
      break;
    default:
      OCM_LOG(3) << "Invalid Framework type" << std::endl;
      ocm_status = OCMStatus::INVALID_FW;
      return;
  }
  
  if((device_id != "CPU") &&  (device_id != "GPU") && (device_id != "MYRIAD") && (device_id != "HDDL")){
    OCM_LOG(3) << "Invalid Device - " << device_id << ". Allowed options are CPU, GPU, MYRIAD or HDDL"  << std::endl;
    ocm_status = OCMStatus::INVALID_DEVICE;
    return;
  }
  ocmFrameworkObj->device_id = device_id;
  
  if((ov_version != "2021.1") && (ov_version != "2021.2") ){
    OCM_LOG(3) << "Invalid OpenVINO version - " << device_id << ". Allowed options are 2021.1 or 2021.2"  << std::endl;
    ocm_status = OCMStatus::INVALID_OV_VERSION;
    return;
  }
  ocmFrameworkObj->ov_version = ov_version;
  
  if(graph == NULL){
    ocm_status = OCMStatus::INVALID_GRAPH;
    OCM_LOG(3) << "Invalid Graph Pointer " << std::endl; 
    return;
  }
  ocmFrameworkObj->SetGraph(graph);
  ocm_status = OCMStatus::SUCCESS;
}

std::vector<void *> FrameworkNodesChecker::MarkSupportedNodes(){
  if(ocm_status == OCMStatus::SUCCESS){
    nodes_list = ocmFrameworkObj->PrepareSupportedNodesList();
  }else{
    OCM_LOG(3) << "OCM Initialization was incomplete with Error code : " << ocm::OCMStatusMsg[int(ocm_status)] << std::endl;
  }
  return nodes_list;
}

std::vector<unsigned int> FrameworkNodesChecker::GetUnSupportedNodesIndices(){
  MarkSupportedNodes();
  // prepare the vector of unsupported node indices for ONNXRT
  return unsupported_nodes_idx;
}


bool OpCheck(const std::string &op_name, std::set<std::string> oplist){
  bool check_passed = oplist.find(op_name)!=oplist.end();
  return check_passed;
}

void FrameworkNodesChecker::SetDisabledOps(const std::set<std::string> disabled_ops){
  ocmFrameworkObj->disabled_ops = disabled_ops;
}

} // namespace ocm 