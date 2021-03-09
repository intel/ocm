/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef _GCM_NODES_CHECKER_
#define _GCM_NODES_CHECKER_

#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <memory>


namespace ocm{

/** 
 * An enum class.
 * It enables the framework selection for OCM
 */
enum class Framework_Names{TF, ONNX} ;

/**
 * function to find if the op is supported based on the ops support set
 * @param opname name of the op to be checked
 * @param oplist set of the supported ops
 * @return returns true if the opname is present in the oplist else false
 */
bool OpCheck(const std::string &opname, std::set<std::string> oplist);

enum class OCMStatus {
  SUCCESS             = 0,
  FAILURE             = 1,
  INVALID_GRAPH       = 2,
  INVALID_DEVICE      = 3,
  INVALID_OV_VERSION  = 4,
  INVALID_FW          = 5,
  OCM_INIT_FAILED     = 6,
};

/**
 * NodesChecker base class. All the common functions are declared as virtual functions
 * and are extended in the TF and ONNX classes
 */
class NodesChecker{
public:
  virtual ~NodesChecker() {
  };

  /**
   * find if the op is supported 
   * @param opname name of the op to be checked
   * @return returns true if the opname is present in the supported_ops else false
   */
  virtual bool IsOpSupported(std::string opname) = 0;

  /**
   * find if the different attributes of the nodes are supported 
   * based on OpenVINO IE plugins  
   * @return returns true if the node is supported for the given device_id and ov_version
   */
  virtual bool IsOpModeSupported() = 0;

  /**
   * The underlying framework's graph is passed as void* to OCM and then
   * this function typecasts it back to the required framework 
   */
  virtual void SetGraph(void* graph) = 0;

  /**
   * Implements all the required checks on each node of the graph
   * to find out whether they could be executed using OpenVINO IE
   * for the given device_id
   * @return list of supported nodes, needs to be typecasted back to the 
   * underlying frameworks  node* for further use
   */
  virtual std::vector<void *> PrepareSupportedNodesList() = 0;

public:
  /**
   * device_id can be "CPU", "GPU", "MYRIAD", "HDDL"
   */
  std::string device_id;

  /**
   * ov_version: MVP supports 2021.2
   */
  std::string ov_version;

  /**
   * Generated internally
   */ 
  std::set<std::string> supported_ops;

  /**
   * set of ops marked unsupported for the graph
   */
  std::set<std::string> disabled_ops;

};


/**
 * FrameworkNodesChecker class uses the functionalities provided by NodesChecker class,
 * which are ideally should be used by user to check which nodes of the graph can
 * run on OpenVINO backends
 */
class FrameworkNodesChecker{
public:
  /**
   * OCM Status
   */
  ocm::OCMStatus ocm_status;
  
  /**
   * Constructor to initialize the object of the class
   * @param fw Framework_Names enum class variable
   * @param device_id device on which the user intend to run the graph
   * i.e. "CPU", "GPU", "MYRIAD" or "HDDL"  
   */
  FrameworkNodesChecker(Framework_Names fw, std::string device_id, std::string ov_version, void* graph);

  /**
   * To be used with Tensorflow graph
   * @return the list of supported nodes
   */
  std::vector<void *> MarkSupportedNodes();

  /**
   * To be used with OnnxRT
   * @return the list of unsupported nodes indices  
   */
  std::vector<unsigned int> GetUnSupportedNodesIndices();

  /**
   * set disabled ops
   * @param disabled_ops Set of disabled ops name 
   */
  void SetDisabledOps(const std::set<std::string>); 

private:
  /**
   * Vector to store the supported nodes
   */
  std::vector<void *> nodes_list;

  /**
   * Object of type NodesChecker class
   */
  std::unique_ptr<NodesChecker> ocmFrameworkObj;

  /**
   * Vector to store the unsupported nodes indices
   */
  std::vector<unsigned int> unsupported_nodes_idx;

};

} // namespace ocm

#endif //_GCM_NODES_CHECKER_