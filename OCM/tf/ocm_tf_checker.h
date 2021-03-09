/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef _OCM_TF_CHECKER_
#define _OCM_TF_CHECKER_

#include "ocm_nodes_checker.h"

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/graph/graph_constructor.h"

#if TF_VERSION < 2
  #include "tensorflow/core/common_runtime/dma_helper.h"
#endif

using namespace tensorflow;
namespace ocm{

using ConfirmationFunction = std::function<tensorflow::Status(tensorflow::Node*, bool*)>;
const std::map<std::string, ConfirmationFunction>& GetConfirmationMap(std::string device_id);

using TypeConstraintMap = std::map<std::string, std::map<std::string, std::set<DataType>>>;
const TypeConstraintMap& GetTypeConstraintMap(std::string device_id);

/**
 * Check if the value of a constant tensor is zero or negative 
 * @param values is the constant tensor values
 * @param result is a bool pointer, which is set false based on the condition
 */
template <typename T>
static void CheckTensorValues(const Tensor& values, bool* result){
    #if TF_VERSION < 2
		const T* array = static_cast<T*>((void*)DMAHelper::base(&values));
    #else
		const T* array = static_cast<T*>(values.data());
    #endif
	for (T i=0; i<values.NumElements(); i++){
		if (T(array[i])<=0){
			*result = false;
			break;
		}
	}                
}

/**
 * Generates the set of the Ops supported for tensorflow
 * @param device_id string with device info for e.g. "CPU", "GPU" etc
 * @param ov_version string with ov version info for e.g. "2021.1"
 * @return Set of tensorflow ops based on the above input info
 */
std::set<std::string> GetTFSupportedOPs(std::string device_id, std::string ov_version);

/**
 * Extends the NodesChecker class for tensorflow framework
 */
class TFNodesChecker: public NodesChecker{
public:
	bool IsOpSupported(std::string op_name) override{
		return OpCheck(op_name, supported_ops);
	}
	bool IsOpModeSupported() override{
	    return true;     
	}
	void SetGraph(void* tf_graph) override{
        //graphDef = static_cast<tensorflow::GraphDef*> (tf_graph);
		graph = static_cast<tensorflow::Graph*> (tf_graph);
	}
	
	/**
	 * Implements all the required checks on each node of the graph
	 * to find out whether they could be executed using OpenVINO IE
	 * for the given device_id
	 * @return list of supported nodes, needs to be typecasted back to the 
	 * underlying frameworks  node* for further use
	 */
	std::vector<void *> PrepareSupportedNodesList() override;
    const tensorflow::Graph* graph;
    const tensorflow::GraphDef* graphDef;
};

} //namespace ocm

#endif //_OCM_TF_CHECKER_
