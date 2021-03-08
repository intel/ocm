/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#ifndef _OCM_ONNX_CHECKER_
#define _OCM_ONNX_CHECKER_

#include "ocm_nodes_checker.h"

namespace ocm{

namespace onnxruntime{
    class GraphViewer{
        public:
            std::string opName;
    };
}

class ONNXRTNodesChecker: public NodesChecker{
public:
	bool IsOpSupported(std::string opName){
		return OpCheck(opName, supported_ops);
	}
	bool IsOpModeSupported(){
	    return true;     
	 }
	 void SetGraph(void* graph){
	 	graph = static_cast< onnxruntime::GraphViewer*> (graph);
	 }
	 std::vector<void *> PrepareSupportedNodesList() override;
public:
	const onnxruntime::GraphViewer* graph;
};

}//namespace ocm 

#endif //_OCM_ONNX_CHECKER_