#ifndef _OCM_ONNX_CHECKER_
#define _OCM_ONNX_CHECKER_

#include "ocm_nodes_checker.h"

namespace onnxruntime{
    class GraphViewer{
        public:
            std::string opName;
    };
}

class ONNXRTNodesChecker: public NodesChecker{
public:
	bool isOpSupported(std::string opName){
		return opcheck(opName, supported_ops);
	}
	bool isOpModeSupported(){
	    return true;     
	 }
	 void setGraph(void* graph){
	 	graph = static_cast< onnxruntime::GraphViewer*> (graph);
	 }
	 std::vector<void *> PrepareSupportedNodesList() override;
public:
	const onnxruntime::GraphViewer* graph;
};


#endif //_OCM_ONNX_CHECKER_