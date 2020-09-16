#ifndef _OCM_TF_CHECKER_
#define _OCM_TF_CHECKER_

#include "ocm_nodes_checker.h"

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/graph/graph_constructor.h"

using namespace tensorflow;

using ConfirmationFunction = std::function<tensorflow::Status(tensorflow::Node*, bool*)>;
const std::map<std::string, ConfirmationFunction>& GetConfirmationMap();

std::set<std::string> getTFSupportedOPs(std::string device_id, std::string ov_version);

class TFNodesChecker: public NodesChecker{
public:
	bool isOpSupported(std::string opName) override{
		return opcheck(opName, supported_ops);
	}
	bool isOpModeSupported() override{
	    return true;     
	}
	void setGraph(void* tf_graph) override{
        //graphDef = static_cast<tensorflow::GraphDef*> (tf_graph);
		graph = static_cast<tensorflow::Graph*> (tf_graph);
        // // print graph nodes
        // for (auto node : graph->op_nodes()){
        //     std::cout << node->type_string() << std::endl;
        // }
	}
	std::vector<void *> PrepareSupportedNodesList() override;
    const tensorflow::Graph* graph;
    const tensorflow::GraphDef* graphDef;
};

#endif //_OCM_TF_CHECKER_