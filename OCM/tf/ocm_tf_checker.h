#ifndef _OCM_TF_CHECKER_
#define _OCM_TF_CHECKER_

#include "ocm_nodes_checker.h"

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/graph/graph_constructor.h"

using namespace tensorflow;
namespace ocm{

using ConfirmationFunction = std::function<tensorflow::Status(tensorflow::Node*, bool*)>;
const std::map<std::string, ConfirmationFunction>& GetConfirmationMap();

using TypeConstraintMap = std::map<std::string, std::map<std::string, std::set<DataType>>>;
const TypeConstraintMap& GetTypeConstraintMap();

std::set<std::string> GetTFSupportedOPs(std::string device_id, std::string ov_version);

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
	std::vector<void *> PrepareSupportedNodesList() override;
    const tensorflow::Graph* graph;
    const tensorflow::GraphDef* graphDef;
};

} //namespace ocm

#endif //_OCM_TF_CHECKER_