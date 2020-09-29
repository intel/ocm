#ifndef _GCM_NODES_CHECKER_
#define _GCM_NODES_CHECKER_

#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <memory>


namespace ocm{

enum class Framework_Names{TF, ONNX} ;

struct NodeInfo{
	void* node;
};

//find if the op is supported based on the ops_support set
bool OpCheck(const std::string &optype, std::set<std::string> oplist);

class NodesChecker{
public:
	virtual ~NodesChecker() {
	};
	virtual bool IsOpSupported(std::string opname) = 0;
	virtual bool IsOpModeSupported() = 0;
	virtual void SetGraph(void* graph) = 0;
	virtual std::vector<void *> PrepareSupportedNodesList() = 0;

public:
	std::string device_id;
	std::string ov_version;
	std::set<std::string> supported_ops;
};


class FrameworkNodesChecker{
public:
	FrameworkNodesChecker(Framework_Names fw, std::string device_id, std::string ov_version, void* graph);
	std::vector<void *> MarkSupportedNodes();
	std::vector<unsigned int> GetUnSupportedNodesIndices();

private:
	std::vector<void *> nodes_list;
	// std::vector<NodeInfo> nodes_list;
	std::unique_ptr<NodesChecker> ocmFrameworkObj;
	std::vector<unsigned int> unsupported_nodes_idx;
};

} // namespace ocm

#endif //_GCM_NODES_CHECKER_