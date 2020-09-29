#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <memory>

#include "ocm_nodes_checker.h"

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/graph/graph_constructor.h"

using namespace ocm;

int main(int argc, char** argv)
{
	//std::string graph_file_name = "/home/chandrakant/codes/models/test_graph/test.pb";
    //std::string graph_file_name = "/home/rrajore/models/inception/inception_v3_2016_08_28_frozen.pb";
    std::string graph_file_name = "/home/chandrakant/codes/models/ocm/resnet50/resnet50_fp32_pretrained_model.pb";

	tensorflow::SessionOptions options;
	std::unique_ptr<tensorflow::Session> session(NewSession(options));
	
	// read the protobuf and populate the graph_def
	tensorflow::GraphDef graph_def;
	auto load_graph_status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
	if (!load_graph_status.ok()) {
		std::cout << "Failed to load compute graph at " <<  graph_file_name << std::endl;
		return 0;
	}
	/*
	for (auto node : graph_def.node()){
		std::cout << node.op() <<  "    " ;//std::endl;
	}
	auto shape = graph_def.node().Get(0).attr().at("shape").shape();
	for (int i = 0; i < shape.dim_size(); i++) {
		std::cout << shape.dim(i).size()<<std::endl;
	}
	*/
	auto session_status = session->Create(graph_def);
	if (!session_status.ok())
	{
			std::cout << session_status.ToString() << std::endl;
	}

	// Convert the GraphDef to Graph
	tensorflow::GraphConstructorOptions opts;
	tensorflow::Graph graph(tensorflow::OpRegistry::Global());
	ConvertGraphDefToGraph(opts, graph_def, &graph);

	// print graph nodes
	/*
	for (auto node : graph.op_nodes()){
		std::cout << node->type_string() << std::endl;
	}
	*/

	Framework_Names fName = Framework_Names::TF;
	FrameworkNodesChecker FC(fName,"CPU", "2020_4", &graph);
	std::vector<void *> nodes_list = FC.MarkSupportedNodes();
	// cast back the nodes in the TF format
	std::cout << "---List of Supported Nodes--- "<<"\n";
	for (auto node : nodes_list){
    	std::cout << ((tensorflow::Node *)node)->type_string() <<  "    " ;
	}
	std::cout<<"\n";
	return 0;
}
