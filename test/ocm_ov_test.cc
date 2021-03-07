/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <memory>
#include <stdlib.h>
#include "ocm_logging.h"

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
    if(argc > 4 || argc < 4){
        std::cerr << "Usage: " << argv[0] << " TF_FROZEN_GRAPH DEVICE_TYPE OpenVINO_VERSION" << std::endl;
        return -1;
    }
    //OV_2021_1/2/3
    std::string graph_file_name = argv[1];
    std::string input_device_type = argv[2];
    std::string ov_version = argv[3];

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
#if 0
    for (auto node : graph_def.node()){
        std::cout << node.op() <<  "  and Attrs are -->   " ;//std::endl;
        for(auto attr : node.attr())
            std::cout << attr.first << "   " ;
        std::cout << " " <<  std::endl;    
    }
#endif
    // Convert the GraphDef to Graph
    tensorflow::GraphConstructorOptions opts;
    tensorflow::Graph graph(tensorflow::OpRegistry::Global());
    ConvertGraphDefToGraph(opts, graph_def, &graph);

    // print graph nodes
    /*
    for (auto node : graph.op_nodes()){
        std::cout << node->type_string() << std::endl;
        tensorflow::DataType dt;
        GetNodeAttr(node->attrs(), "dtype", &dt);
        std::cout  << node->type_string()  <<  "type is " << dt << std::endl;
    }
    */
    const char* ocm_log_env_var = std::getenv("OCM_LOG_LEVEL");
    if (ocm_log_env_var == nullptr) {
      if (!setenv("OCM_LOG_LEVEL", "0", 0)){
        std::cout <<"OCM_LOG_LEVEL environment variable is not set properly and will fallback to level INFO for the test application"<<std::endl;
      }
    }
    Framework_Names fName = Framework_Names::TF;
    std::string device_id = input_device_type;
    std::cout << "OpenVINO version " << ov_version << std::endl;
    FrameworkNodesChecker FC(fName, device_id, ov_version, &graph);
    if(FC.ocm_status == OCMStatus::SUCCESS){
      std::vector<void *> nodes_list = FC.MarkSupportedNodes();
    
      if(nodes_list.size() == graph.num_op_nodes())
        OCM_LOG(0) <<"All nodes are supported" << std::endl;
    }
    // cast back the nodes in the TF format
    //std::cout << "---List of Supported Nodes--- "<<"\n";
    // for (auto node : nodes_list){
    //     //std::cout << ((tensorflow::Node *)node)->type_string() <<  "    " ;
    // }

    return 0;
}
