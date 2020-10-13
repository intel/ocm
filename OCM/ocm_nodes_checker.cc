#include "ocm_nodes_checker.h"
#include "tf/ocm_tf_checker.h"
#include "onnx/ocm_onnx_checker.h"

namespace ocm{
    
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
            throw std::runtime_error("Invalid Framework type");
    }
    ocmFrameworkObj->SetGraph(graph);
    ocmFrameworkObj->device_id = device_id;
    ocmFrameworkObj->ov_version = ov_version;
}

std::vector<void *> FrameworkNodesChecker::MarkSupportedNodes(){
    //std::cout << "Inside MarkSupportedNodes" << std::endl;
    nodes_list = ocmFrameworkObj->PrepareSupportedNodesList();
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

} //namespace ocm 