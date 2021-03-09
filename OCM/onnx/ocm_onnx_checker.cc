/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include "ocm_onnx_checker.h"

namespace ocm{

std::vector<void *> ONNXRTNodesChecker::PrepareSupportedNodesList(){
	std::vector<void *> nodeList;
/*
	// TODO: Get NG supported ops list too

	// get OV supported ops list for ONNX
	//supported_ops = NULL;//getONNXSupportedOPs(device_id, ov_version);

	for (auto node : graph->op_nodes()){
		bool isNodeSupported = true;
		// check if the optype supported
		do{
			// CHECK_1: if the op is supported
			isNodeSupported &= isOpSupported(node->type_string());
			if(isNodeSupported == false){break;}

			// CHECK_2: OP Type and Dimensions Check...
			
			// CHECK_3: OP mode check based on attributes

			// CHECK_4: TBD

		} while(false);
	}
	*/
	return nodeList;
}

} //namespace ocm 