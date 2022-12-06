# ******************************************************************************
#  Copyright (C) 2021-2022 Intel Corporation
# 
#  SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

cmake_minimum_required(VERSION 3.4)
project (ocm CXX)

# Find TensorFlow
if(NOT PY_SCRIPT_BUILD)
    # Use a separate TensorFlow finder if we're building with standalone CMake
    include(../cmake/FindTensorFlowStandalone.cmake)
else()
    find_package(TensorFlow REQUIRED)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wno-comment -Wno-sign-compare -Wno-backslash-newline-escape")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wno-comment -Wno-sign-compare")
endif()

include_directories(../include/)

if(WIN32)
	add_compile_definitions(OCM_COMPILE_LIBRARY=1)
endif()

# Make sure that the TF library exists
if ( APPLE )
    set(TF_LIB_NAME libtensorflow_framework.dylib)
endif()

set(SRC
    ocm_nodes_checker.cc
    tf/ocm_tf_checker.cc
    onnx/ocm_onnx_checker.cc
)

message(STATUS "Compile Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Shared Link Flags: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "TensorFlow_FRAMEWORK_LIBRARY path: ${TensorFlow_FRAMEWORK_LIBRARY}")

add_library(ocm SHARED ${SRC})

target_include_directories(ocm PRIVATE ${TensorFlow_INCLUDE_DIR}
                           ${TensorFlow_INCLUDE_DIR}/external/nsync/public)

if ( APPLE OR WIN32)
    target_link_libraries(
            ocm
            ${TensorFlow_FRAMEWORK_LIBRARY}
        )
endif()

string(REPLACE "." ";" TF_VERSION_LIST ${TensorFlow_VERSION})
list(GET TF_VERSION_LIST 0 TF_MAJOR_VERSION)
add_compile_definitions(TF_VERSION=${TF_MAJOR_VERSION})
