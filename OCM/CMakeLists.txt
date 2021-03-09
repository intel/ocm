# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

cmake_minimum_required(VERSION 3.4)

project (ocm CXX)
find_package(TensorFlow REQUIRED)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-comment -Wno-sign-compare -Wno-backslash-newline-escape")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -Wno-comment -Wno-sign-compare")
endif()

include_directories(${TensorFlow_INCLUDE_DIR})
include_directories(${TensorFlow_INCLUDE_DIR}/external/nsync/public)

include_directories(../include/)

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

string(REPLACE "." ";" TF_VERSION_LIST ${TensorFlow_VERSION})
list(GET TF_VERSION_LIST 0 TF_MAJOR_VERSION)
add_compile_definitions(TF_VERSION=${TF_MAJOR_VERSION})

