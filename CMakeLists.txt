#*******************************************************************************
#  Copyright (C) 2021 Intel Corporation
# 
#  SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

cmake_minimum_required(VERSION 3.4)
project (ocm CXX)
set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll" ".so")
# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

if(WIN32)
	add_definitions(-DNOMINMAX)
endif()

# find_package(Threads)

option(FLOAT16_SUPPORT "Enable/Disable float16 support" OFF)
if(FLOAT16_SUPPORT)
    message("DT_HALF datatype support for all the devices will be enabled")
    add_definitions(-DENABLE_DT_HALF=1)
endif()
unset(FLOAT16_SUPPORT CACHE)

add_subdirectory(OCM)

include(cmake/FindTensorFlow.cmake)
if (DEFINED TF_SRC_DIR)
    message(STATUS "TensorFlow source directory: ${TF_SRC_DIR}")
else()
    message(FATAL_ERROR "Provide TensorFlow source directory: -DTF_SRC_DIR=<directory>")
endif()

# Rest of the stuff is to build the TF test application
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wno-comment -Wno-sign-compare -Wno-backslash-newline-escape")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -Wno-comment -Wno-sign-compare")
endif()

include_directories(${TF_SRC_DIR})
# Files that are generated during TF build are here
include_directories(${TF_SRC_DIR}/bazel-genfiles)
include_directories(${TensorFlow_INCLUDE_DIR})
include_directories(${TensorFlow_INCLUDE_DIR}/external/nsync/public)

# for OCM test app include files
include_directories(./include/)

# Make sure that the TF library exists
if ( APPLE )
    set(TF_LIB_NAME libtensorflow_framework.dylib)
elseif ( WIN32 )
    set(TF_LIB_NAME _pywrap_tensorflow_internal.lib)    
else()
    set(TF_LIB_NAME libtensorflow_framework.so.1)
endif()

add_library(tensorflow_cc_lib SHARED IMPORTED)
if(WIN32)
	set_target_properties(
		tensorflow_cc_lib
		PROPERTIES IMPORTED_LOCATION
		${TF_SRC_DIR}/bazel-bin/tensorflow/tensorflow_cc.dll
	)
	set_target_properties(
		tensorflow_cc_lib
		PROPERTIES IMPORTED_IMPLIB ${TF_SRC_DIR}/bazel-bin/tensorflow/tensorflow_cc.lib
	)
else()
  if(NOT(TensorFlow_VERSION LESS 2.0))
      set_target_properties(
          tensorflow_cc_lib
          PROPERTIES IMPORTED_LOCATION
          ${TF_SRC_DIR}/bazel-bin/tensorflow/libtensorflow_cc.so.2
      )
  else()
      set_target_properties(
          tensorflow_cc_lib
          PROPERTIES IMPORTED_LOCATION
          ${TF_SRC_DIR}/bazel-bin/tensorflow/libtensorflow_cc.so.1
      )
  endif()
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

message(STATUS "Compile Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Shared Link Flags: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "TensorFlow_FRAMEWORK_LIBRARY path: ${TensorFlow_FRAMEWORK_LIBRARY}")

set(APP_NAME ov_ocm)
add_executable(
    ${APP_NAME} test/ocm_ov_test.cc
)

target_link_libraries(
    ${APP_NAME}
    # pthread
    tensorflow_cc_lib
    ${TensorFlow_FRAMEWORK_LIBRARY}
    # absl_synchronization
    ocm
)