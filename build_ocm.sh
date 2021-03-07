#!/bin/bash

#*******************************************************************************
#  Copyright (C) 2021 Intel Corporation
# 
#  SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

# Check if Tensorflow source directory is provided as input, otheriwse use default installation path
if [ -z "$1" ]
  then
    TF_SRC_DIR="$(pwd)/setup/tensorflow/"
else
    TF_SRC_DIR=$1
fi

# Check the input as well as default path, and if both of them doesn't exists then exit the script
if [ -d  ${TF_SRC_DIR} ]
then
    echo "Tensorflow source directory is - ${TF_SRC_DIR}"
    echo "Activating the virtual environment"
    source "$(pwd)/setup/env/bin/activate"
else
    echo "Error: Default Tensorflow source directory doesn't exists"
    exit 1
fi

printf "\n---- Starting the OCM Build ------ \n"
mkdir build
cd build
cmake .. -DTF_SRC_DIR=${TF_SRC_DIR} -DFLOAT16_SUPPORT=ON
# the C library should be present at following path ${TF_SRC_DIR}/bazel-bin/tensorflow/libtensorflow_cc.so.2`
make
