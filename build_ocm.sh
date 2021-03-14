#!/bin/bash

#*******************************************************************************
#  Copyright (C) 2021 Intel Corporation
# 
#  SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

# Check if Tensorflow source directory is provided as input, otheriwse use default installation path
if [ -z "$1" ]
  then
    echo "--- Tensorflow source directory path is not provided ---"
    exit
else
    TF_SRC_DIR=$1
    if [ ! -d  ${TF_SRC_DIR}/tensorflow ]
    then
      echo "Tensorflow sourc code doesn't exists at the ${TF_SRC_DIR} "
      exit
    fi
fi

printf "\n---- Starting the OCM Build ------ \n"
mkdir build
cd build
cmake .. -DTF_SRC_DIR=${TF_SRC_DIR}/tensorflow -DFLOAT16_SUPPORT=ON
# the C library should be present at following path ${TF_SRC_DIR}/bazel-bin/tensorflow/libtensorflow_cc.so.2`
make

if [ -f  ov_ocm ]
then
  echo "--- Build Complete ---"
else
  echo "--- Build Failed ---"
fi