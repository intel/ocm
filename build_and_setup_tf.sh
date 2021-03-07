#!/bin/bash

#*******************************************************************************
#  Copyright (C) 2021 Intel Corporation
# 
#  SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

#Download Tensorflow
TF_SRC_DIR="$(pwd)/setup/tensorflow/"
if [ -d  ${TF_SRC_DIR} ]
then
  echo "Tensorflow repo is already available"
else
  mkdir setup
	cd setup
  git clone https://github.com/tensorflow/tensorflow.git
  cd tensorflow; git checkout r2.2; 
  cd ../.. # back to base ocm directory
fi 

#Create virtual python env
cd setup
echo "Creating python virtual env"
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip setuptools  
pip install 'psutil' 'numpy>=1.16.0,<1.19.0' 'six>=1.12.0' 'wheel>=0.26'
pip install -U wheel
pip install -U 'keras_preprocessing>=1.1.1,<1.2' --no-deps

# Tensorflow Setup
cd tensorflow # get into the tensorflow codebase

# Configure TF build options
PYTHON_BASE_DIR=`python -c "import sys; print(sys.prefix)"`
PYTHON_VERSION=`python -c 'import sys; version=sys.version_info[:3]; print("{0}.{1}".format(*version))'`

export PYTHON_BIN_PATH="${PYTHON_BASE_DIR}/bin/python"
export PYTHON_LIB_PATH="${PYTHON_BASE_DIR}/lib/python${PYTHON_VERSION}/site-packages"
export TF_ENABLE_XLA="0"
export TF_CONFIGURE_IOS="0"
export TF_NEED_OPENCL_SYCL="0"
export TF_NEED_COMPUTECPP="0"
export TF_NEED_ROCM="0"
export TF_NEED_MPI="0"
export TF_NEED_CUDA="0"
export TF_NEED_TENSORRT="0"
export TF_DOWNLOAD_CLANG="0"
export TF_SET_ANDROID_WORKSPACE="0"
export CC_OPT_FLAGS="-march=native -Wno-sign-compare"

./configure

# Compile Tensorflow Python and C Library
bazel build --config=nonccl --config=noaws --config=nogcp --config=nohdfs --local_cpu_resources=8 --local_ram_resources 10240 --jobs=8  //tensorflow/tools/pip_package:build_pip_package 
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
bazel build --config=nonccl --config=noaws --config=nogcp --config=nohdfs --local_cpu_resources=8 --local_ram_resources 10240 --jobs=8  //tensorflow:libtensorflow_cc.so

# echo "Installing the built Tensorflow python package"
pip install /tmp/tensorflow_pkg/tensorflow-2.2.*.whl

echo "Deactivating the virtual environment"
deactivate