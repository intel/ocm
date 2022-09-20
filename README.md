<p>English | <a href="https://github.com/intel/ocm/blob/master/README_cn.md">简体中文</a></p>


# Operator Capability Manager (OCM)

OCM checks Op support to run a Deep Learning model using OpenVINO on Intel Platforms (CPU, GPU, MYRIAD and HDDL). 
Currently it supports only Tensorflow models.

## Build 
Requirements:
- Tensorflow Version: 2.10.0
- Tensorflow CC library
- Tensorflow for python 

### Step 1 -  To Build Tensorflow (One Time Process)
- If you already have Tensorflow CC library and Tensorflow Framework library installed then move to step 2 to build OCM
- Setup bazel, follow the steps mentioned on the following link
    https://www.tensorflow.org/install/source#install_bazel
- Run the following bash file and it will build Tensorflow CC and Python packages and install python package in a virtual environment   
`build_and_setup_tf.sh ${TF_SRC_DIR}`   
`# if the Tensorflow source code is already cloned at ${TF_SRC_DIR} then it will checkout the required version and will rebuild it`   
`# otherwise it will clone Tensorflow source code at ${TF_SRC_DIR} path and will build it`   
`# Note: building tensorflow could take several hours based on system config`

### Step 2 - To Build OCM 
- Pre-requisite - Tensorflow Framework and CC libraries i.e. libtensorflow_framework.so and libtensorflow_cc.so
- If tensorflow is built and setup using Step 1 then activate the already created virtual environment,
`source ${TF_SRC_DIR}/ocm_venv/bin/activate`
- Otherwise Make sure that the current python environment has Tensorflow of required version pre-installed
`build_ocm.sh ${TF_SRC_DIR}`  
`# make sure Tensorflow CC library(libtensorflow_cc.so) is present at this path ${TF_SRC_DIR}/tensorflow/bazel-bin/tensorflow/`

## Steps to run the OCM C++ test application
`cd build`  
`./ov_ocm ${PATH_TO_TF_FROZEN_PB_FILE} ${DEVICE} {OPENVINO_VERSION}`  

### Usage Examples:
`./ov_ocm test.pb CPU 2021.2`   
`./ov_ocm test.pb MYRIAD 2021.1`

