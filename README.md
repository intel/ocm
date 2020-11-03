# OCM

Operator Capability Manager for Tensorflow models support analysis on Intel Hardwares 

## Build 
Requirements:
- Preferred TF version: 2.2
- Tensorflow C library
- Tensorflow for python 

### Step 1 -  To Build Tensorflow (One Time Process)
- If you already have tensorflow c library and tensorflow python library installed then move to step 2 to build OCM
- Setup bazel, follow the steps mentioned on the following link
    https://www.tensorflow.org/install/source#install_bazel
- Run the following bash file and it will build TF C and Python packages and install python package in a virtual environment
`build_and_setup_tf.sh`  
`#Note: building tensorflow could take around 1-1.5 hours`

### Step 2 - To Build OCM
- If the TF is built using step 1, then run,  
`build_ocm.sh`

- If tensorflow is installed in virtual environment, then activate it first and then run,  
`build_ocm.sh ${TF_SRC_DIR}`  
`# make sure the C library is present at following path ${TF_SRC_DIR}/bazel-bin/tensorflow/libtensorflow_cc.so.2`


## Steps to run the OCM C++ test application
- If Tensorflow for python is installed in virtual environment, then activate it  
`cd build`  
`./ov_ocm ${PATH_TO_TF_FROZEN_PB_FILE}`  

