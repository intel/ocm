#*******************************************************************************
#  Copyright (C) 2021 Intel Corporation
# 
#  SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

OV_PATH=$1

#Download Tensorflow
if [ -d "tensorflow" ]
then
  echo "Tensorflow repo is already available"
else
  git clone https://github.com/tensorflow/tensorflow.git
  cd tensorflow; git checkout r2.2; 
  git apply ../scripts/tf_test_update.patch
  git apply ../scripts/tf_rem_unsupported_op_update.patch
  cd ..
fi 

#Download Unit test runner script
if [ -f "./scripts/tf_unittest_runner.py" ]; then
  echo "TF unit test runner script is already available"
else
  wget https://raw.githubusercontent.com/tensorflow/ngraph-bridge/master/test/python/tensorflow/tf_unittest_runner.py -O ./scripts/
  patch ./scripts/tf_unittest_runner.py ./patch/tf_test_dump_pb.patch
fi

#Create virtual python env
echo "Creating python virtual env"
python3 -m venv env
source env/bin/activate

echo "Installing dependencies in env"
python3 -m pip install --upgrade pip setuptools  
pip3 install numpy
pip3 install six
pip3 install -U tensorflow==2.2
pip3 install unittest-xml-reporting
pip3 install xmlrunner
pip3 install networkx
pip3 install defusedxml

#Disable eager execution
patch ./env/lib/python3.6/site-packages/tensorflow/python/framework/test_util.py ./patch/disable_eager_execution.patch

#OpenVINO set up
source $OV_PATH/bin/setupvars.sh
pip3 install -r $OV_PATH/deployment_tools/model_optimizer/requirements_tf2.txt

#Build benchmark app
echo "Building benchmark app"
$OV_PATH/deployment_tools/inference_engine/samples/cpp/build_samples.sh benchmark_app
