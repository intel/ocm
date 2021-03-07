#*******************************************************************************
#  Copyright (C) 2021 Intel Corporation
# 
#  SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

OV_PATH=$1
BUILD_TYPE=$2
MODE=$3
TEST_LIST=$4
DEVICES=$5
MODEL_PATH=$6

source $OV_PATH/bin/setupvars.sh
ov_name=$(basename $OV_PATH)

#Clean up
echo "Activating python virtual env"
source env/bin/activate

#Build benchmark app
echo "Building benchmark app"
$OV_PATH/deployment_tools/inference_engine/samples/cpp/build_samples.sh benchmark_app

#Applying TF patch 
echo "Applying TF Unit test update patch"
cd tensorflow
git checkout tensorflow/python/kernel_tests/
git apply ../scripts/tf_test_update.patch
git apply ../scripts/tf_rem_unsupported_op_update.patch
cd ..

generate_unittest_pbfiles(){
  rm -rf ./pbfiles
  #Run test script
  python3 ./scripts/tf_unittest_runner.py --tensorflow_path ./tensorflow/tensorflow/python --run_tests_from_file $TEST_LIST
}
#Run through model checker
ocm_checker(){
  rm -rf tf_ocm_logs/$DEVICE
  echo "Running through OCM"
  python3 ./scripts/run_ocm.py -i $MODEL_PATH -d $DEVICE
  echo "Finished running through OCM"
}

#Run through model optimizer
model_optimize(){
  rm -rf tf_mo_logs/$DEVICE
  mo_op_path=("pbfiles_mo/"$ov_name"/"$device)
  rm -rf $mo_op_path
  echo "Generating IR files"
  python3 ./scripts/generate_ir.py -i $MODEL_PATH -t $TEST_LIST -m $MODE -d $DEVICE
  echo "End Generating IR files"
}


#Run inference
run_infer(){
  rm -rf tf_infer_logs/$DEVICE
  mo_op_path=("pbfiles_mo/"$ov_name"/"$device)
  echo "Running inference"
  python3 ./scripts/run_inference.py -i $mo_op_path -d $DEVICE
  echo "End Running inference"
}

if [[ $MODE == "UTEST" ]]; then
    echo "Unit Test of Ops"
    if [[ -z $MODEL_PATH ]]; then
      generate_unittest_pbfiles
      MODEL_PATH='./pbfiles'
    else
      echo "PB file path is " + $MODEL_PATH
    fi
   
elif [[ $MODE == "MTEST" ]]; then
    echo "Model Testings"
   

else
    echo $MODE
    echo "Sorry!! Wrong Mode Option."
fi

if [[ -n $DEVICES ]]; then
    IFS=',' read -ra DEVICE_LIST <<< "$DEVICES"
    for i in ${!DEVICE_LIST[@]}; do
        #echo "Loop Count "$i" and value is "${DEVICE_LIST[i]}
        if [[ ${DEVICE_LIST[i]} == "CPU" || ${DEVICE_LIST[i]} == "GPU" ||  ${DEVICE_LIST[i]} == "MYRIAD" || ${DEVICE_LIST[i]} == "HDDL" ]]; then
            echo "Device name "${DEVICE_LIST[i]}
        else
            echo "Device name "${DEVICE_LIST[i]}
            echo "Sorry!! Wrong Device Option. Removing this from list"
            unset DEVICE_LIST[i]
        fi
    done
else
    echo "No Device provided. Default setting to CPU!!"
    DEVICE_LIST="CPU"
fi
if [[ -z ${DEVICE_LIST[@]} ]]; then
    echo "No valid Device provided. Default setting to CPU!!"
    DEVICE_LIST="CPU"
fi

if [[ $BUILD_TYPE  == "None" ]]; then
    echo "No Build type selected!!"
else
    for DEVICE in ${DEVICE_LIST[@]}; do
        IFS=',' read -ra BUILD_LIST <<< "$BUILD_TYPE"
        for BUILD in ${BUILD_LIST[@]}; do
	        if [[ $BUILD  == "OCM" ]]; then
	            echo $BUILD
	            ocm_checker
	        elif [[ $BUILD  == "MO" ]]; then
	            echo $BUILD
	            model_optimize
	        elif [[ $BUILD  == "INFER" ]]; then
	            echo $BUILD
	            run_infer
	        else
	            echo $BUILD
	            echo "Sorry!! Wrong Build Option."
	        fi
        done
    done
fi
    
#deactivate virtual env
echo "Deactivate virtual env"
#deactivate