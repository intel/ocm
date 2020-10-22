OV_PATH=$INTEL_OPENVINO_DIR
BUILD_TYPE=$1
MODE=$2
MODEL_PATH=$3

#Clean up
echo "Clearing up existing logs"

echo "Activating python virtual env"
source env/bin/activate

generate_unittest_pbfiles(){
  rm -rf ./pbfiles
  #Run test script
  python3 ./scripts/tf_unittest_runner.py --tensorflow_path ./tensorflow/ --run_tests_from_file test_list.txt
}
#Run through model checker
ocm_checker(){
  rm -rf tf_ocm_logs
  echo "Running through OCM"
  python3 ./scripts/run_ocm.py -i $MODEL_PATH
  echo "Finished running through OCM"
}

#Run through model optimizer
model_optimize(){
  rm -rf tf_mo_logs
  echo "Generating IR files"
  python3 ./scripts/generate_ir.py -i $MODEL_PATH -t $TEST_LIST -m $MODE
  echo "End Generating IR files"
}


#Run inference
run_infer(){
  rm -rf tf_infer_logs
  echo "Running inference"
  python3 ./scripts/run_inference.py -i $MODEL_PATH  
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
    TEST_LIST='test_list.txt'
elif [[ $MODE == "MTEST" ]]; then
    echo "Model Testings"
    TEST_LIST='test_model_list.txt'

else
    echo $MODE
    echo "Sorry!! Wrong Mode Option."
fi

if [[ $BUILD_TYPE  == "None" ]]; then
    echo "No Build type selected!!"
else
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
fi
    
#deactivate virtual env
echo "Deactivate virtual env"
#deactivate