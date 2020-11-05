OV_PATH=$1
BUILD_TYPE=$2
MODE=$3
TEST_LIST=$4
MODEL_PATH=$5

source $OV_PATH/bin/setupvars.sh

#Clean up
echo "Clearing up existing logs"

echo "Activating python virtual env"
source env/bin/activate

generate_unittest_pbfiles(){
  rm -rf ./pbfiles
  #Run test script
  python3 ./scripts/tf_unittest_runner.py --tensorflow_path ./tensorflow/tensorflow/python --run_tests_from_file $TEST_LIST
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
   
elif [[ $MODE == "MTEST" ]]; then
    echo "Model Testings"
   

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