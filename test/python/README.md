The bash script run_test_setup,
- Downloads tensorflow repo which has the op unit test sources
- Downloads ngraph unit test runner & patches this script to dump .pb files (after changing inputs to placeholders)
- Creates a virtual python env & installs all dependencies & patches tf test_util.py to disable eager execution
- Build OpenVINO benchmark which is used for inferencing.
- To run the setup file, please use the command  
  `$./run_tests.sh ${OV_PATH}`

The bash script run_test
- Runs OCM, MO and INFER jobs.
- Runs on different devices "CPU", "GPU", "MYRIAD"
- Runs unit test runner to dump .pb files inside
- Runs .pb through OCM
- Runs .pb files through OpenVINO MO to generate IR files 
- Runs inference through benchmark app for generated IRs

Run Tests:
=========
OV_PATH is the openvino path and TEST_LIST is the list of unit tests.  
UNIT_TEST_FILE has TF ops unit tests listed out. The tests specified in this file will be executed by the bash script. Required with UTEST mode of run_test.sh  
MODEL_TEST_FILE has names of models to be tested and relevant parameters e.g. input_shape etc. Required with MTEST mode of run_test.sh.  

To run OCM   
    - On models  
         `$./run_tests.sh OV_PATH OCM MTEST MODEL_TEST_FILE DEVICE_TYPE`  
    - On unit tests (it generates PB files  too)
         `$./run_tests.sh OV_PATH OCM UTEST UNIT_TEST_FILE DEVICE_TYPE`  
    - On unit tests on pre-generated PB files  
         `$./run_tests.sh OV_PATH OCM UTEST UNIT_TEST_FILE DEVICE_TYPE ./pbfiles`  
        
To run MO on models  
    - On models  
         `$./run_tests.sh OV_PATH MO MTEST MODEL_TEST_FILE DEVICE_TYPE`
    - On unit test PB files  
         `$./run_tests.sh OV_PATH MO UTEST UNIT_TEST_FILE DEVICE_TYPE ./pbfiles`  
  
To run INFER   
    - On models  
         `$./run_tests.sh OV_PATH INFER MTEST MODEL_TEST_FILE DEVICE_TYPE`
    - On unit test PB files   
         `$./run_tests.sh OV_PATH INFER UTEST UNIT_TEST_FILE DEVICE_TYPE ./pbfiles`  

Output directories post execution of tests:
  - pbfiles: generated .pb files for the unit tests
  - tf_ocm_logs: OCM logs
  - tf_mo_logs: MO execution logs
  - tf_infer_logs: Benchmark App execution logs
