<p>English | <a href="https://github.com/intel/ocm/blob/master/test/python/README_cn.md">简体中文</a></p>

The bash script run_test_setup,
- Downloads tensorflow repo which has the op unit test sources
- Downloads openvino_tensorflow unit test runner & patches this script to dump .pb files (after changing inputs to placeholders)
- Creates a virtual python env & installs all dependencies & patches tf test_util.py to disable eager execution
- To run the setup file, please use the command  
  `$./run_tests.sh`

The bash script run_test
- Runs OCM, MO and INFER jobs.
- Runs on different devices "CPU", "GPU", "MYRIAD" and HDDL
- Build OpenVINO benchmark which is used for inferencing.
- Runs unit test runner to dump .pb files inside
- Runs .pb through OCM
- Runs .pb files through OpenVINO MO to generate IR files 
- Runs inference through benchmark app for generated IRs

Run Tests:
=========
 `$./run_tests.sh <OV_PATH> <BUILD_TYPE> <MODE> <TEST_FILE> <DEVICE_TYPE> <MODEL_PATH>`  

OV_PATH is the openvino path 
BUILD_TYPE can be OCM, MO and INFER. One or more supported options can be given in seperated form e.g. OCM,MO,INFER.  
MODE can be either UTEST (Unit testing) and MTEST (Model testing).  
TEST_FILE has TF ops unit tests listed out. The tests specified in this file will be executed by the bash script. Required with UTEST mode of run_test.sh  
DEVICE_TYPE can be "CPU", "GPU", "MYRIAD" or "HDDL"
MODEL_TEST_FILE has names of models to be tested and relevant parameters e.g. input_shape etc. Required with MTEST mode of run_test.sh.  

To run OCM   
    - On models  
         `$./run_tests.sh OV_NAME OCM MTEST MODEL_TEST_FILE DEVICE_TYPE`  
    - On unit tests (it generates PB files  too)
         `$./run_tests.sh OV_NAME OCM UTEST UNIT_TEST_FILE DEVICE_TYPE`  
    - On unit tests on pre-generated PB files  
         `$./run_tests.sh OV_NAME OCM UTEST UNIT_TEST_FILE DEVICE_TYPE ./pbfiles`  
        
To run MO on models  
    - On models  
         `$./run_tests.sh OV_NAME MO MTEST MODEL_TEST_FILE DEVICE_TYPE`
    - On unit test PB files  
         `$./run_tests.sh OV_NAME MO UTEST UNIT_TEST_FILE DEVICE_TYPE ./pbfiles`  
  
To run INFER   
    - On models  
         `$./run_tests.sh OV_NAME INFER MTEST MODEL_TEST_FILE DEVICE_TYPE`
    - On unit test PB files   
         `$./run_tests.sh OV_NAME INFER UTEST UNIT_TEST_FILE DEVICE_TYPE ./pbfiles`  

Output directories post execution of tests:
  - pbfiles         : generated .pb files for the unit tests
  - pbfiles_mo      : generated model optimzer output files.
  - tf_ocm_logs     : OCM logs
  - tf_mo_logs      : MO execution logs
  - tf_infer_logs   : Benchmark App execution logs
