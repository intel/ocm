The bash script run_test_setup,
- Downloads tensorflow repo which has the op unit test sources
- Downloads ngraph unit test runner & patches this script to dump .pb files (after changing inputs to placeholders)
- Creates a virtual python env & installs all dependencies & patches tf test_util.py to disable eager execution
- Build OpenVINO benchmark which is used for inferencing.
- To run the setup file, please use the command
  $./run_tests.sh ${OV_PATH}

The bash script run_test
- Runs OCM, MO and INFER jobs.
- Runs unit test runner to dump .pb files inside
- Runs .pb through OCM
- Runs .pb files thru OpenVINO MO to generate IR files 
- Runs inference through benchmark app for generated IRs

Run Tests:
=========
test_list.txt has TF ops unit tests listed out. The tests specified in this file will be executed by the bash script. Required with UTEST mode of run_test.sh

test_models_list.txt has names of models to be tested and relevant parameters e.g. input_shape etc. Required with MTEST mode of run_test.sh
$OV_PATH is the openvino path and $TEST_LIST is the list of test models.
To run OCM 
    -On models
         $./run_tests.sh OCM MTEST /home/rrajore/models/ocm/sprint-1/
    - On unit tests generate PB files
         $./run_tests.sh OCM UTEST
    - On unit tests on pre-generated PB files
         $./run_tests.sh $OV_PATH OCM UTEST $TEST_FILE ./pbfiles
        
To run MO on models
    - On models
         $./run_tests.sh MO MTEST /home/rrajore/models/ocm/sprint-1/
    - On unit test PB files 
         $./run_tests.sh $OV_PATH MO UTEST $TEST_FILE ./pbfiles

To run INFER 
    - On models
         $./run_tests.sh INFER MTEST /home/rrajore/models/ocm/sprint-1/
    - On unit test PB files 
         $./run_tests.sh $OV_PATH INFER UTEST $TEST_FILE ./pbfiles

Output directories post execution of tests:
  - pbfiles: generated .pb files for the unit tests
  - tf_ocm_logs: OCM logs
  - tf_mo_logs: MO execution logs
  - tf_infer_logs: Benchmark App execution logs
