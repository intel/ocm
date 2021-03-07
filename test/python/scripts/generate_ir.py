#*******************************************************************************
#  Copyright (C) 2021 Intel Corporation
# 
#  SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

import os
import pathlib
import argparse
import subprocess
import checkmarx

def run_thru_mo(ov_path, path, test_list,mode, device):

  ov_name=os.path.basename(ov_path)
  files=[]
  mo_log_path = "tf_mo_logs/"+device
  os.system("mkdir -p "+mo_log_path)
  for r,d,f in os.walk(path):
    for file in f:
      if '.pb' in file:
        files.append(os.path.join(r,file))
  
  inv_file="invalid_tests_list_%s.txt"%device
  invalid_test=open(inv_file, 'r') 
  invalid_list = [line.partition('#')[0].rstrip() for line in invalid_test if not line.startswith('#')]
  for fname in files:
    if mode == 'UTEST':
      mo_args = " --input_model " + fname
    else:
      test_info=open(test_list, 'r') 
      all_models=test_info.readlines() 
      match = 0
      for model in all_models:
        #print("Model file is " + model)
        model=model.strip('\r\n')
        model_name, input_shape = model.split()
        file_name = os.path.basename(fname)
        if model_name == file_name:
          match = 1
          break
      if match == 0:
        continue 
   
    mo_op_path = "pbfiles_mo/"+ov_name+"/"+device

    if not os.path.exists(mo_op_path):
      os.system("mkdir -p "+ mo_op_path)

    mo_out = str(pathlib.Path(fname).parent.absolute())
    mo_out = mo_out.replace("pbfiles", mo_op_path)
    test_name, ext=os.path.splitext(fname[10:].replace("/","_"))
    
    if mode == 'UTEST':
      if test_name in invalid_list:
        continue
      if(device == "MYRIAD"):
        cmd = [ov_path + "/deployment_tools/model_optimizer/mo_tf.py", "--input_model", fname, "-o",mo_out, "--data_type", "FP16" ]
      elif(device == "GPU"):
        cmd = [ov_path + "/deployment_tools/model_optimizer/mo_tf.py", "--input_model", fname, "-o",mo_out, "--data_type", "FP16" ]
      elif(device == "HDDL"):
        cmd = [ov_path + "/deployment_tools/model_optimizer/mo_tf.py", "--input_model", fname, "-o",mo_out, "--data_type", "FP16" ]
      else:
        cmd = [ov_path + "/deployment_tools/model_optimizer/mo_tf.py", "--input_model", fname, "-o",mo_out ]
        #cmd = [ov_path + "/deployment_tools/model_optimizer/mo_tf.py",  "--log_level", "DEBUG","--input_model", fname, "-o",mo_out ]
    else:
      if(device == "MYRIAD" or device == "GPU"):
        cmd = [ov_path + "/deployment_tools/model_optimizer/mo_tf.py", "--input_model", fname,"--input_shape", input_shape, "-o",mo_out, "--data_type", "FP16" ]
      else:
        cmd = [ov_path + "/deployment_tools/model_optimizer/mo_tf.py", "--input_model", fname,"--input_shape", input_shape, "-o",mo_out ]
    mo_log = mo_log_path + "/" + test_name
    #mo_log, ext = os.path.splitext(mo_log)
    mo_log += ".log"

    print("File log {} exists?: {}".format(mo_log,os.path.exists(mo_log)))

    if not os.path.exists(mo_log):
      result = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)

      mo_log_file = open(mo_log, "w")
      mo_log_file.write(result.stdout.decode("utf-8"))
      mo_log_file.close()

      print("Log file written to " + mo_log)
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-i',
                      '--model_path',
                      help='enter input model(.pb) path',
                      required=True)
  parser.add_argument('-t',
                    '--test_list',
                    help='Give file name having test list',
                    required=True)
  parser.add_argument('-m',
                    '--mode',
                    help='Unit test=UTEST or Model Test=MTEST',
                    required=True)
  parser.add_argument('-d',
                    '--device',
                    help='Device CPU, GPU, MYRIAD or HDDL',
                    required=True)
  args = parser.parse_args()
  checkmarx.checkmarx_validation_Mode(args.mode)
  checkmarx.checkmarx_validation_Device(args.device)
  checkmarx.checkmarx_validation_ModelPath(args.model_path)
  checkmarx.checkmarx_validation_TestList(args.test_list)
  ov_path = os.environ['INTEL_OPENVINO_DIR']
  run_thru_mo(ov_path, args.model_path, args.test_list, args.mode, args.device)
