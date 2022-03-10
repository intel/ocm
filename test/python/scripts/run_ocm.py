#*******************************************************************************
#  Copyright (C) 2021 Intel Corporation
# 
#  SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

import os
import pathlib
import argparse
import subprocess
import parameter_test

def run_thru_ocm(path, ov_ver, device):
  files=[]
  for r,d,f in os.walk(path):
    for file in f:
      if '.pb' in file:
        files.append(os.path.join(r,file))
  
  ocm_log_path = "tf_ocm_logs/"+device
  if not os.path.isdir(ocm_log_path):
    os.makedirs(ocm_log_path, exist_ok=True)

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
  
  ov_version = None
  if ov_ver == "openvino_2021.1.110":
    ov_version = "2021.1"
  elif ov_ver == "openvino_2021.2.185":
    ov_version = "2021.2"
  elif ov_ver == "openvino_2021.3.394":
    ov_version = "2021.3"
  elif "openvino_2021.4" in ov_ver:
    ov_version = "2021.4"
  elif "openvino_2022.1" in ov_ver :
    ov_version = "2022.1"

  if ov_version is None:
    raise AssertionError("OV Version is incorrect")
  print (ov_ver, ov_version)
  inv_file="invalid_tests_list_%s.txt"%device
  invalid_test=open(inv_file, 'r') 
  invalid_list = [line.partition('#')[0].rstrip() for line in invalid_test if not line.startswith('#')]
  for f in files:
      cmd = ["../../build/ov_ocm", f, device, ov_version]
      test_name,ext=os.path.splitext(f[10:].replace("/","_"))
      if test_name in invalid_list:
        continue
      ocm_log_path = "./tf_ocm_logs/" + device + "/" + test_name
      #ocm_log_path, ext = os.path.splitext(ocm_log_path)
      ocm_log_path += ".log"

      print("File log {} exists?: {}".format(ocm_log_path,os.path.exists(ocm_log_path)))

      if not os.path.exists(ocm_log_path):
        result = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)

        mc_log_file = open(ocm_log_path, "w")
        mc_log_file.write(result.stdout.decode("utf-8"))
        mc_log_file.close()

        print("Log file written to " + ocm_log_path)
  inv_file.close()
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-i',
                      '--model_path',
                      help='enter input model(.pb) path',
                      required=True)
  parser.add_argument('-d',
                    '--device',
                    help='Device CPU, GPU, MYRIAD or HDDL',
                    required=True)
  parser.add_argument('-v',
                      '--ov_version',
                      help='Specify OV version to infer',
                      required=True)

  args = parser.parse_args()
  parameter_test.device_validation(args.device)
  parameter_test.modelpath_validation(args.model_path)
  in_path = args.model_path
  ov_ver = args.ov_version
  run_thru_ocm(in_path, ov_ver, args.device)

