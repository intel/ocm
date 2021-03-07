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

def run_thru_ocm(path, device):


  files=[]
  
  for r,d,f in os.walk(path):
    for file in f:
      if '.pb' in file:
        files.append(os.path.join(r,file))
  
  ocm_log_path = "tf_ocm_logs/"+device
  os.system("mkdir -p "+ocm_log_path)

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

  ov_path = os.environ['INTEL_OPENVINO_DIR']
  ov_ver = os.path.basename(ov_path)
  
  ov_version = "2021.1"
  if ov_ver == "openvino_2021.1.110":
    ov_version = "2021.1"
  elif ov_ver == "openvino_2021.2.185":
    ov_version = "2021.2"
  print (ov_path, ov_ver, ov_version)
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
        print(cmd)
        result = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)

        mc_log_file = open(ocm_log_path, "w")
        mc_log_file.write(result.stdout.decode("utf-8"))
        mc_log_file.close()

        print("Log file written to " + ocm_log_path)

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

  args = parser.parse_args()
  checkmarx.checkmarx_validation_Device(args.device)
  checkmarx.checkmarx_validation_ModelPath(args.model_path)
  in_path = args.model_path
  run_thru_ocm(in_path, args.device)

