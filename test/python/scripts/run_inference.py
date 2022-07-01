#*******************************************************************************
#  Copyright (C) 2021-2022 Intel Corporation
# 
#  SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

import os
import importlib
import argparse
import subprocess
import parameter_test
import time

def run_inference(path, device, ov_version):

  files=[]
  for r,d,f in os.walk(path):
    for file in f:
      if '.xml' in file:
        files.append(os.path.join(r,file))
  infer_log_path = "tf_infer_logs/"+device
  if not os.path.isdir(infer_log_path):
    os.makedirs(infer_log_path, exist_ok=True)

  for f in files:
      # timeout of 60 seconds 
      if device == "MYRIAD":
        time.sleep(10) 
      cmd = ["timeout","60", benchmark_app_exe_path, "-m", f,"-d", device,"-load_config","config.json", "-niter", "1","-nireq", "1"]
      start=13+len(device)+len(ov_version)
      infer_log = "./tf_infer_logs/" + device + "/" + f[start:].replace("/","_")
      infer_log, ext = os.path.splitext(infer_log)
      infer_log += ".log"
      print(" Infer log = ", infer_log)
      print("File log {} exists?: {}".format(infer_log,os.path.exists(infer_log)))

      if not os.path.exists(infer_log):
        result = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if device == "HDDL":
          time.sleep(10)

        with open(infer_log, "w") as infer_log_file:
            infer_log_file.write(result.stdout.decode("utf-8"))
        print("Log file written to " + infer_log)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-i',
                      '--model_path',
                      help='enter input model(.pb) path',
                      required=True)
  
  parser.add_argument('-v',
                      '--ov_version',
                      help='Specify OV version to infer',
                      required=True)

  parser.add_argument('-d',
                    '--device',
                    help='Device CPU, GPU, MYRIAD or HDDL',
                    required=True)
                      
  #Build benchmark app
  args = parser.parse_args()
  parameter_test.device_validation(args.device)
  parameter_test.modelpath_validation(args.model_path)
  parameter_test.ov_validation(args.ov_version)
  home = os.environ['HOME']
  benchmark_app_exe_path = home + "/benchmark_build/intel64/Release/benchmark_app"
  if not os.path.exists(benchmark_app_exe_path):
    raise AssertionError("Benchmark app Path does not exists")
  else:
    print("Benchmark app Path is ok")
  run_inference(args.model_path, args.device, args.ov_version)