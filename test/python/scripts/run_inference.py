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
import time

def run_inference(benchmark_app_path, path, device, ov_name):

  files=[]
  
  for r,d,f in os.walk(path):
    for file in f:
      if '.xml' in file:
        files.append(os.path.join(r,file))
  
  infer_log_path = "tf_infer_logs/"+device
  os.system("mkdir -p "+infer_log_path)
  for f in files:
      # timeout of 60 seconds 
      cmd = ["timeout","60", benchmark_app_path + "/benchmark_app", "-m", f,"-d", device,"-load_config","config.json", "-niter", "1"]

      start=13+len(device)+len(ov_name)
      infer_log = "./tf_infer_logs/" + device + "/" + f[start:].replace("/","_")
      infer_log, ext = os.path.splitext(infer_log)
      infer_log += ".log"

      print("File log {} exists?: {}".format(infer_log,os.path.exists(infer_log)))

      if not os.path.exists(infer_log):
        print(cmd)
        result = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if device == "HDDL":
          time.sleep(10)
        infer_log_file = open(infer_log, "w")
        infer_log_file.write(result.stdout.decode("utf-8"))
        infer_log_file.close()

        print("Log file written to " + infer_log)

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
                      
  #Build benchmark app
  print("Building benchmark app")
  ov_path = os.environ['INTEL_OPENVINO_DIR']
  ov_name=os.path.basename(ov_path)
  #cmd=[ov_path+"/deployment_tools/inference_engine/samples/cpp/build_samples.sh", 'benchmark_app']
  #print[subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  
  args = parser.parse_args()
  home=os.environ['HOME']
  benchmark_app_path=home + "/inference_engine_python_samples_build/intel64/Release"
  checkmarx.checkmarx_validation_Device(args.device)
  run_inference(benchmark_app_path, args.model_path, args.device, ov_name)
