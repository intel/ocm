#*******************************************************************************
#  Copyright (C) 2021 Intel Corporation
# 
#  SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

import os
import pathlib
import argparse
import subprocess

def run_thru_mc():

  #path = './pbfiles'
  path = '/home/rrajore/models/ocm/sprint-1'
  
  files=[]
  
  for r,d,f in os.walk(path):
    for file in f:
      if '.pb' in file:
        files.append(os.path.join(r,file))
  
  os.system("mkdir -p tf_mc_logs")
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
  for f in files:

      #cmd = ["python3","../model_checker.py", "-i",f]
      cmd = ["../../build/ov_ocm", f]

      mc_log = "./tf_mc_logs/" + f[10:].replace("/","_")
      mc_log, ext = os.path.splitext(mc_log)
      mc_log += ".log"

      print("File log {} exists?: {}".format(mc_log,os.path.exists(mc_log)))

      if not os.path.exists(mc_log):
        print(cmd)
        result = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)

        mc_log_file = open(mc_log, "w")
        mc_log_file.write(result.stdout.decode("utf-8"))
        mc_log_file.close()

        print("Log file written to " + mc_log)

if __name__ == '__main__':

  run_thru_mc()

