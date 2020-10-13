import os
import pathlib
import argparse
import subprocess

def run_thru_mc(path):


  files=[]
  
  for r,d,f in os.walk(path):
    for file in f:
      if '.pb' in file:
        files.append(os.path.join(r,file))
  
  os.system("mkdir -p tf_ocm_logs")
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
  for f in files:

      cmd = ["../../build/ov_ocm", f]

      mc_log = "./tf_ocm_logs/" + f[10:].replace("/","_")
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
  parser = argparse.ArgumentParser()
  parser.add_argument('-i',
                      '--model_path',
                      help='enter input model(.pb) path',
                      required=True)

  args = parser.parse_args()
  in_path = args.model_path
  run_thru_mc(in_path)

