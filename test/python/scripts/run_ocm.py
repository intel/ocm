import os
import pathlib
import argparse
import subprocess

def run_thru_ocm(path, device):


  files=[]
  
  for r,d,f in os.walk(path):
    for file in f:
      if '.pb' in file:
        files.append(os.path.join(r,file))
  
  ocm_log_path = "tf_ocm_logs/"+device
  os.system("mkdir -p "+ocm_log_path)

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
  for f in files:
      cmd = ["../../build/ov_ocm", f, device]

      ocm_log_path = "./tf_ocm_logs/" + device + "/" + f[10:].replace("/","_")
      ocm_log_path, ext = os.path.splitext(ocm_log_path)
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
  in_path = args.model_path
  run_thru_ocm(in_path, args.device)

