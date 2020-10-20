import os
import pathlib
import argparse
import subprocess

def run_inference(benchmark_app_path, path):

  files=[]
  
  for r,d,f in os.walk(path):
    for file in f:
      if '.xml' in file:
        files.append(os.path.join(r,file))
  
  
  os.system("mkdir -p tf_infer_logs")
  for f in files:
      cmd = [benchmark_app_path + "/benchmark_app", "-m", f,"-d","CPU","-load_config","config.json", "-niter", "10"]

      infer_log = "./tf_infer_logs/" + f[10:].replace("/","_")
      infer_log, ext = os.path.splitext(infer_log)
      infer_log += ".log"

      print("File log {} exists?: {}".format(infer_log,os.path.exists(infer_log)))

      if not os.path.exists(infer_log):
        print(cmd)
        result = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
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
  #Build benchmark app
  print("Building benchmark app")
  ov_path = os.environ['INTEL_OPENVINO_DIR']
  #cmd=[ov_path+"/deployment_tools/inference_engine/samples/cpp/build_samples.sh", 'benchmark_app']
  #print[subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  
  args = parser.parse_args()
  home=os.environ['HOME']
  benchmark_app_path=home + "/inference_engine_python_samples_build/intel64/Release"
  run_inference(benchmark_app_path, args.model_path)
