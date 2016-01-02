import subprocess
import sys

def run_process(cmd):
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
  (output, _) = process.communicate()
  process.wait()
  return output

def find_free_gpu():
  gpus = run_process(["nvidia-smi", "-L"]).splitlines()

  for gpu in range(len(gpus)):
    status = run_process(["nvidia-smi", "-i", str(gpu)])
    if "No running processes found" in status:
      return gpu
  # raise Exception("No free gpu found!")
  return -1

print(find_free_gpu())
