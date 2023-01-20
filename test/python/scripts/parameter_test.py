#*******************************************************************************
#  Copyright (C) 2023 Intel Corporation
# 
#  SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

import argparse
import re
import os
from termcolor import colored

def mode_validation(mode):
  regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
  if mode == "":
    print("{0}".format(colored("Mode Is Empty", "red")))
  if " " in mode:
    print("{0}: {1} ".format(colored("Mode name Contain Spaces", "red"), mode))
  if regex.search(mode) != None:
    print("{0}: {1}".format(colored("Mode name Contain Special Characters", "red"), mode))
  else:
    print("Mode is ok")
  

def device_validation(device):
  regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
  device_list=["CPU","GPU","MYRIAD","HDDL"]
  if(device in device_list ):
    if device == "":
        print("{0}".format(colored("Device List is found Empty", "red")))
    if " " in device:
        print("{0}: {1} ".format(colored("Device Name Contains Spaces", "red"), device))
    if regex.search(device) != None:
        print("{0}: {1}".format(colored("Device Name Contain Special Characters", "red"), device))
    else:
        print("Devices is ok")
  else:
    print("{0}".format(colored("Please Choose Device", "red")))

def modelpath_validation(model_path):
  if not os.path.exists(model_path) and not model_path=="None":
    print("{0}".format(colored("Model Path does not exists or Please give None in model path", "red")))
  else:
    print("Path is ok")

def testlist_validation(test_list):
  try:
    file_size = os.path.getsize('test_list.txt')
  except FileNotFoundError:
    print("{0}".format(colored("File is not Available", "red")))
    pass
  if not os.path.isfile(test_list):
    print("{0}".format(colored("Path doesn't Exists", "red")))
  elif file_size == 0:
    print("{0}".format(colored("File is found empty", "red")))
  else:
    print("Test List is ok")

def ov_validation(ov_version):
  regex = re.compile('[@!#$%^&*()<>?/\|}{~:]')
  if ov_version == "":
    print("{0}".format(colored("OpenVino Version in None", "red")))
  if " " in ov_version:
    print("{0}: {1} ".format(colored("OpenVino name Contain Spaces", "red"), ov_version))
  if regex.search(ov_version) != None:
    print("{0}: {1}".format(colored("Openvino name Contains Special Characters", "red"), ov_version))
  ov_path = "/opt/intel/"+ ov_version
  if not os.path.exists(ov_path):
    raise AssertionError("OV Path does not exists")
  else:
    if ov_version == "openvino_2021.1.110" or ov_version == "openvino_2021.2.185" or ov_version == "openvino_2021.3.394"or "openvino_2021.4" in ov_version or "openvino_2022.1.0" or "openvino_2022.2.0" in ov_version:
      print("OV version is ok")
    else:
      print("OV Version is incorrect")