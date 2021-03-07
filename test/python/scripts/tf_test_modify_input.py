#*******************************************************************************
#  Copyright (C) 2021 Intel Corporation
# 
#  SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

import tensorflow as tf
import argparse
import os
import sys
from mo.utils.summarize_graph import summarize_graph
#import sys
#sys.path.append('../')
#import model_checker as mc

'''
Input: .pb files generated from unit test
Output: .pb files with input replaced to placeholder if applicable 

Steps:
  1. Load graph 
  2. Get output nodes of the graph. 
  3. Extract subgraph based on the output nodes. (Tests have multiple iterations, for ex. BiasAddTest.testIntType has many iterations with different int types like int8, int32 etc.)
  4. For each subgraph, 
        a. If first node is const, 
            - find node with this const as input
            - get tensor shape info of the node's first input and create placeholder
            - replace input with placeholder
        b. write new/ graph.
'''

def get_graph(input_file):

  # read input file into graph_def
  with tf.io.gfile.GFile(input_file, "rb") as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def,name='')
      
  return graph
  
def add_placeholder(graph_def):
  graph = tf.Graph()
  with graph.as_default():
    tf.import_graph_def(graph_def,name='')

    #Find the first constant node, get node with this const as input
    #add placeholder based on the node's first input
    node_to_replace=""
    if  graph.get_operations()[0].type == "Const":
        first_op = graph.get_operations()[0].name
        for op in graph.get_operations():
            if len(op.inputs) > 0:
                for input in op.inputs:
                    if input.op.name == first_op:
                        node_to_replace = op.name
                        shape = op.inputs[0].shape
                        dtype = op.inputs[0].dtype
                        # Create placeholder
                        x = tf.compat.v1.placeholder(dtype, shape, "newplaceholder")
                        break

            if not len(node_to_replace)==0:
                break

    return graph, node_to_replace

def replace_input_to_placeholder(graph_def, node_to_replace):

    #Replace const with newplaceholder
    for node in graph_def.node:
        if node.name == node_to_replace:
            print("Replacing input for op {}".format(node.name))
            node.input[0] = "newplaceholder"
            print("Replaced input with new placeholder")
            break

#Skipping tests with complex inputs & tests with unsupported OV ops
def skipTest(graph_def):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

        #for op in graph.get_operations():
            #if op.type not in mc.supported_ops:
                #print("Unsupported op {}".format(op.type))
                #return True

        return False

def process_graph(graph,file):
    #Disabled tests
    invalid_tests = read_tests_from_file("./invalid_tests_list.txt")
    full_test_name_placeholder = ""

   #Get outputs
    result = summarize_graph(graph.as_graph_def())
    print("This model has {} outputs".format(len(result['outputs'])))

    test_dir = ""
    test_case_name = ""
    test_names = file.split('.')
    for i in range(0,len(test_names)):
        if i == (len(test_names)-1):
            test_case_name = test_names[i]
        test_dir+="/"
        test_dir+=test_names[i]
        full_test_name_placeholder+=test_names[i] + '_'

    #Process each subgraph (test case has multiple iterations)
    for output in result['outputs']:

      index = str(result['outputs'].index(output))
      out_dir = "./pbfiles/" + test_dir
      out_file = test_case_name + '-' + index + '.pb'

      sub_test = full_test_name_placeholder + test_case_name + '-' + index

      print("Testing --- ", sub_test)
      if sub_test in invalid_tests:
        print("Skipping test: {}".format(sub_test))
        continue

      print("File log {} exists?: {}".format(out_dir + '/' + out_file,os.path.exists(out_dir + '/' + out_file)))

      if not os.path.exists(out_dir + '/' + out_file):

        graph_def = tf.compat.v1.graph_util.extract_sub_graph(graph.as_graph_def(), [output])

        if not skipTest(graph_def):
            #Add placeholder node, get new graph with placeholder node added & const op name
            new_graph, node_to_replace = add_placeholder(graph_def)
            new_graph_def = new_graph.as_graph_def()

            if not len(node_to_replace)==0:
              replace_input_to_placeholder(new_graph_def, node_to_replace)

            #Write the new graph
            nodes = [node for node in new_graph_def.node]
            if len(nodes) > 1:
              new_graph_def = tf.compat.v1.graph_util.extract_sub_graph(new_graph_def, [output])
              tf.io.write_graph(new_graph_def, out_dir, out_file , as_text=False)
            else:
              print(nodes)
              print("Skipping graphs with just Placeholder/Const node")

def read_tests_from_file(filename):
    with open(filename) as list_of_tests:
        return [
            line.split('#')[0].rstrip('\n').strip(' ')
            for line in list_of_tests.readlines()
            if line[0] != '#' and line != '\n'
        ]

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-i',
                      '--input_dir',
                      default=[],
                      help='enter input models(.pb) path',
                      required=True)

  args = parser.parse_args()
  files = os.listdir(args.input_dir)
  
  for file in files:
    input_model = args.input_dir + '/' + file
    print("\nProcessing model" + input_model)

    #Get graph
    graph = get_graph(input_model)

    try:
      process_graph(graph, file)
    except Exception as e:
      print("Error processing file:" + input_model)
      print(e)

  print("\nEnd of Script")
