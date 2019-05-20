import tensorflow as tf
from argparse import ArgumentParser
from tensorflow.python.framework import graph_util
import os,sys

parser = ArgumentParser()
parser.add_argument("metaFile", help="Meta filename")
parser.add_argument("modelName", help="Model Name")
args = parser.parse_args()

saver = tf.train.import_meta_graph(args.metaFile, clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, args.modelName)

output_node_names="y_pred"
output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session
            input_graph_def, # input_graph_def is useful for retrieving the nodes 
            output_node_names.split(",")  
)

output_graph="/fcn_model.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
 
sess.close()
