from __future__ import print_function
from os import path
import tensorflow as tf
import mlflow
from mlflow import tensorflow, tracking

MODEL_DIR = path.join(path.dirname(__file__), "..", "models")
MODEL_PATH = path.join(MODEL_DIR, "model1.cpkt")
v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)

inc_v1 = v1.assign(v1 + 1)
dec_v2 = v2.assign(v2 - 1)

init_op = tf.global_variables_initializer()

hello = tf.constant('Hello, Tensorflow!')
saver = tf.train.Saver()

with mlflow.start_run():
    with tf.Session() as sess:
        print(sess.run(hello))
        sess.run(init_op)
        inc_v1.op.run()
        dec_v2.op.run()
        save_path = saver.save(sess,MODEL_PATH )
        print("Model saved in path: %s" % save_path)
        mlflow.tensorflow.log_saved_model(saved_model_dir=MODEL_DIR, signature_def_key="hello_v1", artifact_path="model")
