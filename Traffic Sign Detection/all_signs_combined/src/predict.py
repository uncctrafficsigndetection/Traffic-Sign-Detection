import numpy as np
import time
from sample_model import Model
from data_loader import data_loader
from generator import Generator  


checkpoint_dir='tf_data/sample_model'
X='C:/Users/Karthick/Desktop/cvproject/data/5/00000_00000.ppmspeed_2_.ppm'
M = Model(mode = 'test')
yhat = M.predict(X = X, checkpoint_dir = checkpoint_dir)
	


# save_dir="C:/Users/Karthick/Desktop/cvproject/speedlimitckp/"
# #saver = tf.train.Saver()
# sess = tf.Session()
# saver = tf.train.import_meta_graph('C:/Users/Karthick/Desktop/cvproject/src/tf_data/sample_model/model_epoch70.ckpt.meta')
# saver.restore(sess,tf.train.latest_checkpoint('C:/Users/Karthick/Desktop/cvproject/src/tf_data/sample_model/'))
# #checkpoint_name = tf.train.latest_checkpoint(save_dir)
# #saver.restore(sess, checkpoint_name)
# yhat_numpy = sess.run(yhat, feed_dict = {X : X, keep_prob: 1.0})
# print(yhat_numpy)

# #C:/Users/Karthick/Desktop/cvproject/src/tf_data/sample_model