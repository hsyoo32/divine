"""
 SiDE: Feature Learning in Signed Directed Networks
 Authors: Junghwan Kim(kjh900809@snu.ac.kr), Haekyu Park(hkpark627@snu.ac.kr),
          Ji-Eun Lee(dreamhunter@snu.ac.kr), U Kang (ukang@snu.ac.kr)
  Data Mining Lab., Seoul National University

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

-------------------------------------------------------------------------
File: train.py
 - A file implementing optimization phase of side model

Version: 1.0
"""

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from embedding.model import Side
from time import time
import numpy as np


def train(config):
  """
  Build side model object and train for epochs_to_train epochs
  Save tensorflow checkpoint file and summary file for later usage

  :param config: a dictionary containing keys like
   'num_walks', 'embed_dim', 'window_size', 'neg_sample_size', 'damping_factor', 'balance_factor',
    'regularization_param', 'batch_size', 'learning_rate', 'clip_norm',
     'epochs_to_train', 'summary_interval', 'save_interval', 'embed_path'

  Returns
  -------
  side model object
  """
  t0 = time()
  t1 = time()
  t2 = time()
  with tf.Session() as sess:
    with tf.device("/cpu:0"):
      model = Side(config, sess)

    for i in range(config['epochs_to_train']):
      start_time = time()

      model.train()

      # '''
      # nayoun edited
      W, = sess.run([model._W_target])
      W_, = sess.run([model._W_context])
      b_in_pos, b_in_neg, b_out_pos, b_out_neg = sess.run(
        [model._b_in_pos, model._b_in_neg, model._b_out_pos, model._b_out_neg])

      np.savetxt(config['embed_path'] + "_ep{}.bias".format(i), np.vstack([b_in_pos, b_in_neg, b_out_pos, b_out_neg]).T)
      np.savetxt(config['embed_path'] + "_ep{}.emb".format(i), W)
      np.savetxt(config['embed_path'] + "_ep{}.emb2".format(i), W_)

      t2 = time()
      print("learn embedding in", t2 - t1, " epoch ", i)
      t1 = t2

      end_time = time()
      # nayoun revised
      with open("./result_time/{}_time.txt".format(config['dataset']), mode="a+") as f_tr:
        f_tr.writelines("{} epoch time: {} seconds \n".format(i, end_time - start_time))


      # '''



  '''
  # original code
    W, = sess.run([model._W_target])
    W_, = sess.run([model._W_context])
    b_in_pos, b_in_neg, b_out_pos, b_out_neg = sess.run(
      [model._b_in_pos, model._b_in_neg, model._b_out_pos, model._b_out_neg])

  np.savetxt(config['embed_path'] + ".bias", np.vstack([b_in_pos, b_in_neg, b_out_pos, b_out_neg]).T)
  np.savetxt(config['embed_path'] + ".emb", W)
  np.savetxt(config['embed_path'] + ".emb2", W_)
  '''
  print("learn embedding in", time() - t0)
  return model
