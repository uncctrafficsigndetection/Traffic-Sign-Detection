import sys, os
import numpy as np
import tensorflow as tf
from datetime import datetime

from data_feeder import data_feeder
from generator import Generator

from alexnet import AlexNet

class Model(object):
    def __init__(self, mode = 'train'):
        
        if mode == 'train' or mode == 'test':
            self.mode = mode
        else:
            print('Mode must be train or test.')

        # Learning params
        self.learning_rate = 1e-3
        self.minibatch_size = 32
        self.num_iterations = 1500
        self.train_test_split = 0.7
        
        self.keep_rate = 0.6
        
        self.num_layers_to_load = 5
        
        self.display_step = 5
        self.save_step = 100
        
        self.save_dir = "tf_data\sample_model"
        
        self.input_image_size = (227, 227)
        self.label_indices = {'20': 0, '30': 1, '50': 2,
                              '60': 3, '70': 4, '80': 5,
                              '80x': 6, '100': 7, '120': 8}
        self.labels_ordered = list(self.label_indices)
        self.num_classes = len(self.label_indices)
        
        #Go ahead and build graph on initialization:
        self.build_graph()

        #Set to true with we create a tf session.
        self.initialized = False
        
    def build_graph(self):
        
        self.X = tf.placeholder(tf.float32, [None, 
                                        self.input_image_size[0], 
                                        self.input_image_size[1], 
                                        3])

        self.y = tf.placeholder(tf.float32, [None, self.num_classes])
        self.keep_prob = tf.placeholder(tf.float32)
        
        self.AN = AlexNet(self.X, self.keep_prob, self.num_layers_to_load)

        alexnet_out = self.AN.pool5
        
        print(alexnet_out)
        
        self.flattened = tf.reshape(alexnet_out, [-1, 6*6*256])
        
        #Dense layer followed by a dropout layer
        self.dense1 = tf.layers.dense(
            inputs=self.flattened, 
            units=1024, 
            activation=tf.nn.relu,
            name="my_dense1"
            )
        print("D1 Shape: ", self.dense1.get_shape())
        
        self.dropout1 = tf.layers.dropout(
            inputs=self.dense1,
            rate=0.4,
            training = self.mode == tf.estimator.ModeKeys.TRAIN,
            name="my_drop1"
            )
        print("Drop1 Shape: ", self.dropout1.get_shape())
        
        #Smaller dense layer followed by a dropout layer
        self.dense2 = tf.layers.dense(
            inputs = self.dropout1,
            units=128,
            activation=tf.nn.relu,
            name="my_dense2"
            )
        print("D2 Shape: ", self.dense2.get_shape())
        
        self.dropout2 = tf.layers.dropout(
            inputs=self.dense2,
            rate=0.4,
            training = self.mode == tf.estimator.ModeKeys.TRAIN,
            name="my_drop2"
            )
        print("Ddrop2 Shape: ", self.dropout2.get_shape())
        
        #Output of 3 for the three available classes
        self.output = tf.layers.dense(
            inputs=self.dropout2,
            units=9,
            name="my_out"
            )
        print("Output Shape: ", self.output.get_shape())
        
        self.yhat = self.output
        print("yHat Shape: ", self.yhat.get_shape())
        
        self.trainable_variable_names = ['conv5', 'my_dense1', 'my_dense2', 'my_out']

    def train(self):
        cost = tf.losses.sparse_softmax_cross_entropy(tf.argmax(self.y, axis=1), self.output)
    
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] \
                                                        in self.trainable_variable_names]
            
        # Train op
        with tf.name_scope("train"):
            # Get gradients of all trainable variables
            gradients = tf.gradients(cost, var_list)
            gradients = list(zip(gradients, var_list))
    
            # Create optimizer and apply gradient descent to the trainable variables
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            train_op = optimizer.apply_gradients(grads_and_vars=gradients)
    
    
        # Add gradients to summary
        for gradient, var in gradients:
            tf.summary.histogram(var.name + '/gradient', gradient)
    
        # Add the variables we train to the summary
        for var in var_list:
            tf.summary.histogram(var.name, var)
    
        # Add the loss to summary
        tf.summary.scalar('cross_entropy', cost)
    
        # Evaluation op: Accuracy of the model
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.yhat, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
        # Add the accuracy to the summary
        tf.summary.scalar('accuracy', accuracy)
    
        # Merge all summaries together
        merged_summary = tf.summary.merge_all()
    
        # Initialize the FileWriter
        train_writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'train'))
        test_writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'test'))
    
        # Initialize an saver for store model checkpoints
        # We'll save the pretrained alexnet weights and the new weight values that we train:
        var_list_to_save = [v for v in tf.trainable_variables() if v.name.split('/')[0] in \
                                        self.AN.all_variable_names + self.trainable_variable_names]
    
        saver = tf.train.Saver(var_list = var_list_to_save)
    
    
        #Load up data from image files. 
        data = data_feeder(label_indices = self.label_indices, 
                   train_test_split = self.train_test_split,
                   input_image_size = self.input_image_size, 
                   data_path = 'GTSRB\Final_Training\SpeedLimitSigns')
    
        #Setup minibatch generators
        G = Generator(data.train.X, data.train.y, minibatch_size = self.minibatch_size)
        GT = Generator(data.test.X, data.test.y, minibatch_size = self.minibatch_size)
    
        #Launch tf session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.initialized = True
    
        # Add the model graph to TensorBoard
        train_writer.add_graph(self.sess.graph)
    
        # Load the pretrained weights into the alexnet portion of our graph
        self.AN.load_initial_weights(self.sess)
    
        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          self.save_dir))
    
    
        #And Train
        for i in range(self.num_iterations):
            G.generate()
            # And run the training op
            self.sess.run(train_op, feed_dict={self.X: G.X, self.y: G.y, self.keep_prob: self.keep_rate})
    
    
            # Generate summary with the current batch of data and write to file
            if i % self.display_step == 0:
                s = self.sess.run(merged_summary, feed_dict={self.X: G.X, self.y: G.y, self.keep_prob: 1.0})
                train_writer.add_summary(s, i)
    
                GT.generate()
                s = self.sess.run(merged_summary, feed_dict={self.X: GT.X, self.y: GT.y, self.keep_prob: 1.0})
                test_writer.add_summary(s, i)
    
                train_acc = self.sess.run(accuracy, feed_dict={self.X: G.X, self.y: G.y, self.keep_prob: 1.0})
                test_acc = self.sess.run(accuracy, feed_dict={self.X: GT.X, self.y: GT.y, self.keep_prob: 1.0})
                               
                print(i, ' iterations,', str(G.num_epochs), 'epochs, train accuracy = ', train_acc, ', test accuracy = ', test_acc)
                
            if i % self.save_step == 0 and i > 0: 
                print("{} Saving checkpoint of model...".format(datetime.now()))
    
                # save checkpoint of the model
                checkpoint_name = os.path.join(self.save_dir,
                                               'model_epoch'+str(G.num_epochs)+'.ckpt')
                save_path = saver.save(self.sess, checkpoint_name)
                
                print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                               checkpoint_name))
    
    
    def predict(self, X, checkpoint_dir = None):
        '''
        X: Numpy array of dimension [n, 227, 227, 3].
        Note that n here may not be the same as the minibatch_size defined above.
    
        Returns:
        yhat_numpy: A numpy array of dimension [n x 3] containing the predicted 
        one hot labels for each image passed in X. 
        '''
        if not self.initialized:
            self.restore_from_checkpoint(checkpoint_dir)
    
        yhat_numpy = self.sess.run(self.yhat, feed_dict = {self.X : X, self.keep_prob: 1.0})
    
    
        return yhat_numpy
    
    
    def restore_from_checkpoint(self, checkpoint_dir = None):
        '''
        Restore model from most recent checkpoint in save dir.
        '''
    
        saver = tf.train.Saver()
        self.sess = tf.Session()
    
        #Load latest checkpont, use savedir if we're not given a checkpoint dir:
        if checkpoint_dir is None:
            checkpoint_name = tf.train.latest_checkpoint(self.save_dir)
        else:
            checkpoint_name = tf.train.latest_checkpoint(checkpoint_dir)
    
        print(checkpoint_name)
    
        #Resore session
        saver.restore(self.sess, checkpoint_name)
        self.initialized = True
    
        
            