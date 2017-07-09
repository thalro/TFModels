




import numpy as np
import pylab
import datetime
import cPickle as pickle
from copy import deepcopy

import tensorflow as tf
from tensorflow.contrib import keras


from .base import TFBaseEstimator,TFBaseClassifier,tfDtype,npDtype


   



    
    


    

class LogisticRegression(TFBaseClassifier):
    """ a simple implementation of Logistic Regression.

        this ist mainly to test the functionality 
        of TFBaseClassifier
        """

    def __init__(self,C = 0.1,**kwargs):
        super(LogisticRegression, self).__init__(**kwargs)
        self.C = C
    
    
    def _predict_step(self):
        layer = tf.layers.dense(self.x,self.n_outputs,name = 'dense_linear',kernel_regularizer=tf.contrib.layers.l2_regularizer(scale = self.C))
        return layer
    




class TextConvNet(TFBaseClassifier):
    
    def __init__(self,filter_sizes = [2,3],n_filters = 20,n_hidden = 100,dropout=0.5,
                 l2_reg_lambda=0.0,**kwargs):
        
        super(TextConvNet,self).__init__(**kwargs)
        
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.l2_reg_lambda = l2_reg_lambda
        
         
    
    def _predict_step(self):
        x = self.x[:,:,:,None]
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, self.feature_shape[1]]
            conv = tf.layers.conv2d(x,self.n_filters,filter_shape,kernel_initializer =tf.contrib.layers.xavier_initializer_conv2d())
            
            h = tf.nn.relu(conv)
            # Maxpooling over the outputs
            pooled = tf.layers.max_pooling2d(h,pool_size = [self.feature_shape[0] - filter_size + 1,1],strides = [1,1])
            
            pooled_outputs.append(pooled)
        num_filters_total = self.n_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        self.h = tf.layers.dense(self.h_pool_flat,self.n_hidden,activation = tf.nn.relu)
        
        
        self.h_drop = tf.layers.dropout(self.h, self.dropout,training = self.is_training)
        return tf.layers.dense(self.h_drop,self.n_outputs,kernel_initializer = tf.contrib.layers.xavier_initializer())

   

         
class DenseNeuralNet(TFBaseClassifier):
    def __init__(self,n_hiddens = [5],dropout=0.2,batch_normalization = False,**kwargs):
        
        super(DenseNeuralNet,self).__init__(**kwargs)

        self.n_hiddens = n_hiddens
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.total_n_samples = 0

    
    def _predict_step(self):
        
        last_activation =self.x

        for i,n_hidden in enumerate(self.n_hiddens):
            linear = tf.layers.dense(last_activation,n_hidden,kernel_initializer = tf.contrib.layers.xavier_initializer())
            
            if self.batch_normalization:
                linear = tf.layers.batch_normalization(linear,training = self.is_training)
            
            activation = tf.nn.relu(linear)
            last_activation = tf.layers.dropout(activation,rate = self.dropout,training = self.is_training)

        output = tf.layers.dense(last_activation,self.n_outputs,kernel_initializer = tf.contrib.layers.xavier_initializer())
        return output




class ConvolutionalNeuralNet(TFBaseClassifier):

    def __init__(self,n_filters = [5,5],filter_sizes = [[3,3],[3,3]],strides =[1,1],pooling = [2],pooling_strides = [1],n_hiddens = [5],dropout=0.2,batch_normalization = True,**kwargs):
        
        super(ConvolutionalNeuralNet,self).__init__(**kwargs)
        
        self.n_filters = n_filters
        assert len(filter_sizes)==len(n_filters) ,  ValueError('n_filters and filter_sizes must be lists of same length')

        self.filter_sizes = filter_sizes
        assert len(strides)==len(n_filters), ValueError('n_filters and strides must be lists of same length')
        self.strides = strides
        assert len(pooling)==len(n_filters)-1 or len(pooling)==len(n_filters),  ValueError('pooling  must have same length as n_filters or one element less')
        assert len(pooling)==len(pooling_strides),ValueError('pooling be same length as pooling strides')
        self.pooling = pooling
        self.pooling_strides = pooling_strides
        if len(pooling)<len(filter_sizes):
            self.pooling += [None]
            self.pooling_strides += [None]
       

        self.n_hiddens = n_hiddens
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.total_n_samples = 0

    def _predict_step(self):
        
        last_activation =self.x
        if self.batch_normalization:
                last_activation = tf.layers.batch_normalization(last_activation,training = self.is_training)
        for i,(n_filter,filter_size,stride,pooling,pooling_strides) in enumerate(zip(self.n_filters,
                                                                                     self.filter_sizes,
                                                                                     self.strides,
                                                                                     self.pooling,
                                                                                     self.pooling_strides)):
            conv = tf.layers.conv2d(last_activation,n_filter,filter_size,strides=stride)
            if self.batch_normalization:
                conv = tf.layers.batch_normalization(conv,training = self.is_training)
            activation = tf.nn.relu(conv)
            if pooling is None:
                # last layer, pool over whole field
                pooling = activation.shape[1:3]
                pooling_strides = 1
            last_activation = tf.layers.max_pooling2d(activation,pooling,pooling_strides)
        flat_shape =  int(last_activation.shape[1]*last_activation.shape[2]*last_activation.shape[3])
        
        last_activation = tf.reshape(last_activation,[-1,flat_shape])
        for i,n_hidden in enumerate(self.n_hiddens):
            linear = tf.layers.dense(last_activation,n_hidden,kernel_initializer = tf.contrib.layers.xavier_initializer())
            
            if self.batch_normalization:
                linear = tf.layers.batch_normalization(linear,training = self.is_training)
            
            activation = tf.nn.relu(linear)
            last_activation = tf.layers.dropout(activation,rate = self.dropout,training = self.is_training)

        output = tf.layers.dense(last_activation,self.n_outputs,kernel_initializer = tf.contrib.layers.xavier_initializer())
        return output


class Resnet50(TFBaseClassifier):
    def __init__(self,fixed_epochs =10,**kwargs):
        super(Resnet50, self).__init__(**kwargs)
        keras.backend.set_session(self.session)
        self.train_feed_dict = {keras.backend.learning_phase():True}
        self.test_feed_dict = {keras.backend.learning_phase():False}
        self.epoch_count = 0
        self.fixed_epochs = fixed_epochs
        self.bottom_fixed = True
        self.base_model = None
    def _predict_step(self):
        input = tf.layers.batch_normalization(self.x,training = self.is_training)
        self.base_model = keras.applications.ResNet50(include_top = False,input_tensor = input)
        for layer in self.base_model.layers:
            layer.trainable = False
        last_activation = self.base_model.output
        flat_shape =  int(last_activation.shape[1]*last_activation.shape[2]*last_activation.shape[3])
        
        last_activation = tf.reshape(last_activation,[-1,flat_shape])
        output = tf.layers.dense(last_activation,self.n_outputs,kernel_initializer = tf.contrib.layers.xavier_initializer())
        
        return output
    def _iteration_callback(self):
        if self.epoch_count ==0:
            # reload the model because pretrained weights are overwritten by initialization 
            self.base_model = keras.applications.ResNet50(include_top = False,input_tensor = self.x)
            for layer in self.base_model.layers:
                layer.trainable = False
        self.epoch_count+=1
        if self.epoch_count==self.fixed_epochs:
            self.bottom_fixed = False
            for layer in base_model.layers:
                layers.trainable = True








class EmbeddingTextConvNet:
    def __init__(self,embedding_size = 100,filter_sizes = [2,3,4,5],n_filters = 20,n_hidden = 100,batchsize = 100,dropout=0.5, l2_reg_lambda=0.0,valid_frac = 0.05,iterations = 10000,evaluate_every = 100,print_every = 100,n_jobs=1):
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.n_hidden = n_hidden
        self.batchsize = batchsize
        self.dropout_keep = 1-dropout
        self.l2_reg_lambda = l2_reg_lambda
        self.valid_frac = valid_frac
        self.iterations  =iterations
        self.evaluate_every = evaluate_every
        self.print_every = print_every
        self.n_jobs = n_jobs

        self.x = None
        self.embedded_chars = None
        self.embedded_chars_expanded = None
        self.h_pool = None
        self.h_pool_flat = None 
        self.h_drop= None
        self.dropout_keep_prob = None 
        self.scores = None
        self.loss = None
        self.predictions = None
        self.accuracy = None 
        self.vocab_size = None
        try:
            tf.reset_default_graph()
        except:
            print 'could not reset default graph'
        
        if n_jobs!=-1:
            config = tf.ConfigProto(intra_op_parallelism_threads=n_jobs, inter_op_parallelism_threads=n_jobs, \
                        allow_soft_placement=True, device_count = {'CPU': n_jobs})
            self.session = tf.InteractiveSession(config=config)
        else:
            self.session = tf.InteractiveSession()
        
        
        
        self.is_fitted = False

        self.classes_ = None
    

    def fit(self,data,targets):

        self.classes_ = np.unique(targets)
        # targets need to be binarized
        lb = LabelBinarizer()
        bin_targets = lb.fit_transform(targets)
        if bin_targets.shape[1]==1:
            bin_targets = pylab.concatenate([1-bin_targets,bin_targets],axis = 1)

        # define a subset of the data for evaluation
        inds = pylab.arange(data.shape[0])
        pylab.shuffle(inds)
        test_inds = inds[:int(self.valid_frac*len(inds))]
        train_inds = inds[int(self.valid_frac*len(inds)):]
        train_data = data[train_inds]
        test_data = data[test_inds]
        train_targets = bin_targets[train_inds]
        test_targets = bin_targets[test_inds]

        
        n_outputs = bin_targets.shape[1]
        n_words = data.shape[1]

        
        
        
        self.x = tf.placeholder(tf.int32,[None,n_words])
        self.y = tf.placeholder(tfDtype,[None,n_outputs])
        self.dropout_keep_prob = tf.placeholder(tfDtype)
        l2_loss = tf.constant(0.0)
        
        # create the embedding layer
        # compute vocab size from data
        self.vocab_size = data.max()+1
        
        W = tf.Variable(
        tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
        name="W")
        self.embedded_chars = tf.nn.embedding_lookup(W, self.x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            
            # Convolution Layer
            filter_shape = [filter_size, self.embedding_size, 1, self.n_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
            conv = tf.nn.conv2d(
                self.embedded_chars_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, n_words - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.n_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # add a hidden layer
        W = tf.Variable(tf.truncated_normal((num_filters_total,self.n_hidden), stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[self.n_hidden]))
        self.h = tf.nn.relu(tf.matmul(self.h_pool_flat, W) + b)
         

        # Add dropout
       
        self.h_drop = tf.nn.dropout(self.h, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        W = tf.get_variable(
                "W",
                shape=[self.n_hidden, n_outputs],
                initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[n_outputs]), name="b")
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y)
        self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        # Initialize all variables
        self.session.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              self.x: x_batch,
              self.y: y_batch,
              self.dropout_keep_prob: self.dropout_keep
            }
            _, step, loss, accuracy = self.session.run(
                [train_op, global_step, self.loss, self.accuracy],
                feed_dict)

            current_step = tf.train.global_step(self.session, global_step)
            if current_step % self.print_every == 0:
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              self.x: x_batch,
              self.y: y_batch,
              self.dropout_keep_prob: 1.
            }
            step, loss, accuracy = self.session.run(
                [global_step, self.loss, self.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()

            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        
        batch_inds = BatchIndGernerator(self.batchsize, train_data.shape[0], self.iterations)
        
        for batch in batch_inds:
            
            train_step(train_data[batch], train_targets[batch])
            current_step = tf.train.global_step(self.session, global_step)
            if current_step % self.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(test_data,test_targets)
                print("")
        self.is_fitted = True
    def predict_proba(self,data):
        if not self.is_fitted:
            print 'not fitted'
            return
        # make sure that no unseen words are passed to the embedding layer
        data[data>self.vocab_size] = 0
        scores =  self.session.run(self.scores,feed_dict = {self.x: data,self.dropout_keep_prob: 1.})
        scores = pylab.exp(scores)
        return scores/scores.sum(axis=1)[:,None]
    def predict(self,data):
        if not self.is_fitted:
            print 'not fitted'
            return
        proba = self.predict_proba(data)
        return self.classes_[pylab.argmax(proba,axis=1)]        

class DeepTextConvNet:
    def __init__(self,filter_sizes = [2,2],n_filters = 20,n_hidden = 100,batchsize = 100,dropout=0.5, l2_reg_lambda=0.0,valid_frac = 0.05,iterations = 10000,evaluate_every = 100,print_every = 100,n_jobs=1):
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.n_hidden = n_hidden
        self.batchsize = batchsize
        self.dropout_keep = 1-dropout
        self.l2_reg_lambda = l2_reg_lambda
        self.valid_frac = valid_frac
        self.iterations  =iterations
        self.evaluate_every = evaluate_every
        self.print_every = print_every
        self.n_jobs = n_jobs

        self.x = None
        self.pooled1 = None
        self.pooled2 = None
        self.h_pool_flat = None 
        self.h_drop= None
        self.h = None
        self.dropout_keep_prob = None 
        self.scores = None
        self.loss = None
        self.predictions = None
        self.accuracy = None 
        
        try:
            tf.reset_default_graph()
        except:
            print 'could not reset default graph'
        
        if n_jobs!=-1:
            config = tf.ConfigProto(intra_op_parallelism_threads=n_jobs, inter_op_parallelism_threads=n_jobs, \
                        allow_soft_placement=True, device_count = {'CPU': n_jobs})
            self.session = tf.InteractiveSession(config=config)
        else:
            self.session = tf.InteractiveSession()
        
        
        
        self.is_fitted = False

        self.classes_ = None
    

    def fit(self,data,targets):
        print data.shape
        self.classes_ = np.unique(targets)
        # targets need to be binarized
        lb = LabelBinarizer()
        bin_targets = lb.fit_transform(targets)
        if bin_targets.shape[1]==1:
            bin_targets = pylab.concatenate([1-bin_targets,bin_targets],axis = 1)

        # define a subset of the data for evaluation
        inds = pylab.arange(data.shape[0])
        pylab.shuffle(inds)
        test_inds = inds[:int(self.valid_frac*len(inds))]
        train_inds = inds[int(self.valid_frac*len(inds)):]
        train_data = data[train_inds]
        test_data = data[test_inds]
        train_targets = bin_targets[train_inds]
        test_targets = bin_targets[test_inds]

        n_outputs = bin_targets.shape[1]
        n_words = data.shape[1]
        word_vec_length = data.shape[2]
        
        self.x = tf.placeholder(tfDtype,[None,n_words,word_vec_length,1])
        self.y = tf.placeholder(tfDtype,[None,n_outputs])
        self.dropout_keep_prob = tf.placeholder(tfDtype)
        l2_loss = tf.constant(0.0)
        # Create a convolution + maxpool layer for each filter size
        #pooled_outputs = []
        
            
        # 1st Convolution Layer
        conv1 = tf.layers.conv2d(
        inputs=self.x,
        filters=self.n_filters,
        kernel_size=[self.filter_sizes[0], word_vec_length],
        padding="valid",
        activation=tf.nn.relu)

        # Maxpooling over the outputs
        self.pooled1  = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 1], strides=1)


        # 2ns Convolution Layer
        conv2 = tf.layers.conv2d(
        inputs=self.pooled1,
        filters=self.n_filters,
        kernel_size=[self.filter_sizes[1],1],
        padding="valid",
        activation=tf.nn.relu)
        # Maxpooling over the outputs
        self.pooled2  = tf.layers.max_pooling2d(inputs=conv2, pool_size=[n_words-sum(self.filter_sizes)+1, 1], strides=1)
        # Combine all the pooled features
        
        
        self.h_pool_flat = tf.reshape(self.pooled2, [-1, self.n_filters])
        
        if self.n_hidden>0:
            # add a hidden layer
            W = tf.Variable(tf.truncated_normal((self.n_filters,self.n_hidden), stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[self.n_hidden]))
            self.h = tf.nn.relu(tf.matmul(self.h_pool_flat, W) + b)
             

            # Add dropout
           
            self.h_drop = tf.nn.dropout(self.h, self.dropout_keep_prob)
        else:
            # Add dropout
           
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            

        # Final (unnormalized) scores and predictions
        W = tf.get_variable(
                "W",
                shape=[self.n_hidden, n_outputs],
                initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[n_outputs]), name="b")
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y)
        self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        # Initialize all variables
        self.session.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              self.x: x_batch,
              self.y: y_batch,
              self.dropout_keep_prob: self.dropout_keep
            }
            _, step, loss, accuracy = self.session.run(
                [train_op, global_step, self.loss, self.accuracy],
                feed_dict)

            current_step = tf.train.global_step(self.session, global_step)
            if current_step % self.print_every == 0:
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              self.x: x_batch,
              self.y: y_batch,
              self.dropout_keep_prob: 1.
            }
            step, loss, accuracy = self.session.run(
                [global_step, self.loss, self.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()

            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        
        batch_inds = BatchIndGernerator(self.batchsize, train_data.shape[0], self.iterations)
        
        for batch in batch_inds:
            
            train_step(train_data[batch,:,:,None], train_targets[batch])
            current_step = tf.train.global_step(self.session, global_step)
            if current_step % self.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(test_data[:,:,:,None],test_targets)
                print("")
        self.is_fitted = True
    def predict_proba(self,data):
        if not self.is_fitted:
            print 'not fitted'
            return
        scores =  self.session.run(self.scores,feed_dict = {self.x: data[:,:,:,None],self.dropout_keep_prob: 1.})
        scores = pylab.exp(scores)
        return scores/scores.sum(axis=1)[:,None]
    def predict(self,data):
        if not self.is_fitted:
            print 'not fitted'
            return
        proba = self.predict_proba(data)
        return self.classes_[pylab.argmax(proba,axis=1)]  




    
