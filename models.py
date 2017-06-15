""" CNN for text classification
    code adapted from https://github.com/dennybritz/cnn-text-classification-tf
    """

from random import choice
from string import ascii_lowercase,digits



import numpy as np
import pylab
import datetime

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator,ClassifierMixin

class TFBaseEstimator(BaseEstimator):
    def __init__(self):

        self.train_step = None
        self.predict_step = None
        self.is_fitted = False
        self._var_scope = None
        self.session = tf.InteractiveSession()
    
    def _get_var_scope(self):
        """ Create a unique string for the variable scope 
            This is to enable several instantiations of the 
            same class without variable conflicts"""
        if self._var_scope is None:
            self._var_scope = type(self).__name__+ '_'+''.join(choice(ascii_lowercase+digits) for i in range(10))
        return self._var_scope



class TFBaseClassifier(TFBaseEstimator,ClassifierMixin):
    """ a base class for classifier models. 
        this class cannot be instantiated.
        """

    def __init__(self):
        super(TFBaseClassifier,self).__init__()
         
        self.classes_ = None
        self.x = None
        self.y_ = None
        

    def fit(self,X,y):
        self.classes_ = np.unique(targets)
        # targets need to be binarized
        lb = LabelBinarizer()
        bin_targets = lb.fit_transform(targets)
        if bin_targets.shape[1]==1:
            bin_targets = pylab.concatenate([1-bin_targets,bin_targets],axis = 1)
        n_outputs = bin_targets.shape[1]
        n_features = data.shape[1]
        # place holder for the input and output
        self.x = tf.placeholder(tf.float32,[None,n_features])
        self.y_ = tf.placeholder(tf.float32,[None,n_outputs])
        # op for prediction_step
        self.predict_step = self._predict_step()
        # op for train step
        self.train_step = self._train_step()
        # initialize variables
        self.session.run(tf.global_variables_initializer())
        # run the training
        self._train_loop()

        self.isfitted = True

        return self
    
    def predict_proba(self,X):
        if not self.isfitted:
            print 'not fitted'
            return
        
        return self.session.run(self.predict_step,feed_dict={self.x:X.astype(float)})
    
    def predict(self,X):
        if not self.isfitted:
            print 'not fitted'
            return
        proba = self.predict_proba(X)
        return self.classes_[pylab.argmax(proba,axis=1)]
    def _predict_step(self):
        # this needs to be filled
        pass
    def _train_step(self):
        # this needs to be filled
        pass
    def _train_loop(self):
        # this needs to be filled
        pass

class LR_old:
    def __init__(self,iterations = 1000,trainstep = 0.5):
        self.iterations  =iterations
        self.trainstep = trainstep
        self.w = None
        self.b = None
        self.x = None
        self.y = None
        self.session = tf.InteractiveSession()
        self.isfitted = False
        self.classes_ = None

    def fit(self,data,targets):
        self.classes_ = np.unique(targets)
        # targets need to be binarized
        lb = LabelBinarizer()
        bin_targets = lb.fit_transform(targets)
        if bin_targets.shape[1]==1:
            bin_targets = pylab.concatenate([1-bin_targets,bin_targets],axis = 1)
        n_outputs = bin_targets.shape[1]
        n_features = data.shape[1]
        self.x = tf.placeholder(tf.float32,[None,n_features])
        y_ = tf.placeholder(tf.float32,[None,n_outputs])
        self.w = tf.Variable(tf.zeros([n_features,n_outputs]))
        self.b = tf.Variable(tf.zeros([n_outputs]))
        self.y = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
        train_step = tf.train.AdamOptimizer(self.trainstep).minimize(cross_entropy)
        self.session.run(tf.global_variables_initializer())
        

        for i in range(self.iterations):
            print 'iteration ',i
            train_step.run(feed_dict={self.x:data.astype(float),y_:bin_targets.astype(float)})
        self.isfitted = True

    def predict_proba(self,data):
        print self.session.run([self.w,self.b])
        if not self.isfitted:
            print 'not fitted'
            return
        
        return self.session.run(self.y,feed_dict={self.x:data.astype(float)})
    def predict(self,data):
        if not self.isfitted:
            print 'not fitted'
            return
        proba = self.predict_proba(data)
        return self.classes_[pylab.argmax(proba,axis=1)]

def test_LR():
    N =1000
    nclasses = 3
    sep = 2
    data = pylab.randn(N,2)
    targets = pylab.zeros((N))
    for cl in range(nclasses):
        angle = cl * 2*pylab.pi/float(nclasses)
        x = sep*pylab.cos(angle)
        y = sep*pylab.sin(angle)
        targets[cl*N/nclasses:(cl+1)*N/nclasses] = cl
        data[cl*N/nclasses:(cl+1)*N/nclasses,0]+=x
        data[cl*N/nclasses:(cl+1)*N/nclasses,1]+=y

    lr = LR()

    lr.fit(data,targets)

    predictions = lr.predict(data)
    
    for t in pylab.unique(targets):
        pylab.plot(data[targets==t,0],data[targets==t,1],'o')
        
    print (predictions == targets).mean()
    pylab.show()

class TextConvNet:
    def __init__(self,filter_sizes = [2,3,4,5],n_filters = 20,n_hidden = 100,batchsize = 100,dropout=0.5, l2_reg_lambda=0.0,valid_frac = 0.05,iterations = 10000,evaluate_every = 100,print_every = 100,n_jobs=1):
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
        self.h_pool = None
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
        
        
        
        self.isfitted = False

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
        word_vec_length = data.shape[2]
        
        self.x = tf.placeholder(tf.float32,[None,n_words,word_vec_length,1])
        self.y = tf.placeholder(tf.float32,[None,n_outputs])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        l2_loss = tf.constant(0.0)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            
            # Convolution Layer
            filter_shape = [filter_size, word_vec_length, 1, self.n_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
            conv = tf.nn.conv2d(
                self.x,
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
            
            train_step(train_data[batch,:,:,None], train_targets[batch])
            current_step = tf.train.global_step(self.session, global_step)
            if current_step % self.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(test_data[:,:,:,None],test_targets)
                print("")
        self.isfitted = True
    def predict_proba(self,data):
        if not self.isfitted:
            print 'not fitted'
            return
        scores =  self.session.run(self.scores,feed_dict = {self.x: data[:,:,:,None],self.dropout_keep_prob: 1.})
        scores = pylab.exp(scores)
        return scores/scores.sum(axis=1)[:,None]
    def predict(self,data):
        if not self.isfitted:
            print 'not fitted'
            return
        proba = self.predict_proba(data)
        return self.classes_[pylab.argmax(proba,axis=1)]        
class BatchIndGernerator:
    def __init__(self, batchsize,N,iterations):
        self.batchsize = batchsize
        self.N = N
        self.iterations = iterations
        self.currentiteration = 0

    def __iter__(self):
        return self

    def next(self):
        if self.currentiteration > self.iterations:
            raise StopIteration
        else:
            self.currentiteration += 1
            inds = pylab.arange(self.N)
            pylab.shuffle(inds)
            return inds[:self.batchsize]


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
        
        
        
        self.isfitted = False

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
        self.y = tf.placeholder(tf.float32,[None,n_outputs])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
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
        self.isfitted = True
    def predict_proba(self,data):
        if not self.isfitted:
            print 'not fitted'
            return
        # make sure that no unseen words are passed to the embedding layer
        data[data>self.vocab_size] = 0
        scores =  self.session.run(self.scores,feed_dict = {self.x: data,self.dropout_keep_prob: 1.})
        scores = pylab.exp(scores)
        return scores/scores.sum(axis=1)[:,None]
    def predict(self,data):
        if not self.isfitted:
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
        
        
        
        self.isfitted = False

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
        
        self.x = tf.placeholder(tf.float32,[None,n_words,word_vec_length,1])
        self.y = tf.placeholder(tf.float32,[None,n_outputs])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
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
        self.isfitted = True
    def predict_proba(self,data):
        if not self.isfitted:
            print 'not fitted'
            return
        scores =  self.session.run(self.scores,feed_dict = {self.x: data[:,:,:,None],self.dropout_keep_prob: 1.})
        scores = pylab.exp(scores)
        return scores/scores.sum(axis=1)[:,None]
    def predict(self,data):
        if not self.isfitted:
            print 'not fitted'
            return
        proba = self.predict_proba(data)
        return self.classes_[pylab.argmax(proba,axis=1)]  

if __name__ == '__main__':
    import data
    from score import balanced_accuracy as score
    from sklearn.cross_validation import StratifiedKFold
    data,targets,test = data.get_compact_vector_data()
    

    predictions = pylab.zeros_like(targets)
    i=0
    for train,test in StratifiedKFold(targets,n_folds = 3):
        print 'fold ',i
        i+=1
        print data[train].shape,targets[train].shape
        tcn = DeepTextConvNet(iterations = 1000)
        tcn.fit(data[train], targets[train])
        predictions[test] = tcn.predict(data[test])

    print score(predictions, targets)
      
    
