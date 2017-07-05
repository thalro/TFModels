
import os
import cPickle as pickle
import numpy as np

from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf

tfDtype = tf.float32
npDtype = np.float32



class BatchIndGernerator:
    def __init__(self, batchsize,N,iterations,shuffle = True,start_iteration = 0):
        
        if batchsize is None:
            self.batchsize = N
        else:
            self.batchsize = batchsize

        self.N = N
        self.iterations = iterations
        self.currentiteration = start_iteration
        self.queue = []
        self.shuffle = shuffle

    def __iter__(self):
        return self

    def next(self):
        if self.iterations == 0 or \
           len(self.queue)==0 and self.currentiteration >= self.iterations:

            raise StopIteration
        else:
            if len(self.queue) ==0 :

                self.currentiteration += 1
                inds = np.arange(self.N)
                if self.shuffle:
                    np.random.shuffle(inds)
                self.queue = inds
            inds = self.queue[:self.batchsize]
            self.queue = self.queue[self.batchsize:]
            return inds,self.currentiteration


class TFBaseEstimator(BaseEstimator):
    def __init__(self,n_jobs = 1):
        self.train_step = None
        self.predict_step = None
        self.is_fitted = False
        self._var_scope = None
        self.tf_vars = {}

        
        try:
            tf.reset_default_graph()
        except:
            print 'could not reset default graph'

        if n_jobs!=-1:
            config = tf.ConfigProto(intra_op_parallelism_threads=n_jobs, inter_op_parallelism_threads=n_jobs, \
                        allow_soft_placement=True, device_count = {'CPU': n_jobs})
            self.session = tf.Session(config=config)
        else:
            self.session = tf.Session()

        

    def get_tf_vars_as_ndarrays(self):
        tf_vars = {}
        for var in self.tf_vars.keys():
            tf_vars[var] = self.session.run(self.tf_vars[var]).astype(npDtype)
        return tf_vars
    def save(self,fname):
        
        #tf_vars  =self.get_tf_vars_as_ndarrays()

        params = self.get_params()

        pickle.dump((params,self.is_fitted), open(fname+'.params','w'),protocol = 2)
        
        # save the tf session

        saver = tf.train.Saver()
        saver.save(self.session,fname+'.session')

    def load(self,fname):
        params,is_fitted = pickle.load(open(fname+'.params'))
        self.set_params(**params)
        
        self.if_fitted = is_fitted

        saver = tf.train.import_meta_graph(fname+'.session.meta')
        saver.restore(self.session,fname+'.session')
    def __del__(self):
        self.session.close()
        del self.session
        

class TFBaseClassifier(TFBaseEstimator,ClassifierMixin):
    """ a base class for classifier models. 
        this class should be instantiated.
        """

    def __init__(self,random_state=None,learning_rate = 0.1,learning_rates=None,iterations = 10,batchsize = None,print_interval= 10,verbose = False,output_type ='softmax',*kwargs):
        super(TFBaseClassifier, self).__init__(*kwargs) 

        self.classes_ = None
        self.n_classes = None
        self.x = None
        self.y_ = None
        self.feature_shape = None
        self.n_outputs = None
        self.warm_start = False
        self.random_state = random_state
        # learning rate and iterations can also be lists
        if not isinstance(iterations, (list, tuple, np.ndarray)):
            iterations = [iterations]
        if learning_rates is None:
            learning_rates = [learning_rate]
        assert len(learning_rates)==len(iterations), 'learning_rates and iterations must have same length'
        self.learning_rates = learning_rates
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batchsize = batchsize
        self.print_interval = print_interval
        self.verbose = verbose
        if output_type not in ['softmax','sigmoid']:
            raise ValueError('output_type must be either softmax or sigmoid')
        self.output_type = output_type
        self.is_training = tf.placeholder(tf.bool)
        
        

    def fit(self,X,y,warm_start = False):
        
        original_batchsize = self.batchsize
        if self.batchsize is None:
            self.batchsize = X.shape[0]
        self.warm_start = warm_start
        if self.random_state is not None:
            
            np.random.seed(self.random_state)
            tf.set_random_seed(self.random_state)


        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        # targets need to be binarized
        lb = LabelBinarizer()
        bin_y = lb.fit_transform(y)
        if bin_y.shape[1]==1:
            bin_y = np.concatenate([1-bin_y,bin_y],axis = 1)
        self.n_outputs = bin_y.shape[1]
        self.feature_shape = list(X.shape[1:])
        
       
        
        if not self.is_fitted:
            # place holder for the input and output
            self.x = tf.placeholder(tf.float32,[None]+self.feature_shape)
            self.y = tf.placeholder(tf.float32,[None,self.n_outputs])
            # create graph
            self.predict_step = self._predict_step()
            # op for train step
            self.train_step = self._train_step()
            if self.output_type == 'softmax':
                self.prediction = tf.nn.softmax(self.predict_step)
            elif self.output_type == 'sigmoid':
                self.prediction = tf.nn.sigmoid(self.predict_step)
        
        # initialize variables
        if not self.warm_start:
            self.session.run(tf.global_variables_initializer())
        # run the training
        self._train_loop(X,bin_y)
        self.is_fitted = True
        self.batchsize = original_batchsize
        return self
    
    def predict_proba(self,X):
        if not self.is_fitted:
            print 'not fitted'
            return

        output = []
        for batch,i in BatchIndGernerator(self.batchsize, X.shape[0], 1,shuffle = False):
            
            output.append(self.session.run(self.prediction,feed_dict={self.x:X[batch].astype(float),self.is_training:False}))
        return np.concatenate(output,axis =0)
    
    def predict(self,X):
        if not self.is_fitted:
            print 'not fitted'
            return
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba,axis=1)]
    
    def _train_loop(self,X,y):
        #ensure that iterations is list in case it has been changed
        if not isinstance(self.iterations, (list, tuple, np.ndarray)):
            self.iterations = [self.iterations]
        iteration = 0
        
        for iterations,learning_rate in zip(self.iterations,self.learning_rates):
            self.learning_rate = learning_rate
            for i,(batch,iteration) in enumerate(BatchIndGernerator(self.batchsize, X.shape[0], iterations+iteration,start_iteration = iteration)):
                self.session.run(self.train_step,feed_dict = {self.x:X[batch],self.y:y[batch],self.is_training:True})
                
                if self.verbose and  i%self.print_interval ==0:
                    loss = self.session.run(self._loss_func(),feed_dict = {self.x:X[batch],self.y:y[batch],self.is_training:False})
                    print 'iteration ',iteration,', batch ',i ,', loss ',loss,', learning rate ',learning_rate
            
    def _loss_func(self):
        # override for more fancy stuff
        if self.output_type == 'softmax':
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.predict_step))+tf.reduce_sum(tf.losses.get_regularization_losses())
        elif self.output_type == 'sigmoid':
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.predict_step))+tf.reduce_sum(tf.losses.get_regularization_losses())
        return loss
    def _train_step(self):
        #override for more fancy stuff
        loss = self._loss_func() 
        # this is needed for so that batch_normalization forks
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            
            train_op =  tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return train_op
    
    def _create_graph(self):
        # this needs to be filled
        raise NotImplementedError

    def _predict_step(self):
        # this needs to be filled
        raise NotImplementedError
   