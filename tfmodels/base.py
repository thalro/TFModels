
import os
import cPickle as pickle
import numpy as np

from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf


class BatchIndGernerator:
    def __init__(self, batchsize,N,iterations):
        
        if batchsize is None:
            self.batchsize = N
        else:
            self.batchsize = batchsize

        self.N = N
        self.iterations = iterations
        self.currentiteration = 0

    def __iter__(self):
        return self

    def next(self):
        if self.iterations == 0 or \
           self.iterations is not None and self.currentiteration > self.iterations:

            raise StopIteration
        else:
            self.currentiteration += 1
            inds = np.arange(self.N)
            np.random.shuffle(inds)
            return inds[:self.batchsize]


class TFBaseEstimator(BaseEstimator):
    def __init__(self,n_jobs = 1):
        self.train_step = None
        self.predict_step = None
        self.is_fitted = False
        self._var_scope = None
        self.tf_vars = {}

        if n_jobs!=-1:
            config = tf.ConfigProto(intra_op_parallelism_threads=n_jobs, inter_op_parallelism_threads=n_jobs, \
                        allow_soft_placement=True, device_count = {'CPU': n_jobs})
            self.session = tf.Session(config=config)
        else:
            self.session = tf.Session()

    def get_tf_vars_as_ndarrays(self):
        tf_vars = {}
        for var in self.tf_vars.keys():
            tf_vars[var] = self.session.run(self.tf_vars[var])
        return tf_vars
    def save(self,fname):
        
        tf_vars  =self.get_tf_vars_as_ndarrays()

        params = self.get_params()

        pickle.dump((params,tf_vars,self.is_fitted), open(fname+'.pickle','w'),protocol = 2)
        

    def load(self,fname):
        params,tf_vars,is_fitted = pickle.load(open(fname+'.pickle'))
        self.set_params(**params)
        for k in tf_vars.keys():
            self.tf_vars[k] = tf.Variable(tf_vars[k])
        self.if_fitted = is_fitted

    
    def __del__(self):
        self.session.close()
        del self.session
        

class TFBaseClassifier(TFBaseEstimator,ClassifierMixin):
    """ a base class for classifier models. 
        this class should be instantiated.
        """

    def __init__(self,random_state=None,learning_rate = 0.5,iterations = 1000,batchsize = None,num_loss_averages = 10,calc_loss_interval= 10,*kwargs):
        super(TFBaseClassifier, self).__init__(*kwargs) 

        self.classes_ = None
        self.n_classes = None
        self.x = None
        self.y_ = None
        self.feature_shape = None
        self.n_outputs = None
        self.warm_start = False
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batchsize = batchsize
        self.num_loss_averages = num_loss_averages
        self.calc_loss_interval = calc_loss_interval
        self.train_feed_dict = {}
        self.predict_feed_dict = {}
        
        

    def fit(self,X,y,warm_start = False):
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
        # place holder for the input and output
        self.x = tf.placeholder(tf.float32,[None]+self.feature_shape)
        self.y = tf.placeholder(tf.float32,[None,self.n_outputs])
       
        # create graph
        if not self.is_fitted and not self.warm_start:
            self._create_graph()

        # op for prediction_step
        self.predict_step = self._predict_step()

        # op for train step
        self.train_step = self._train_step()
        # initialize variables
        if not self.warm_start:
            self.session.run(tf.global_variables_initializer())
        # run the training
        self._train_loop(X,bin_y)
        self.is_fitted = True

        return self
    
    def predict_proba(self,X):
        if not self.is_fitted:
            print 'not fitted'
            return
        self.predict_feed_dict.update({self.x:X.astype(float)})
        return self.session.run(self.predict_step,feed_dict=self.predict_feed_dict)
    
    def predict(self,X):
        if not self.is_fitted:
            print 'not fitted'
            return
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba,axis=1)]
    
    def _train_loop(self,X,y):
        iteration = 0
        losses = [10000.] * self.num_loss_averages
        
        for batch in BatchIndGernerator(self.batchsize, X.shape[0], self.iterations):
            
            self.train_feed_dict.update({self.x:X[batch],self.y:y[batch]})
            self.session.run(self.train_step,feed_dict = self.train_feed_dict)
            iteration += 1
            
            if self.iterations is None and iteration%self.calc_loss_interval ==0:
                
                loss = self.session.run(self._loss_func(),feed_dict = self.train_feed_dict)
                losses = [loss] + losses[:-1]
                if np.diff(losses).max()<min(losses)/100.:
                    break
    def _loss_func(self):
        # override for more fancy stuff
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.predict_step))
    def _train_step(self):
        #override for more fancy stuff
        loss = self._loss_func() 
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
    
    def _create_graph(self):
        # this needs to be filled
        raise NotImplementedError

    def _predict_step(self):
        # this needs to be filled
        raise NotImplementedError
   