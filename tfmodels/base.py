
import os
import cPickle as pickle
import inspect


import numpy as np

from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

tfDtype = tf.float32
npDtype = np.float32


def recursive_argspec(cls):
    if cls is object: 
        return []
    try:
        argspec = inspect.getargspec(cls.__init__)
        args = argspec.args
    except:
        args = []
    if isinstance(cls,type):
        bases = cls.__bases__
    else:
        bases = cls.__class__.__bases__
    for base in bases:
        args += recursive_argspec(base)

    return [a for a in args if a!='self']


class BatchIndGernerator(object):
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


class BatchGernerator(object):
    def __init__(self,X,y=None,batchsize=128,iterations=10,shuffle = True,start_iteration = 0,preprocessors = [],preprocessor_args = [],is_training = True):
        self.ind_gernerator = BatchIndGernerator(batchsize=batchsize, N=X.shape[0], iterations=iterations,shuffle = shuffle,start_iteration = start_iteration)
        self.preprocessors = [p(**pa) for p,pa in zip(preprocessors,preprocessor_args)]
        self.X = X
        self.y = y
        self.is_training = is_training
    def __iter__(self):
        return self
    def next(self):
        inds,currrentiteration = self.ind_gernerator.next()

        Xbatch = self.X[inds]
        if self.y is not None:
            ybatch = self.y[inds]
        else:
            ybatch = None
        for prep in self.preprocessors:
            Xbatch = prep.transform(Xbatch,is_training = self.is_training)
        return Xbatch,ybatch,currrentiteration




class TFBaseEstimator(BaseEstimator):
    def __init__(self,n_jobs = 1):
        self.train_step = None
        self.predict_step = None
        self.is_fitted = False
        
        
        
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
        self.global_step_tensor = tf.Variable(0,name = 'global_step', trainable=False)

    @classmethod
    def _get_param_names(cls):
        """ this overrides the method of sklearn BaseEstimator
            so that param names are also collected from 
            super classes.
            """
        return sorted(recursive_argspec(cls))  



    def get_tf_vars(self):
        vars = tf.trainable_variables()
        var_names = [v.name for v in vars]
        return zip(self.session.run(vars),var_names)
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

    def __init__(self,random_state=None,learning_rate = 0.1,learning_rates=None,iterations = 10,batchsize = None,print_interval= 10,verbose = False,output_type ='softmax',epsilon = 1e-9,multilabel = False,multilabel_threshold = 0.2,batch_preprocessors = [], batch_preprocessor_args = [],*kwargs):
        super(TFBaseClassifier, self).__init__(*kwargs) 

        self.classes_ = None
        self.n_classes = None
        self.x = None
        self.y_ = None
        self.feature_shape = None
        self.n_outputs = None
        self.warm_start = False
        self.learning_rate_tensor = None
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
        self.epsilon = epsilon
        self.is_training = tf.placeholder(tf.bool)
        self.multilabel = multilabel
        self.multilabel_threshold = multilabel_threshold
        self.batch_preprocessors = batch_preprocessors
        self.batch_preprocessor_args = batch_preprocessor_args
        
        self.train_feed_dict = {}
        self.test_feed_dict = {}
        

    def fit(self,X,y,warm_start = False):
        
        original_batchsize = self.batchsize
        if self.batchsize is None:
            self.batchsize = X.shape[0]
        self.warm_start = warm_start
        if self.random_state is not None:
            
            np.random.seed(self.random_state)
            tf.set_random_seed(self.random_state)


        
        
        # targets need to be binarized
        lb = LabelBinarizer()
        bin_y = lb.fit_transform(y)
        if bin_y.shape[1]==1:
            bin_y = np.concatenate([1-bin_y,bin_y],axis = 1)
        self.classes_ = np.arange(bin_y.shape[1])

        self.n_classes = len(self.classes_)
        self.n_outputs = bin_y.shape[1]
        self.feature_shape = list(X.shape[1:])

        
       
        
        if not self.is_fitted:
            # place holder for the input and output
            self.x = tf.placeholder(tf.float32,[None]+self.feature_shape,name = 'self.x')
            self.y = tf.placeholder(tf.float32,[None,self.n_outputs],name = 'self.y')
            self.learning_rate_tensor = tf.placeholder(tf.float32,shape = [],name = 'learning_rate')
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
            self._init_vars()
        # run the training
        self._train_loop(X,bin_y)
        self.is_fitted = True
        self.batchsize = original_batchsize
        return self
    def _init_vars(self):
        self.session.run(tf.global_variables_initializer())
    def _opt_var_list(self):
        return tf.trainable_variables()
    def predict_proba(self,X):
        if not self.is_fitted:
            print 'not fitted'
            return
        
        output = []
        batches = BatchGernerator(X,None,self.batchsize,1,shuffle = False,preprocessors = self.batch_preprocessors,preprocessor_args = self.batch_preprocessor_args,is_training = False)
        for (Xbatch,ybatch,iteration) in batches:
            feed_dict = {self.x:Xbatch.astype(float),self.is_training:False}
            feed_dict.update(self.test_feed_dict)
            
            output.append(self.session.run(self.prediction,feed_dict=feed_dict))
        return np.concatenate(output,axis =0)
    
    def predict(self,X):
        if not self.is_fitted:
            print 'not fitted'
            return
        proba = self.predict_proba(X)
        
        if self.multilabel:
            return proba>self.multilabel_threshold
        else:
            return self.classes_[np.argmax(proba,axis=1)]
    
    def _train_loop(self,X,y):
        #ensure that iterations is list in case it has been changed
        if not isinstance(self.iterations, (list, tuple, np.ndarray)):
            self.iterations = [self.iterations]
        current_iteration = 0
        iteration  = 0
        for iterations,learning_rate in zip(self.iterations,self.learning_rates):
            
            self.learning_rate = learning_rate
            
            batches = BatchGernerator(X,y,self.batchsize, iterations+iteration,start_iteration = iteration,preprocessors = self.batch_preprocessors,preprocessor_args = self.batch_preprocessor_args,is_training = True)
            for i,(Xbatch,ybatch,iteration) in enumerate(batches):
                if iteration>current_iteration:
                    current_iteration+=1
                    self._iteration_callback()
                feed_dict = {self.x:Xbatch,self.y:ybatch,self.is_training:True,self.learning_rate_tensor:self.learning_rate}
                feed_dict.update(self.train_feed_dict)
                self.session.run(self.train_step,feed_dict = feed_dict)
                
                if self.verbose and  i%self.print_interval ==0:
                    feed_dict = {self.x:Xbatch,self.y:ybatch,self.is_training:False}
                    feed_dict.update(self.test_feed_dict)
                    loss = self.session.run(self._loss_func(),feed_dict = feed_dict)
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
            train_op =  tf.train.AdamOptimizer(learning_rate = self.learning_rate_tensor,epsilon = self.epsilon).minimize(loss,global_step = self.global_step_tensor,var_list = self._opt_var_list())
        return train_op

    def _iteration_callback(self):
        # this is executed at the beginning of each iteration 
        # and can be overridden with useful stuff
        return None
    def _create_graph(self):
        # this needs to be filled
        raise NotImplementedError

    def _predict_step(self):
        # this needs to be filled
        raise NotImplementedError
   
