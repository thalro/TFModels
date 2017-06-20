
import os
import cPickle as pickle
import numpy as np

from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf

class TFBaseEstimator(BaseEstimator):
    def __init__(self):
        self.train_step = None
        self.predict_step = None
        self.is_fitted = False
        self._var_scope = None
        self.session = tf.Session()
        self.tf_vars = {}

    # def _get_var_scope(self):
    #     """ Create a unique string for the variable scope 
    #         This is to enable several instantiations of the 
    #         same class without variable conflicts"""
    #     if self._var_scope is None:
    #         self._var_scope = type(self).__name__+ '_'+''.join(choice(ascii_lowercase+digits) for i in range(10))
    #     return self._var_scope
    def save(self,fname):
        
        tf_vars = {}
        for var in self.tf_vars.keys():
            tf_vars[var] = self.session.run(self.tf_vars[var])

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
        this class cannot be instantiated.
        """

    def __init__(self,random_state=None,learning_rate = 0.5):
        super(TFBaseClassifier, self).__init__() 
        self.classes_ = None
        self.x = None
        self.y_ = None
        self.feature_shape = None
        self.n_outputs = None
        self.loss =None
        self.warm_start = False
        self.random_state = random_state
        self.learning_rate = learning_rate
        
        

    def fit(self,X,y,warm_start = False):
        self.warm_start = warm_start
        if self.random_state is not None:
            
            np.random.seed(self.random_state)
            tf.set_random_seed(self.random_state)


        self.classes_ = np.unique(y)
        # targets need to be binarized
        lb = LabelBinarizer()
        bin_y = lb.fit_transform(y)
        if bin_y.shape[1]==1:
            bin_y = pylab.concatenate([1-bin_y,bin_y],axis = 1)
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
        
        return self.session.run(self.predict_step,feed_dict={self.x:X.astype(float)})
    
    def predict(self,X):
        if not self.is_fitted:
            print 'not fitted'
            return
        proba = self.predict_proba(X)
        return self.classes_[pylab.argmax(proba,axis=1)]
    
    def _train_loop(self,X,y):
        iteration = 0
        losses = [10000.] * 5#self.num_loss_averages
        while True:
            if self.iterations is not None and iteration>=self.iterations:
                break
            self.session.run(self.train_step,feed_dict = {self.x:X,self.y:y})
            iteration += 1
            
            if self.iterations is None and iteration%10 ==0:
                
                loss = self.session.run(self.loss,feed_dict = {self.x:X,self.y:y})
                losses = [loss] + losses[:-1]
                if np.diff(losses).max()<min(losses)/100.:
                    break
    def _create_graph(self):
        # this needs to be filled
        raise NotImplementedError

    def _predict_step(self):
        # this needs to be filled
        raise NotImplementedError
    def _train_step(self):
        # this needs to be filled
        raise NotImplementedError