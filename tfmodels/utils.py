import os
import numpy as np
import pylab

import cPickle as pickle

from sklearn.base import BaseEstimator
from skimage.transform import rotate,resize
from random import choice


def _random_rotation(image):
    angle = choice([0.,90.,180.,270.])
    angle += choice([0,-pylab.rand()*10.,pylab.rand()*10.])
    return rotate(image,angle,resize = False)
def _random_rescaling(image):
    # rescale between +/- 25%
    scale = 0.75 + 0.5 * pylab.rand()
    originalsize = image.shape
    try:
        newsize = [int(s*scale) for s in originalsize[:2]]+[originalsize[2]]
    except:
        newsize = [int(s*scale) for s in originalsize[:2]]
    scaled = resize(image, newsize)
    if scale <=1:
        # apply zero padding
        image = pylab.zeros(originalsize)
        offset = (originalsize[0]-newsize[0])/2
        
        image[offset:offset+newsize[0],offset:offset+newsize[1]] = scaled
    else:
        #crop
        xoffset = pylab.randint(0,(newsize[0]-originalsize[0])/2+1)
        yoffset = pylab.randint(0,(newsize[1]-originalsize[1])/2+1)
        image = scaled[xoffset:xoffset+originalsize[0],yoffset:yoffset+originalsize[1]]
    return image

class ImageAugmenter(object):
    def __init__(self,transform_prob = 0.2):
        self.transform_prob =  transform_prob
    def fit(self,X,y=None):
        pass
    def fit_transform(self,X,y=None):
        return self.transform(X)
    def transform(self,X,is_training = False):
        
        X_out = X.copy()
        if is_training:
            for i in range(X.shape[0]):
                transform_list = [pylab.fliplr,pylab.flipud,_random_rotation,_random_rescaling]
                for transform in transform_list:
                    
                    if pylab.rand()<self.transform_prob:

                       X_out[i] = transform(X[i]) 

        
        return X_out

class ImageNetScaler(object):
    """ this seems to be required for models pretrained
        on imagenet. """
   
    def fit(self,X,y=None):
        pass
    def fit_transform(self,X,y=None):
        return self.transform(X)
    def transform(self,X,is_training = False):
        X_out = X.copy()
        if not X_out.max()>1.:
            X_out *= 255.
        X_out = X_out[:, :, :, ::-1]
        # Zero-center by mean pixel
        X_out[:, :, :, 0] -= 103.939
        X_out[:, :, :, 1] -= 116.779
        X_out[:, :, :, 2] -= 123.68
        return X_out
      
def accuracy(y_true,y_pred):
    return (y_true==y_pred).mean()

class NetworkTrainer(BaseEstimator):
    def __init__(self,model_type=None,model_args={},stagnation_slope  = 0.05,max_epochs = 100,
                      valid_fraction = 0.1,live_plot = True,score_func = accuracy,
                      training_dir = None,save_interval = None,control_file = 'run_control'):
        self.model_type = model_type
        self.model_params = model_args
        try:
            self.model = model_type(**model_args)
        except:
            self.model  = None

        self.stagnation_slope = stagnation_slope
        self.max_epochs = max_epochs
        self.valid_fraction = valid_fraction
        self.live_plot = live_plot
        self.score_func = score_func
        self.save_interval = save_interval
        self.control_file = control_file

        self.train_loss = []
        self.valid_loss = []

        self.valid_inds = None
        self.train_inds = None
        self.current_iteration = 0
        
        self.figure = None
        self.valid_plot = None
        self.train_plot = None
        self.model_params = {self.current_iteration:self.model.get_params()}
        self.training_dir = training_dir
        if training_dir is not None:
            if not os.path.exists(training_dir):
                os.mkdir(training_dir)
        self.last_saved = 0
        
    def init_plot(self):
        pylab.ion()
        if self.figure is None or not pylab.fignum_exists(self.figure.number):
            self.figure = pylab.figure()
            pylab.xlabel('epoch')
            pylab.ylabel('score')
            if self.train_plot is None:
                try:
                    tr = self.train_plot = pylab.plot(self.train_loss,'k',label = 'train')[0]
                    te = self.valid_plot = pylab.plot(self.valid_loss,'r',label = 'test')[0]
                    pylab.figlegend((tr,te),('train','test'),loc = 'upper center',frameon = False,ncol =2)
                
                    pylab.xlim(0,self.current_iteration+1)
                    ylim = list(pylab.ylim())
                    maxval = max(max(self.train_loss),max(self.valid_loss))
                    minval = min(min(self.train_loss),min(self.valid_loss))
                    if maxval>=ylim[1]:
                        ylim[1] = maxval + 0.1 * (ylim[1]-ylim[0])
                    if minval<=ylim[0]:
                        ylim[0] = minval - 0.1 * (ylim[1]-ylim[0])

                    pylab.ylim(minval - 0.1*(maxval-minval),maxval + 0.1*(maxval-minval))
                except:
                    pass
            
            pylab.xlim(0,self.current_iteration+1)

    def get_params(self,deep = False):
        return super(NetworkTrainer, self).get_params(deep = False)

    def _update_plot(self):
        
        if not self.live_plot:
            return 
        try:
            

            if self.train_plot is None:
                tr = self.train_plot = pylab.plot(self.train_loss,'k',label = 'train')[0]
                te = self.valid_plot = pylab.plot(self.valid_loss,'r',label = 'test')[0]
                pylab.figlegend((tr,te),('train','test'),loc = 'upper center',frameon = False,ncol =2)
            else:
                self.train_plot.set_data(range(1,len(self.train_loss)+1),self.train_loss)
                self.valid_plot.set_data(range(1,len(self.valid_loss)+1),self.valid_loss)
            pylab.xlim(0,self.current_iteration+1)
            ylim = list(pylab.ylim())
            maxval = max(max(self.train_loss),max(self.valid_loss))
            minval = min(min(self.train_loss),min(self.valid_loss))
            if maxval>=ylim[1]:
                ylim[1] = maxval + 0.1 * (ylim[1]-ylim[0])
            if minval<=ylim[0]:
                ylim[0] = minval - 0.1 * (ylim[1]-ylim[0])

            pylab.ylim(ylim)
               
            pylab.draw()
            pylab.pause(0.0000001)
        except:
            pass

    def add_epochs(self,n):
        self.max_epochs += n

    def save(self,fname = 'trainer'):
        fname += '_'+str(self.current_iteration)
        self.model.save(os.path.join(self.training_dir,fname))
        things_to_save = (self.get_params(deep = False),self.train_loss,self.valid_loss,self.current_iteration,self.model_params,self.last_saved,self.train_inds,self.valid_inds)
        pickle.dump(things_to_save, open(os.path.join(self.training_dir,fname),'w'),protocol  = 2)

    def load(self,fname = None):
        if fname is None:
            highest_iteration = 0
            files = os.listdir(self.training_dir)
            for f in files:
                try:
                    iteration = int(f.split('_')[1])
                    highest_iteration = max(iteration,highest_iteration)
                except:
                    pass
            fname = os.path.join(self.training_dir,'trainer_'+str(highest_iteration))
        loaded_stuff = pickle.load(open(fname))
        self.set_params(**loaded_stuff[0]) 
        self.train_loss,self.valid_loss,self.current_iteration,self.model_params,self.last_saved,self.train_inds,self.valid_inds = loaded_stuff[1:]
        
        self.model = self.model_type()
        self.model.load(fname)

    def fit(self,X,y):
        if self.live_plot:
            self.init_plot()
        if self.train_inds is None or len(self.valid_inds)+len(self.train_inds)!=X.shape[0]:
            inds = np.arange(X.shape[0])
            np.random.shuffle(inds)
            n_valid = int(self.valid_fraction*len(inds))
            self.valid_inds = inds[:n_valid]
            self.train_inds = inds[n_valid:] 
            X_train = X[self.train_inds]
            y_train = y[self.train_inds]
            X_valid = X[self.valid_inds]
            y_valid = y[self.valid_inds]
        while self.current_iteration < self.max_epochs:
            self.model.iterations = 1
            print self.current_iteration
            print 'fitting model'
            self.model.fit(X_train,y_train,warm_start = self.model.is_fitted) 
            self.model_params[self.current_iteration] = self.model.get_params()
            self.current_iteration += 1
            print 'computing training score'
            self.train_loss.append(self.score_func(y_train,self.model.predict(X_train)))
            print 'computing validation score'
            self.valid_loss.append(self.score_func(y_valid,self.model.predict(X_valid)))
            
            self._update_plot()

            if self.save_interval is not None:
                if (self.current_iteration-self.last_saved) >= self.save_interval:
                    print 'saving model'
                    self.save()
                    self.last_saved = self.current_iteration

            if self.control_file is not None:
                try:
                    controls = open(os.path.join(self.training_dir,self.control_file)).read()
                    if 'stop now' in controls:
                        controls = controls.replace('stop now','')
                        cfile = open(os.path.join(self.training_dir,self.control_file),'w')
                        cfile.write(controls)
                        cfile.close()
                        break
                    controls = controls.split('\n')
                    model_params = self.model.get_params()
                    for control in controls:
                        if 'model_param' in control:
                            param,value = control.split(':')[1:3]
                            value = eval(value)
                            if model_params[param]!= value:
                                self.model.set_params(**{param:value})
                                print 'set ',param,' to ',value
                except:
                    # file doesn't exist
                    pass


        
class BaggingClassifier(object):
    pass