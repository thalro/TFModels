import unittest
import numpy as np
import os

class TestBatchIndGenerator(unittest.TestCase):

    def test_simple(self):

        N = 1000
        iterations = 10
        batchsize = 125

        from base import BatchIndGernerator
        big = BatchIndGernerator(batchsize, N, iterations)

        samples = []
        for batch,it  in big:
           samples.append(batch)

        self.assertEqual(len(samples), iterations*N/batchsize)


    def test_uneven(self):

        N = 1000
        iterations = 10
        batchsize = 130
         
        import math
        from base import BatchIndGernerator
        big = BatchIndGernerator(batchsize, N, iterations)

        samples = []
        for batch,it in big:
            samples.append(batch)

        self.assertEqual(len(samples), iterations*math.ceil(N/float(batchsize)))

    def test_batchsize_none(self):

        N = 1000
        iterations = 10
        batchsize = None
         
        import math
        from base import BatchIndGernerator
        big = BatchIndGernerator(batchsize, N, iterations)

        samples = []
        for batch,it in big:
            samples.append(batch)

        self.assertEqual(len(samples), iterations)


class TestLR(unittest.TestCase):
    """ this tests the basic functionality of classifiers
        using LogisticRegression as an Example."""

    def test_import_and_init(self):
        from models import LogisticRegression as LR

        lr = LR()

    def test_2D_separable(self):
        """ LR should give perfect performance on separable data. """
        from models import LogisticRegression as LR

        
        X = np.random.rand(100,2)
        y = np.random.randint(0,2,100)
        X[y==1] += 2

        lr = LR(iterations =100,learning_rate = 1.)
        lr.fit(X, y)
        
        self.assertEqual(lr.score(X,y), 1.)

    
    def test_loop_reinit(self):
        """ during cross validation, the same model
            will be initialized several times.
            """

        from models import LogisticRegression as LR
        from sklearn.model_selection import StratifiedKFold
        X = np.random.rand(100,2)
        y = np.random.randint(0,2,100)
        
        pred = np.zeros_like(y).astype(float)
        xval = StratifiedKFold(n_splits=3)
        for train,test in xval.split(X,y):
            lr = LR()
            lr.fit(X[train],y[train])
            pred = lr.predict_proba(X[test])[:,1]
        
    def test_random_state_consistency(self):
        from models import LogisticRegression as LR

        X = np.random.rand(100,2)
        y = np.random.randint(0,2,100)

        lr = LR(random_state = 1,iterations = 1)
        lr.fit(X, y)
        p1 = lr.predict_proba(X)
        
        lr = LR(random_state = 1,iterations = 1)
        lr.fit(X, y)
        p2 = lr.predict_proba(X)

        lr = LR(random_state = 2,iterations = 1)
        lr.fit(X, y)
        p3 = lr.predict_proba(X)

        lr = LR(random_state = None,iterations = 1)
        lr.fit(X, y)
        p4 = lr.predict_proba(X)

        self.assertTrue(np.allclose(p1,p2))
        self.assertFalse(np.allclose(p1,p3))
        self.assertFalse(np.allclose(p1,p4))
        np.random.seed(None)
         
    def test_warm_start(self):
        
        """ LR should give perfect performance on separable data. """
        from models import LogisticRegression as LR
        
        
        X = np.random.rand(100,2)
        y = np.random.randint(0,2,100)
        X[y==1] += 2
        
        lr = LR(iterations =100,learning_rate = 0.1)
        lr.fit(X, y)

        self.assertEqual(lr.score(X,y), 1.)    

        lr.iterations = 0

        lr.fit(X, y,warm_start=True)
        self.assertEqual(lr.score(X,y), 1.)   
        
        lr.fit(X, y,warm_start=False)
        self.assertTrue(lr.score(X,y)<1.)   

    def test_save_load(self):

        from models import LogisticRegression as LR

        X = np.random.rand(100,2)
        y = np.random.randint(0,2,100)

        

        lr = LR()
        lr.fit(X, y)
        y1 = lr.predict_proba(X)
        tmpfile = 'tempsave'

        lr.save(tmpfile)
        
        
        lr2 = LR()

        lr2.load(tmpfile)
        y2 = lr.predict_proba(X)

        self.assertTrue(np.allclose(y1,y2))

        

        session_files = [f for f in os.listdir('.') if tmpfile in f]
        for f in session_files:
            os.remove(f)


    def test_learing_rate_changes(self):
        from models import LogisticRegression as LR
        import tensorflow as tf

        # test for effect of learning rate on model instantiation

        X = np.random.rand(100,2)
        y = np.random.randint(0,2,100)

        lr = LR(random_state = 1,iterations = 0,learning_rate = 0.1)
        lr.fit(X, y)
        vars = tf.global_variables()
        
        weights_1_0 = lr.session.run([v for v in vars if 'dense_linear/kernel:0' in v.name])[0]
        lr.iterations = [1]
        lr.fit(X, y,warm_start = True)
        vars = tf.global_variables()
        
        weights_1_1 = lr.session.run([v for v in vars if 'dense_linear/kernel:0' in v.name])[0]

        

        lr = LR(random_state = 1,iterations = 0,learning_rate = 0.01)
        lr.fit(X, y)
        vars = tf.global_variables()
        
        weights_2_0 = lr.session.run([v for v in vars if 'dense_linear/kernel:0' in v.name])[0]
        lr.iterations = [1]
        lr.fit(X, y,warm_start = True)
        vars = tf.global_variables()
        
        weights_2_1 = lr.session.run([v for v in vars if 'dense_linear/kernel:0' in v.name])[0]

       

        self.assertTrue(np.sum((weights_1_1-weights_1_0)**2)>np.sum((weights_2_1-weights_2_0)**2))


        # test for effect of learning rate if it is changed on existing model.


        lr = LR(random_state = 1,iterations = 0,learning_rate = 0.1)
        lr.fit(X, y)
        vars = tf.global_variables()
        
        weights_3_0 = lr.session.run([v for v in vars if 'dense_linear/kernel:0' in v.name])[0]
        lr.iterations = [1]
        lr.fit(X, y,warm_start = True)
        vars = tf.global_variables()
        
        weights_3_1 = lr.session.run([v for v in vars if 'dense_linear/kernel:0' in v.name])[0]
        lr.iterations = [1]
        lr.fit(X, y,warm_start = True)
        vars = tf.global_variables()
        
        weights_3_2 = lr.session.run([v for v in vars if 'dense_linear/kernel:0' in v.name])[0]

       


        lr = LR(random_state = 1,iterations = 0,learning_rate = 0.1)
        lr.fit(X, y)
        vars = tf.global_variables()
        
        weights_4_0 = lr.session.run([v for v in vars if 'dense_linear/kernel:0' in v.name])[0]
        lr.iterations = [1]
        lr.fit(X, y,warm_start = True)
        vars = tf.global_variables()
        
        weights_4_1 = lr.session.run([v for v in vars if 'dense_linear/kernel:0' in v.name])[0]
        lr.iterations = [1]
        lr.learning_rates =[0.001]
        lr.fit(X, y,warm_start = True)
        vars = tf.global_variables()
        
        weights_4_2 = lr.session.run([v for v in vars if 'dense_linear/kernel:0' in v.name])[0]

        

        self.assertTrue(np.sum((weights_3_2-weights_3_1)**2)>np.sum((weights_4_2-weights_4_1)**2))
        
         

class TestTextConvNet(unittest.TestCase):
    def test_import_and_init(self):
        from models import TextConvNet as TCN

        tcn = TCN()

    def test_loop_reinit(self):
        """ during cross validation, the same model
            will be initialized several times.
            """

        from models import TextConvNet as TCN

        
        from sklearn.model_selection import StratifiedKFold
        X = np.random.rand(20,10,5)
        y = np.random.randint(0,2,20)
        
        pred = np.zeros_like(y).astype(float)
        xval = StratifiedKFold(n_splits=2)
        for train,test in xval.split(X,y):
            tcn = TCN(filter_sizes = [2,3],n_filters = 3,n_hidden= 10,iterations  =5)
            tcn.fit(X[train],y[train])
            pred = tcn.predict_proba(X[test])[:,1]

    def test_save_load(self):

        from models import TextConvNet as TCN

        X = np.random.rand(100,5,2)
        y = np.random.randint(0,2,100)

        

        tcn = TCN()
        tcn.fit(X, y)
        y1 = tcn.predict_proba(X)
        tmpfile = 'tempsave'

        tcn.save(tmpfile)
        
        
        tcn2 = TCN()

        tcn2.load(tmpfile)
        y2 = tcn.predict_proba(X)

        self.assertTrue(np.allclose(y1,y2))

        session_files = [f for f in os.listdir('.') if tmpfile in f]
        for f in session_files:
            os.remove(f)

    def test_random_state_consistency(self):
        from models import TextConvNet as TCN

        X = np.random.rand(100,5,2)
        y = np.random.randint(0,2,100)

        tcn = TCN(random_state = 1,iterations = 2,n_filters = 2,filter_sizes = [2,3],n_hidden = 10)
        tcn.fit(X, y)
        p1 = tcn.predict_proba(X)
        
        tcn = TCN(random_state = 1,iterations = 2,n_filters = 2,filter_sizes = [2,3],n_hidden = 10)
        tcn.fit(X, y)
        p2 = tcn.predict_proba(X)

        

        tcn = TCN(random_state = 2,iterations = 2,n_filters = 2,filter_sizes = [2,3],n_hidden = 10)
        tcn.fit(X, y)
        p3 = tcn.predict_proba(X)

        tcn = TCN(random_state = None,iterations = 2,n_filters = 2,filter_sizes = [2,3],n_hidden = 10)
        tcn.fit(X, y)
        p4 = tcn.predict_proba(X)

        self.assertTrue(np.allclose(p1,p2))
        self.assertFalse(np.allclose(p1,p3))
        self.assertFalse(np.allclose(p1,p4))
        np.random.seed(None)
         
    def test_warm_start(self):
        
        """ TCN should give perfect performance on separable data. """
        from models import TextConvNet as TCN
        
        
        X = np.random.rand(51,5,2)
        y = np.random.randint(0,2,51)
        X[y==1] += 1.
        
        tcn = TCN(iterations =0,learning_rate = 0.5)
        tcn.fit(X, y)

        p1 = tcn.predict_proba(X)

        
        
        tcn.fit(X, y,warm_start=True)
        p2 = tcn.predict_proba(X)
        self.assertTrue(np.allclose(p1,p2)) 
        
        tcn.fit(X, y,warm_start=False)
        p3 = tcn.predict_proba(X)
        self.assertFalse(np.allclose(p1,p3)) 



class TestDNN(unittest.TestCase):
    """ this tests the basic functionality of classifiers
        using LogisticRegression as an Example."""

    def test_import_and_init(self):
        from models import DenseNeuralNet as DNN

        dnn = DNN()

    def test_2D_separable(self):
        """ DNN should give perfect performance on separable data. """
        from models import DenseNeuralNet as DNN

        
        X = np.random.rand(100,2)
        y = np.random.randint(0,2,100)
        X[y==1] += 2

        dnn = DNN(n_hiddens = [10,10],batch_normalization=True,dropout = 0.1,iterations =500,learning_rate = 0.001)
        dnn.fit(X, y)
        self.assertEqual(dnn.score(X,y), 1.)

    
    def test_loop_reinit(self):
        """ during cross validation, the same model
            will be initialized several times.
            """

        from models import DenseNeuralNet as DNN
        from sklearn.model_selection import StratifiedKFold
        X = np.random.rand(100,2)
        y = np.random.randint(0,2,100)
        
        pred = np.zeros_like(y).astype(float)
        xval = StratifiedKFold(n_splits=3)
        for train,test in xval.split(X,y):
            dnn = DNN()
            dnn.fit(X[train],y[train])
            pred = dnn.predict_proba(X[test])[:,1]
        
    def test_random_state_consistency(self):
        from models import DenseNeuralNet as DNN

        X = np.random.rand(100,2)
        y = np.random.randint(0,2,100)

        dnn = DNN(random_state = 1,iterations = 1)
        dnn.fit(X, y)
        p1 = dnn.predict_proba(X)
        
        dnn = DNN(random_state = 1,iterations = 1)
        dnn.fit(X, y)
        p2 = dnn.predict_proba(X)

        dnn = DNN(random_state = 2,iterations = 1)
        dnn.fit(X, y)
        p3 = dnn.predict_proba(X)

        dnn = DNN(random_state = None,iterations = 1)
        dnn.fit(X, y)
        p4 = dnn.predict_proba(X)

        self.assertTrue(np.allclose(p1,p2))
        self.assertFalse(np.allclose(p1,p3))
        self.assertFalse(np.allclose(p1,p4))
        np.random.seed(None)
         
    def test_warm_start(self):
        
        """ DNN should give perfect performance on separable data. """
        from models import DenseNeuralNet as DNN
        
        
        X = np.random.rand(100,2)
        y = np.random.randint(0,2,100)
        X[y==1] += 2
        
        dnn = DNN(iterations =500,learning_rate = 0.2,n_hiddens = [])
        dnn.fit(X, y)

        self.assertEqual(dnn.score(X,y), 1.)    

        dnn.iterations = 0

        dnn.fit(X, y,warm_start=True)
        self.assertEqual(dnn.score(X,y), 1.)   
        
        dnn.fit(X, y,warm_start=False)
        self.assertTrue(dnn.score(X,y)<1.)   

    def test_save_load(self):

        from models import DenseNeuralNet as DNN

        X = np.random.rand(10,2)
        y = np.random.randint(0,2,10)

        

        dnn = DNN(dropout = 0.5)
        dnn.fit(X, y)
        y1 = dnn.predict_proba(X)
        tmpfile = 'tempsave'

        dnn.save(tmpfile)
        
        
        dnn2 = DNN()

        dnn2.load(tmpfile)
        y2 = dnn.predict_proba(X)
       
        
        self.assertTrue(np.allclose(y1,y2))

        session_files = [f for f in os.listdir('.') if tmpfile in f]
        for f in session_files:
            os.remove(f)



class TestCNN(unittest.TestCase):
    """ this tests the basic functionality of classifiers
        using LogisticRegression as an Example."""

    def test_import_and_init(self):
        from models import ConvolutionalNeuralNet as CNN

        cnn = CNN()

    def test_2D_separable(self):
        """ CNN should give perfect performance on separable data. """
        from models import ConvolutionalNeuralNet as CNN

        
        X = np.random.rand(100,10,8,2)
        y = np.random.randint(0,2,100)
        X[y==1] += 2

        cnn = CNN(n_hiddens = [20],batch_normalization=True,dropout = 0.1,iterations =[10,10,20],learning_rates = [0.1,0.01,0.001],verbose = True)

        cnn.fit(X, y)
        
        self.assertEqual(cnn.score(X,y), 1.)

    
    def test_loop_reinit(self):
        """ during cross validation, the same model
            will be initialized several times.
            """

        from models import ConvolutionalNeuralNet as CNN
        from sklearn.model_selection import StratifiedKFold
        X = np.random.rand(100,10,8,2)
        y = np.random.randint(0,2,100)
        
        pred = np.zeros_like(y).astype(float)
        xval = StratifiedKFold(n_splits=3)
        for train,test in xval.split(X,y):
            cnn = CNN()
            cnn.fit(X[train],y[train])
            pred = cnn.predict_proba(X[test])[:,1]
        
    def test_random_state_consistency(self):
        from models import ConvolutionalNeuralNet as CNN

        X = np.random.rand(100,10,8,2)
        y = np.random.randint(0,2,100)

        cnn = CNN(random_state = 1,iterations = 1)
        cnn.fit(X, y)
        p1 = cnn.predict_proba(X)
        
        cnn = CNN(random_state = 1,iterations = 1)
        cnn.fit(X, y)
        p2 = cnn.predict_proba(X)

        cnn = CNN(random_state = 2,iterations = 1)
        cnn.fit(X, y)
        p3 = cnn.predict_proba(X)

        cnn = CNN(random_state = None,iterations = 1)
        cnn.fit(X, y)
        p4 = cnn.predict_proba(X)

        self.assertTrue(np.allclose(p1,p2))
        self.assertFalse(np.allclose(p1,p3))
        self.assertFalse(np.allclose(p1,p4))
        np.random.seed(None)
         
    def test_warm_start(self):
        
        """ CNN should give perfect performance on separable data. """
        from models import ConvolutionalNeuralNet as CNN
        
        
        X = np.random.rand(100,10,8,2)
        y = np.random.randint(0,2,100)
        X[y==1] += 2
        
        cnn = CNN(iterations =500,learning_rate = 0.2,n_hiddens = [])
        cnn.fit(X, y)

        self.assertEqual(cnn.score(X,y), 1.)    

        cnn.iterations = 0

        cnn.fit(X, y,warm_start=True)
        self.assertEqual(cnn.score(X,y), 1.)   
        
        cnn.fit(X, y,warm_start=False)
        self.assertTrue(cnn.score(X,y)<1.)   

    def test_save_load(self):

        from models import ConvolutionalNeuralNet as CNN

        X = np.random.rand(10,10,8,2)
        y = np.random.randint(0,2,10)

        

        cnn = CNN(dropout = 0.5)
        cnn.fit(X, y)
        y1 = cnn.predict_proba(X)
        tmpfile = 'tempsave'

        cnn.save(tmpfile)
        
        
        cnn2 = CNN()

        cnn2.load(tmpfile)
        y2 = cnn.predict_proba(X)
       
        
        self.assertTrue(np.allclose(y1,y2))

        session_files = [f for f in os.listdir('.') if tmpfile in f]
        for f in session_files:
            os.remove(f)



class TestResnet(unittest.TestCase):
    """ this tests the basic functionality of classifiers
        using LogisticRegression as an Example."""

    def test_import_and_init(self):
        from models import Resnet as RSN

        rsn = RSN(N=10,N_fixed =7,batchsize =8)


    
    def test_loop_reinit(self):
        """ during cross validation, the same model
            will be initialized several times.
            """

        from models import Resnet as RSN
        from sklearn.model_selection import StratifiedKFold
        X = np.random.rand(10,224,224,3)
        y = np.random.randint(0,2,10)
        
        pred = np.zeros_like(y).astype(float)
        xval = StratifiedKFold(n_splits=3)
        for train,test in xval.split(X,y):
            rsn = RSN(N=10,N_fixed =7,batchsize =8)
            rsn.fit(X[train],y[train])
            pred = rsn.predict_proba(X[test])[:,1]
        
    def test_random_state_consistency(self):
        from models import Resnet as RSN

        X = np.random.rand(4,224,224,3)
        y = np.random.randint(0,2,4)

        rsn = RSN(N=10,N_fixed =7,random_state = 1,iterations = 1,batchsize =8)
        rsn.fit(X, y)
        p1 = rsn.predict_proba(X)
        
        rsn = RSN(N=10,N_fixed =7,random_state = 1,iterations = 1,batchsize =8)
        rsn.fit(X, y)
        p2 = rsn.predict_proba(X)

        rsn = RSN(N=10,N_fixed =7,random_state = 2,iterations = 1,batchsize =8)
        rsn.fit(X, y)
        p3 = rsn.predict_proba(X)

        rsn = RSN(N=10,N_fixed =7,random_state = None,iterations = 1,batchsize =8)
        rsn.fit(X, y)
        p4 = rsn.predict_proba(X)

        self.assertTrue(np.allclose(p1,p2))
        #self.assertFalse(np.allclose(p1,p3))
        self.assertFalse(np.allclose(p1,p4))
        np.random.seed(None)
         
    

    def test_save_load(self):

        from models import Resnet as RSN

        X = np.random.rand(4,224,224,3)
        y = np.random.randint(0,2,4)
        

        rsn = RSN(N=10,N_fixed =7,dropout = 0.5,batchsize =8)
        rsn.fit(X, y)
        y1 = rsn.predict_proba(X)
        tmpfile = 'tempsave'

        rsn.save(tmpfile)
        
        
        rsn2 = RSN(N=10,N_fixed =7,dropout = 0.5,batchsize =8)

        rsn2.load(tmpfile)
        y2 = rsn.predict_proba(X)
       
        
        self.assertTrue(np.allclose(y1,y2))

        session_files = [f for f in os.listdir('.') if tmpfile in f]
        for f in session_files:
            os.remove(f)

    def test_fine_tuning(self):
        from models import Resnet as RSN

        X = np.random.rand(4,224,224,3)
        y = np.random.randint(0,2,4)

        rsn = RSN(N=10,N_fixed =7,fixed_epochs=2,random_state = 1,iterations = 1,learning_rate = 0.1,batchsize =4)
        rsn.fit(X, y)
        
        vars0 =  rsn.get_tf_vars()
        
        rsn.iterations = 1
        rsn.fit(X, y)

        vars1 =  rsn.get_tf_vars()

        for v0,v1 in zip(vars0,vars1):
            if 'base_model' in v0[1]:
                self.assertTrue((v0[0]==v1[0]).all())
            elif 'dense_top' in v0[1]:
                self.assertFalse((v0[0]==v1[0]).all())


        rsn.iterations = 1
        rsn.fit(X, y)
        self.assertFalse(rsn.bottom_fixed)
        vars2 =  rsn.get_tf_vars()
        base_changed = False
        for v0,v2 in zip(vars0,vars2):
            if 'base_model' in v0[1]:
                if not (v0[0]==v2[0]).all():
                    base_changed = True
            elif 'dense_top' in v0[1]:
                self.assertFalse((v0[0]==v2[0]).all())

        self.assertTrue(base_changed)


        rsn = RSN(N=10,N_fixed =10,fixed_epochs=2,random_state = 1,iterations = 2,batchsize =4,learning_rate = 0.1,verbose = True,print_interval  =1)
        rsn.fit(X, y)
        vars1 =  rsn.get_tf_vars()
        rsn = RSN(N=10,N_fixed =None,fixed_epochs=1,random_state = 1,iterations = 2,batchsize =4,learning_rate = 0.1,verbose = True,print_interval  =1)
        rsn.fit(X, y)

        vars2 =  rsn.get_tf_vars()
        base_changed = False
        for v1,v2 in zip(vars1,vars2):
            
            if 'base_model' in v1[1]:
                if not (v1[0]==v2[0]).all():
                    base_changed = True
            elif 'dense_top' in v1[1]:
                pass#self.assertFalse((v1[0]==v2[0]).all())

        self.assertTrue(base_changed)






if __name__ == '__main__':
    unittest.main()
