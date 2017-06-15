import unittest
import numpy as np


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

        lr = LR(iterations =None)
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

         
    def test_warm_start(self):
        
        """ LR should give perfect performance on separable data. """
        from models import LogisticRegression as LR

        
        X = np.random.rand(100,2)
        y = np.random.randint(0,2,100)
        X[y==1] += 2

        lr = LR(iterations =None)
        lr.fit(X, y)
        self.assertEqual(lr.score(X,y), 1.)    

        lr.iterations = 0

        lr.fit(X, y,warm_start=True)
        self.assertEqual(lr.score(X,y), 1.)   
        
        lr.fit(X, y,warm_start=False)
        self.assertTrue(lr.score(X,y)<1.)   







if __name__ == '__main__':
    unittest.main()