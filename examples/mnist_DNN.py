import sys;sys.path.append('..')
from sklearn.datasets import fetch_mldata
from tfmodels.models import DenseNeuralNet as DNN
from sklearn.preprocessing import StandardScaler


""" mnist_DNN.py 
    
    This example demonstrates the use of a
    DenseNeuralNet (DNN) to predict the MNIST
    hand written digits. 
    """

# download the data or load from file

mnist = fetch_mldata('MNIST original')

X, y = mnist.data , mnist.target

# rescale features to zero mean and unit variance
sc = StandardScaler()
X = sc.fit_transform(X)

# divide up into train and test sets
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# create a DNN
dnn = DNN(n_hiddens = [500,500],batchsize =512,iterations = 10,
          batch_normalisation=True,dropout = 0.1,verbose = True)
# fit it to the training data
dnn.fit(X_train,y_train)

# print out the test set accuracy
print 'Test accuracy:, ',dnn.score(X_test,y_test)
