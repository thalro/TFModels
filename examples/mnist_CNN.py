import sys;sys.path.append('..')
from sklearn.datasets import fetch_mldata
from tfmodels.models import ConvolutionalNeuralNet as CNN
from sklearn.preprocessing import StandardScaler
import pylab

""" mnist_CNN.py 
    
    This example demonstrates the use of a
    ConvolutionalNeuralNet (CNN) to predict the MNIST
    hand written digits. 
    """

# download the data or load from file

mnist = fetch_mldata('MNIST original')

X, y = mnist.data , mnist.target

# rescale features to zero mean and unit variance
sc = StandardScaler()
X = sc.fit_transform(X)

X = X.reshape((X.shape[0],28,28,1))

# divide up into train and test sets
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


# create a CNN
CNN = CNN(n_filters = [64,128,64,64],filter_sizes = [[3,3],[3,3],[3,3],[3,3]],strides = [1,1,1,2],pooling = [1,1,2],pooling_strides = [1,1,2],n_hiddens = [1024],batchsize =512,iterations = 100,
          batch_normalization=True,dropout = 0.4,verbose = True,learning_rate = 0.1)

# fit it to the training data
CNN.fit(X_train,y_train)

# print out the test set accuracy
print 'Test accuracy:, ',CNN.score(X_test,y_test)