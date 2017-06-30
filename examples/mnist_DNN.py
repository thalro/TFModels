import sys;sys.path.append('..')
from sklearn.datasets import fetch_mldata
from tfmodels.models import DenseNeuralNet as DNN

mnist = fetch_mldata('MNIST original')

X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


dnn = DNN(n_hiddens = [1000],batchsize =1000,iterations = 1000,verbose = True,batch_normalisation=True,learning_rate = 0.05)
dnn.fit(X_train,y_train)

print dnn.score(X_test,y_test)
