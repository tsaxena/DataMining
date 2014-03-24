

#### Libraries
# Standard library
import cPickle

# Third-party libraries
import numpy as np
import csv


def load_data():
    #read the test and train csv files in a numpy array
    kaggle_train_data = np.loadtxt(open("../data/train.csv","rb"),delimiter=",",skiprows=1)  
    kaggle_test_data = np.loadtxt(open("../data/test.csv","rb"),delimiter=",",skiprows=1)  

    return (kaggle_train_data, kaggle_test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, test_data)``.

    The ``training_data`` is a list containing 2-tuples ``(x, y)``.  
    ``x`` is a 784-dimensional numpy.ndarray containing the input image. 
    ``y`` is a 10-dimensional numpy.ndarray representing the unit vector 
          corresponding to the correct digit for ``x``"""
   
    k_tr_d, k_te_d = load_data()
  
    #Kaggle data 
    train_tuples           = [ ( map(lambda y: y/255, x[1:]) , vectorized_result(x[0]) ) for x in k_tr_d]
    kaggle_training_data   = [  (np.reshape(x, (784,1)) ,  y) for x,y in train_tuples]
    
     
    #Kaggle test data
    test_vectors = [map(lambda y : y/255, x) for x in k_te_d] 
    kaggle_test_data = [np.reshape(x,(784,1) ) for x in test_vectors]
    

    test_inputs = [np.reshape(x, (784, 1)) for x in k_tr_d[0]]
    test_data = zip(test_inputs, k_tr_d[1])
    return (kaggle_training_data, test_data, kaggle_test_data)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digitm
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


