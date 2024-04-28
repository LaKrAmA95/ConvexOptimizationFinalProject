### CODE FOR DATA GENERATION for SGD Experiments
import numpy as np
from sklearn.model_selection import train_test_split

#Generate Linear Data for SGD Experiments
#n_train: number of training samples
#n_test: number of test samples
#d: number of features
#noise_std: Standard Deviation of Gaussian Noise to add to Response Variable(i.e. Y)
#intercept: whether the linear model has intercept
def linear(n_train, n_test, d, noise_std, intercept = False):
    X = np.random.normal(size = (n_train + n_test, d)) #Generate X
    W = np.random.normal(size = (d, 1)) #Generate W(Weights of Linear Model)
    b = np.random.normal() if intercept else 0 #Generate bias of Linear Model
    Y = X @ W + b #Generate Y
    
    #Add Noise to Y
    noise = np.random.normal(loc = 0, scale = noise_std, size = Y.shape)
    Y = Y + noise
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42) #Split the Data into Training and Test
    
    return X_train, X_test, Y_train, Y_test, W, b

