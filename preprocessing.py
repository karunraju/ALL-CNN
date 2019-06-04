import numpy as np

def sample_zero_mean(x):
    """
    Make each sample have a mean of zero by subtracting mean along the feature axis.
    :param x: float32(shape=(samples, features))
    :return: array same shape as x
    """
    mean_ = np.mean(x, axis=1)    
    mean_ = np.expand_dims(mean_, axis=1)

    return x - mean_

def gcn(x, scale=55., bias=0.01):
    """
    GCN each sample (assume sample mean=0)
    :param x: float32(shape=(samples, features))
    :param scale: factor to scale output
    :param bias: bias for sqrt
    :return: scale * x / sqrt(bias + sample variance)
    """
    var_ = np.var(x, axis=1)    
    var_ = np.expand_dims(var_, axis=1)
    var_ = np.sqrt(var_ + bias)
    x = (scale*x)/var_

    return x

def feature_zero_mean(x, xtest):
    """
    Make each feature have a mean of zero by subtracting mean along sample axis.
    Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features))
    :param xtest: float32(shape=(samples, features))
    :return: tuple (x, xtest)
    """
    mean_ = np.mean(x, axis=0)    
    x = x - mean_
    xtest = xtest - mean_

    return x, xtest

def zca(x, xtest, bias=0.1):
    """
    ZCA training data. Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features)) (assume mean=0)
    :param xtest: float32(shape=(samples, features))
    :param bias: bias to add to covariance matrix
    :return: tuple (x, xtest)
    """
    n = x.shape[0]
    xx = np.dot(x.T, x)/n
    xx = xx + np.eye(xx.shape[0])*bias
    ux, sx, vx = np.linalg.svd(xx)
    sx = np.diag(1.0/np.sqrt(sx))
    WX = np.dot(np.dot(ux, sx), ux.T)

    Y = np.dot(x, WX)
    Yt = np.dot(xtest, WX)
    return Y, Yt


def cifar_10_preprocess(x, xtest, image_size=32):
    """
    1) sample_zero_mean and gcn xtrain and xtest.
    2) feature_zero_mean xtrain and xtest.
    3) zca xtrain and xtest.
    4) reshape xtrain and xtest into NCHW
    :param x: float32 flat images (n, 3*image_size^2)
    :param xtest float32 flat images (n, 3*image_size^2)
    :param image_size: height and width of image
    :return: tuple (new x, new xtest), each shaped (n, 3, image_size, image_size)
    """
    x = sample_zero_mean(x)
    x = gcn(x)
    xtest = sample_zero_mean(xtest)
    xtest = gcn(xtest) 

    x, xtest = feature_zero_mean(x, xtest)
    x, xtest = zca(x, xtest)

    x = np.reshape(x, (x.shape[0], 3, image_size, image_size))
    xtest = np.reshape(xtest, (xtest.shape[0], 3, image_size, image_size))

    return x, xtest
    
