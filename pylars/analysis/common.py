import numpy as np

def Gaussean(x, A, mu, sigma):
    y = A * np.exp(-((x - mu) / sigma)**2 / 2) / np.sqrt(2 * np.pi * sigma**2)
    return y

def func_linear(x, a,b):
    return a*x+b