import numpy as np

def L2(yhat:np.array,y:np.array):
    # print(sum((yhat-y)**2))
    return 0.5*sum((yhat-y)**2)

def L2grad(yhat:np.array,y:np.array):
    return sum((yhat-y))


if __name__=="__main__":
    L2(np.array([1,2,3]),np.array([1,2,4]))
    