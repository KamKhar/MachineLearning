import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import preprocessing
digits = load_digits()

class LogReg:

    def __init__(self,X,tar,k,tr,val):
        #matrix N*d
        self.X_char = self.standartisation(X)

        self.t = tar
        self.k = k

        #matrix N*k
        self.tar_onehoten= self.tar2onehotenc(tar,k)

        self.x_train, self.t_train, self.x_valid, self.t_valid,self. x_test, self.t_test = self.data_set(X,tar,tr,val)

    def standartisation(self,X):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        for i in range(len(X[0, :])):
            X[:, i] -= mean[i]
        for i in range(len(X[0, :])):
            if (std[i] == 0):
                X[:, i] = 0
            else:
                X[:, i] /= std[i]
        return X

    def data_set(self,x,t,tr,val):
        N = len(t)
        ind = np.arange(N)
        ind_prm = np.random.permutation(ind)
        train_ind = ind_prm[:np.int32(tr*N)]
        valid_ind = ind_prm[np.int32(tr*N):np.int32((val+tr)*N)]
        test_ind = ind_prm[np.int32((val+tr)*N):]
        x_train = x[train_ind]
        t_train = t[train_ind]
        x_valid = x[valid_ind]
        t_valid = t[valid_ind]
        x_test = x[test_ind]
        t_test = t[test_ind]
        return x_train,t_train,x_valid,t_valid,x_test,t_test

    def tar2onehotenc(self,t,k):
        tar = np.zeros((len(t),k))
        for i in range(len(t)):
            tar[i,t[i]] = 1
        return tar

    def inicial_normal(self,k,d,sigma):
        w = np.random.randn(k,d) * sigma
        b = np.random.randn(d) * sigma
        return w,b
    def inicial_uniform(self,k,d,epsilon):
        w = np.random.uniform(-epsilon,epsilon,(k,d))
        b = np.random.uniform(-epsilon,epsilon,d)
        return w,b
    def inicial_Xavier(self,k,d,n_in,n_out):
        w = np.random.randn(k,d) * (2/(n_in+n_out))
        b = np.random.randn(d) * (2/(n_in+n_out))
        return w,b
    def inicial_He(self,k,d,n_in):
        w = np.random.randn(k, d) * (2 / (n_in))
        b = np.random.randn(d) * (2 / (n_in))
        return w, b

    def softmax(self,arr):
        for i in range(len(arr[0,:])):
            arr[:, i]-= np.max(arr[:,i])
            lst = np.exp(arr[:,i])
            arr[:,i] = lst/sum(lst)
        return arr

    def calc_y(self,w,b,X):
        #matrix k*N
        y = w.dot(X.T) + b
        y = self.softmax(y)
        return y

    def calc_nablaW(self,y,t,X):
        return (y-t.T).dot(X)
    def calc_nablaB(self,y,t):
        return (y - t.T)

    def GD_count(self,count,gamma,w,b,t,X):
        for i in range(count):
            y = self.calc_y(w,b,X)
            w = w - gamma*self.calc_nablaW(y,t,X)
            b = b - gamma*self.calc_nablaB(y,t)
    def GD_epsilonW(self,epsilon,gamma,w,b,t,X):
        while True:
            y = self.calc_y(w, b, X)
            w_k = w - gamma*self.calc_nablaW(y,t,X)

def normalisation(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for i in range(len(X[0,:])):
        X[:, i] -= mean[i]
    for i in range(len(X[0, :])):
        if(std[i] == 0):
            X[:,i] = 0
        else:
            X[:,i] /= std[i]
    return X

X = digits.data
tar = digits.target
LogReg(X,tar,10,0.8,0.1)


conf = np.zeros(3)
    idxconf = np.zeros(3,np.int8)
    nconf =  np.zeros(3)
    idxnconf = np.zeros(3,np.int8)
    tr = np.zeros(3,np.int8)
    f = np.zeros(3,np.int8)
    p = np.zeros(3,np.int8)
    for i in range(len(y[:, 0])):
        cl = np.argmax(y[i, :])
        c = np.max(y[i, :])
        if(cl == t[i]):
            if(conf[0] < c):
                ik = idxconf[0]
                k = conf[0]
                l = conf[1]
                il = idxconf[1]
                conf[0] = c
                conf[1] = k
                conf[2] = l
                idxconf[0] = i
                idxconf[1] = ik
                idxconf[2] = il
                p1 = p[0]
                p2 = p[1]
                p[0] = cl
                p[1] = p1
                p[2] = p2
            elif (conf[1] < c):
                k = conf[1]
                ik = idxconf[1]
                conf[1] = c
                conf[2] = k
                idxconf[1] = i
                idxconf[2] = ik
                p1 = p[1]
                p[1] = cl
                p[2] = p1
            elif (conf[2] < c):
                conf[2] = c
                idxconf[2] = i
                p[2] = cl
        elif (cl != t[i]):
            if (nconf[0] < c):
                ik = idxnconf[0]
                k = nconf[0]
                l = nconf[1]
                il = idxnconf[1]
                nconf[0] = c
                nconf[1] = k
                nconf[2] = l
                idxnconf[0] = i
                idxnconf[1] = ik
                idxnconf[2] = il
                f1 = f[0]
                f2 = f[1]
                f[0] = cl
                f[1] = f1
                f[2] = f2
                t1 = tr[0]
                t2 = tr[1]
                tr[0] = t[i]
                tr[1] = t1
                tr[2] = t2
            elif (nconf[1] < c):
                k = nconf[1]
                ik = idxnconf[1]
                nconf[1] = c
                nconf[2] = k
                idxnconf[1] = i
                idxnconf[2] = ik
                f1 = f[1]
                f[1] = cl
                f[2] = f1
                t1 = tr[1]
                tr[1] = t[i]
                tr[2] = t1
            elif (nconf[2] < c):
                nconf[2] = c
                idxnconf[2] = i
                f[2] = cl
                tr[2] = t[i]
    print(np.argmax(y[idxnconf[0],:]))
    print(np.argmax(y[idxnconf[1], :]))
    print(np.argmax(y[idxnconf[2], :]))
    print(t[idxnconf[0]])
    print(t[idxnconf[1]])
    print(t[idxnconf[2]])
    print(tr)
    print(f)
    print(p)
    return idxconf,idxnconf