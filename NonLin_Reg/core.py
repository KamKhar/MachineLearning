import numpy as np
def create_data(N):
    x = np.linspace(0,1,N)
    e = 10*np.random.randn(N)
    t = np.array([[]])


    z = np.zeros(N)
    for i in range(N):
        z[i] = 20 * np.sin(2 * np.pi * 3 * x[i]) + 100 * np.exp(x[i])
        t = np.append(t,20 * np.sin(2*np.pi *3*x[i])+ 100*np.exp(x[i]) + e[i])
    return (x, t, z)
def data_set(x,t,tr,val):
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
possible_functions = [lambda x:x**0,lambda x:x**1,lambda x:x**2,lambda x:x**3,
                      lambda x:x**4,lambda x:x**5,lambda x:x**6,lambda x:x**7,
                      lambda x:x**8,lambda x:x**9,lambda x:x**10,lambda x:np.sin(x),
                      lambda x:np.cos(x),lambda x:np.exp(x),lambda x:np.sqrt(x)]
def get_design_matrix(basic_func_ind,arr):
    F = np.zeros((len(arr),len(basic_func_ind)))
    for i in range(len(basic_func_ind)):
        F[:,i] = possible_functions[basic_func_ind[i]](arr)
    return F
def get_w(F,lamb,arr):
    I = np.eye(F.shape[1])
    w = ((np.linalg.inv(F.T.dot(F) + lamb*I)).dot(F.T)).dot(arr)
    return(w)
def find_y(f,w):
    y = w.dot(f.T)
    return(y)
def get_error(w,t,F,lamb):
    y = w.dot(F.T)
    e = 0
    for i in range(len(t)):
        e += (y[i] - t[i]) ** 2
    e = e / 2
    e += lamb/2 * w.T.dot(w)
    return e