import numpy as np
def create_data_set():
    x = np.linspace(0,1,1000)
    e = 10*np.random.randn(1000)
    t = np.array([[]])
    z = np.zeros(1000)
    for i in range(1000):
        z[i] = 20 * np.sin(2 * np.pi * 3 * x[i]) + 100 * np.exp(x[i])
        t = np.append(t,20 * np.sin(2*np.pi *3*x[i])+ 100*np.exp(x[i]) + e[i])
    return(x,t,z)
def make_plan_matrix(i,t):
    f = np.zeros((1000,i+1))
    for j in range(i+1):
        f[:,j] = t**j
    return(f)
def find_y(f,w):
    #y = np.zeros(1000)
    #for i in range(1000):
        #y[i]=w.T.dot(f[i,:])
    y = w.dot(f.T)
    return(y)
def find_e(y,t):
    e = 0
    for i in range(1000):
        e += (y[i] - t[i])**2
    e = e/2
    return e