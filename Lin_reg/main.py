import core
import matplotlib.pyplot as plt
import numpy as np
x,t,z = core.create_data_set()
st_deviat = np.zeros((100,))
e_arr = np.zeros(101)
y = np.zeros((1000,101))
for i in range(101):
    f = core.make_plan_matrix(i,x)
    w = ((np.linalg.inv(f.T.dot(f))).dot(f.T)).dot(t)
    y[:,i] = core.find_y(f,w)
    e = core.find_e(y[:,i],t)
    e_arr[i] = e
m = np.arange(0,101)
plt.plot(m,e_arr)
plt.show()
plt.plot(x,t,'*')
plt.plot(x,y[:,1])
plt.plot(x,z)
plt.show()
plt.plot(x,t,'*')
plt.plot(x,y[:,8])
plt.plot(x,z)
plt.show()
plt.plot(x,t,'*')
plt.plot(x,y[:,100])
plt.plot(x,z)
plt.show()
