import core
import matplotlib.pyplot as plt
import numpy as np
x,t,z = core.create_data(1000)
x_train,t_train,x_valid,t_valid,x_test,t_test = core.data_set(x,t,0.8,0.1)
lambdas = np.array([0.00001,0.0005,0.0001,0.005,0.001,0.05,0.01,0.5,0.1,1,5,10,50,100])

N = 1000
E_min = 10**10
lamb_min = -1
w_best = np.array([])
basics_func_min = np.array([])
for i in range(N):
    current_lambda = np.random.choice(lambdas)
    current_basics_func_ind = np.random.choice(np.arange(15),10,replace=False)
    F_cur = core.get_design_matrix(current_basics_func_ind,x_train)
    w_cur = core.get_w(F_cur,current_lambda,t_train)
    F_valid = core.get_design_matrix(current_basics_func_ind,x_valid)
    E_cur = core.get_error(w_cur,t_valid,F_valid,current_lambda)
    if(E_cur < E_min):
        E_min = E_cur
        lamb_min = current_lambda
        basics_func_min = current_basics_func_ind
        w_best = w_cur

F = core.get_design_matrix(basics_func_min,x_test)
E = core.get_error(w_best,t_test,F,lamb_min)
y = core.find_y(F,w_best)
plt.plot(x_test,t_test,'*')
plt.plot(x_test,y,'+')
plt.plot(x,z)
plt.show()
print(lamb_min)
print(basics_func_min)