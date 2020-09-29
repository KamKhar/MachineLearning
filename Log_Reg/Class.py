import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import preprocessing
import pickle
digits = load_digits()


def standartisation(X):
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

def data_set(x,t,img,tr):
    N = len(t)
    ind = np.arange(N)
    ind_prm = np.random.permutation(ind)
    train_ind = ind_prm[:np.int32(tr*N)]
    valid_ind = ind_prm[np.int32(tr*N):]
    x_train = x[train_ind,:]
    t_train = t[train_ind]
    x_valid = x[valid_ind,:]
    t_valid = t[valid_ind]
    img_valid = img[valid_ind,:,:]
    return x_train,t_train,x_valid,t_valid,img_valid

def tar2onehotenc(t,k):
    tar = np.zeros((len(t),k))
    for i in range(len(t)):
        tar[i,t[i]] = 1
    return tar

def inicial_normal(k,d,sigma):
    w = np.random.randn(k,d) * sigma
    b = np.random.randn(k) * sigma
    return w,b
def inicial_uniform(k,d,epsilon):
    w = np.random.uniform(-epsilon,epsilon,(k,d))
    b = np.random.uniform(-epsilon,epsilon,k)
    return w,b
def inicial_Xavier(k,d,n_in,n_out):
    w = np.random.randn(k,d) * (2/(n_in+n_out))
    b = np.random.randn(k) * (2/(n_in+n_out))
    return w,b
def inicial_He(k,d,n_in):
    w = np.random.randn(k, d) * (2 / (n_in))
    b = np.random.randn(k) * (2 / (n_in))
    return w, b

def softmax(arr):
    for i in range(len(arr[:,0])):
        arr[i, :]-= np.max(arr[i,:])
        lst = np.exp(arr[i,:])
        arr[i,:] = lst/sum(lst)
    return arr

def calc_y(w,b,X):
    #matrix N*k
    y = (w.dot(X.T)).T + b
    y = softmax(y)
    return y
def calc_costfunc(t,x,w,b,lamb = 0):
    e = 0
    y = (w.dot(x.T)).T + b
    for i in range(len(x[:,0])):
        e+= np.log(sum(np.exp(y[i,:]))) - y[i,t[i]]
    e+=(lamb / 2)* np.sum(w**2)
    return e
def calc_nablaW(y,t,X,w,lamb = 0):
    return ((y-t).T).dot(X)+ lamb*w
def calc_nablaB(y,t):
    return (y - t).T.dot(np.ones(len(y[:,0])))

def GD_count(w,b,t1hot_train,t_train,x_train,t_valid,x_valid,lamb = 0.1,count = 150,gamma=0.01):
    er_tr = np.array([])
    ac_tr = np.array([])
    er_val = np.array([])
    ac_val = np.array([])
    y = calc_y(w, b, x_train)
    for i in range(count):
        w = w - gamma*calc_nablaW(y,t1hot_train,x_train,w,lamb)
        b = b - gamma*calc_nablaB(y,t1hot_train)

        y = calc_y(w, b, x_train)
        CM = confusionmatx(t_train, y, 10)
        ac = Accuracy(CM)

        y1 = calc_y(w,b,x_valid)
        CM1 = confusionmatx(t_valid, y1, 10)
        ac1 = Accuracy(CM1)

        e = calc_costfunc(t_train,x_train,w,b,lamb)

        e1 = calc_costfunc(t_valid, x_valid, w, b,lamb)

        er_tr = np.append(er_tr,e)
        ac_tr = np.append(ac_tr,ac)

        er_val = np.append(er_val,e1)
        ac_val = np.append(ac_val,ac1)
        if (i % 10 == 0):
            print("Accuracy train: ", ac)
            print("Accuracy valid: ", ac1)
            print("Error train: ", e)
            print("Error valid: ", e1)
            it = np.arange(len(er_tr))
            plt.plot(it, er_tr)
            plt.title("Целевая функция на обучающей выборке.Итерация: " +i.__str__())
            plt.show()
            plt.plot(it, ac_tr)
            plt.title("Точность на обучающей выборке.Итерация: " +i.__str__())
            plt.show()
            plt.plot(it, er_val)
            plt.title("Целевая функция на валидационной выборке.Итерация: " +i.__str__())
            plt.show()
            plt.plot(it, ac_val)
            plt.title("Точность на валидационной выборке.Итерация: " +i.__str__())
            plt.show()
    return w, b, er_tr, ac_tr, er_val, ac_val
def GD_epsilonW(w,b,t1hot_train,t_train,x_train,t_valid,x_valid,epsilon=0.1,gamma=0.01):
    i = 0
    er_tr = np.array([])
    ac_tr = np.array([])
    er_val = np.array([])
    ac_val = np.array([])
    y = calc_y(w, b, x_train)
    while True:
        w_k = w - gamma*calc_nablaW(y,t1hot_train,x_train)
        if(np.linalg.norm(w_k - w) < epsilon):
            break

        w = w_k
        b = b - gamma*calc_nablaB(y,t1hot_train)

        y = calc_y(w, b, x_train)
        CM = confusionmatx(t_train, y, 10)
        ac = Accuracy(CM)

        y1 = calc_y(w, b, x_valid)
        CM1 = confusionmatx(t_valid, y1, 10)
        ac1 = Accuracy(CM1)

        e = calc_costfunc(t_train, x_train, w, b)

        e1 = calc_costfunc(t_valid, x_valid, w, b)

        er_tr = np.append(er_tr, e)
        ac_tr = np.append(ac_tr, ac)

        er_val = np.append(er_val, e1)
        ac_val = np.append(ac_val, ac1)
        if (i % 10 == 0):
            print("Accuracy train: ", ac)
            print("Accuracy valid: ", ac1)
            print("Error train: ", e)
            print("Error valid: ", e1)
            it = np.arange(len(er_tr))
            plt.plot(it, er_tr)
            plt.title("Целевая функция на обучающей выборке.Итерация: " +i.__str__())
            plt.show()
            plt.plot(it, ac_tr)
            plt.title("Точность на обучающей выборке.Итерация: " +i.__str__())
            plt.show()
            plt.plot(it, er_val)
            plt.title("Целевая функция на валидационной выборке.Итерация: " +i.__str__())
            plt.show()
            plt.plot(it, ac_val)
            plt.title("Точность на валидационной выборке.Итерация: " +i.__str__())
            plt.show()
        i+=1
    return w, b, er_tr, ac_tr, er_val, ac_val

def GD_valid(w,b,t1hot_train,t_train,x_train,t_valid,x_valid,gamma=0.01):
    e = calc_costfunc(t_valid,x_valid,w,b)
    er_tr = np.array([])
    ac_tr = np.array([])
    er_val = np.array([])
    ac_val = np.array([])
    i = 0
    y = calc_y(w, b, x_train)
    while True:
        w_k = w - gamma * calc_nablaW(y, t1hot_train, x_train)
        b_k = b - gamma*calc_nablaB(y,t1hot_train)
        e_k = calc_costfunc(t_valid,x_valid,w_k,b_k)
        if(e < e_k):
            break
        w = w_k
        b = b_k
        e = e_k

        y = calc_y(w, b, x_train)
        CM = confusionmatx(t_train, y, 10)
        ac = Accuracy(CM)

        y1 = calc_y(w, b, x_valid)
        CM1 = confusionmatx(t_valid, y1, 10)
        ac1 = Accuracy(CM1)

        e1 = calc_costfunc(t_train, x_train, w, b)

        er_tr = np.append(er_tr, e1)
        ac_tr = np.append(ac_tr, ac)

        er_val = np.append(er_val, e_k)
        ac_val = np.append(ac_val, ac1)
        if (i % 10 == 0):
            print("Accuracy train: ", ac)
            print("Accuracy valid: ", ac1)
            print("Error train: ", e1)
            print("Error valid: ", e_k)
            it = np.arange(len(er_tr))
            plt.plot(it, er_tr)
            plt.title("Целевая функция на обучающей выборке.Итерация: " +i.__str__())
            plt.show()
            plt.plot(it, ac_tr)
            plt.title("Точность на обучающей выборке.Итерация: " +i.__str__())
            plt.show()
            plt.plot(it, er_val)
            plt.title("Целевая функция на валидационной выборке.Итерация: " +i.__str__())
            plt.show()
            plt.plot(it, ac_val)
            plt.title("Точность на валидационной выборке.Итерация: " +i.__str__())
            plt.show()
        i+=1
    return w, b, er_tr, ac_tr, er_val, ac_val

def GD_epsilonE(w,b,t1hot_train,t_train,x_train,t_valid,x_valid,epsilon=5,gamma =0.01):
    i = 0
    er_tr = np.array([])
    ac_tr = np.array([])
    er_val = np.array([])
    ac_val = np.array([])
    y = calc_y(w, b, x_train)
    while True:
        nW = calc_nablaW(y,t1hot_train,x_train)
        if(np.linalg.norm(nW) < epsilon):
            break
        w =  w - gamma*nW
        b = b - gamma*calc_nablaB(y,t1hot_train)

        y = calc_y(w, b, x_train)
        CM = confusionmatx(t_train, y, 10)
        ac = Accuracy(CM)

        y1 = calc_y(w, b, x_valid)
        CM1 = confusionmatx(t_valid, y1, 10)
        ac1 = Accuracy(CM1)

        e = calc_costfunc(t_train, x_train, w, b)

        e1 = calc_costfunc(t_valid, x_valid, w, b)

        er_tr = np.append(er_tr, e)
        ac_tr = np.append(ac_tr, ac)

        er_val = np.append(er_val, e1)
        ac_val = np.append(ac_val, ac1)
        if (i % 10 == 0):
            print("Accuracy train: ", ac)
            print("Accuracy valid: ", ac1)
            print("Error train: ", e)
            print("Error valid: ", e1)
            it = np.arange(len(er_tr))
            plt.plot(it, er_tr)
            plt.title("Целевая функция на обучающей выборке.Итерация: " +i.__str__())
            plt.show()
            plt.plot(it, ac_tr)
            plt.title("Точность на обучающей выборке.Итерация: " +i.__str__())
            plt.show()
            plt.plot(it, er_val)
            plt.title("Целевая функция на валидационной выборке.Итерация: " +i.__str__())
            plt.show()
            plt.plot(it, ac_val)
            plt.title("Точность на валидационной выборке.Итерация: " +i.__str__())
            plt.show()
        i+=1
    return w, b,er_tr,ac_tr,er_val,ac_val

def confusionmatx(t,y,k):
    CM = np.zeros((k,k))
    for i in range(len(y[:,0])):
        cl = np.argmax(y[i,:])
        CM[t[i],cl]+=1
    return CM

def Accuracy(CM):
    s = 0
    for i in range(len(CM[0,:])):
        s += CM[i,i]
    return s/sum(sum(CM))
def conf(y,t):
    arr = np.array([],np.int8)
    arr1 = np.array([],np.int8)
    val_arr = np.array([])
    val_arr1 = np.array([])
    for i in range(len(y[:, 0])):
       cl = np.argmax(y[i,:])
       if(t[i] != cl):
            arr = np.append(arr,i)
            val_arr =np.append(val_arr,np.max(y[i,:]))
       else:
            arr1 = np.append(arr1, i)
            val_arr1 = np.append(val_arr1,np.max(y[i, :]))
    sortv = np.argsort(val_arr)[::-1]
    sortv1 = np.argsort(val_arr1)[::-1]
    idx = np.array([arr[sortv[0]],arr[sortv[1]],arr[sortv[2]]])
    idx1 = np.array([arr1[sortv1[0]], arr1[sortv1[1]], arr1[sortv1[2]]])
    return idx1,idx
def save (w,b,t_test,x_test,er_tr,ac_tr,er_val,ac_val):
    with open('W.pickle', 'wb') as f:
        pickle.dump(w, f)
    with open('b.pickle', 'wb') as f:
        pickle.dump(b, f)
    with open('t_test.pickle', 'wb') as f:
        pickle.dump(t_test, f)
    with open('x_test.pickle', 'wb') as f:
        pickle.dump(x_test, f)
    with open('er_tr.pickle', 'wb') as f:
        pickle.dump(er_tr, f)
    with open('ac_tr.pickle', 'wb') as f:
        pickle.dump(ac_tr, f)
    with open('er_val.pickle', 'wb') as f:
        pickle.dump(er_val, f)
    with open('ac_val.pickle', 'wb') as f:
        pickle.dump(ac_val, f)
def load():
    with open('W.pickle', 'rb') as f:
        w = pickle.load(f)
    with open('b.pickle', 'rb') as f:
        b = pickle.load(f)
    with open('t_test.pickle', 'rb') as f:
        t_test = pickle.load(f)
    with open('x_test.pickle', 'rb') as f:
        x_test = pickle.load(f)
    with open('er_tr.pickle', 'rb') as f:
        er_tr = pickle.load(f)
    with open('ac_tr.pickle', 'rb') as f:
        ac_tr = pickle.load(f)
    with open('er_val.pickle', 'rb') as f:
        er_val = pickle.load(f)
    with open('ac_val.pickle', 'rb') as f:
        ac_val = pickle.load(f)
    return w,b,t_test,x_test,er_tr,ac_tr,er_val,ac_val


X1 = digits.data
tar = digits.target
X = standartisation(X1)
img = digits.images
w, b = inicial_normal(10,64,1)
x_train,t_train,x_valid,t_valid,img_valid = data_set(X,tar,img,0.8)
t_1hottrain= tar2onehotenc(t_train,10)

y = calc_y(w,b,x_valid)
CM = confusionmatx(t_valid,y,10)
print("До обучения:")
print("Confusion matrix:")
print(CM)
print("Точность: ",Accuracy(CM))

w,b,er_tr,ac_tr,er_val,ac_val = GD_count(w,b,t_1hottrain,t_train,x_train,t_valid,x_valid)
#save(w,b,t_valid,x_valid,er_tr,ac_tr,er_val,ac_val)
#w,b,t_valid,x_valid,er_tr,ac_tr,er_val,ac_val = load()

print(calc_costfunc(t_valid, x_valid, w, b))
y = calc_y(w,b,x_valid)
CM = confusionmatx(t_valid,y,10)
print("После обучения:")
print("Confusion matrix:")
print(CM)
print("Точность: ",Accuracy(CM))

it = np.arange(len(er_tr))
plt.plot(it,er_tr)
plt.title("Целевая функция на обучающей выборке")
plt.show()
plt.plot(it,ac_tr)
plt.title("Точность на обучающей выборке")
plt.show()
plt.plot(it,er_val)
plt.title("Целевая функция на валидационной выборке")
plt.show()
plt.plot(it,ac_val)
plt.title("Точность на валидационной выборке")
plt.show()


idxconf,idxnconf = conf(y,t_valid)

#Верно определено
plt.imshow(img_valid[idxconf[0]])
plt.show()
print(t_valid[idxconf[0]])

plt.imshow(img_valid[idxconf[1]])
plt.show()
print(t_valid[idxconf[1]])

plt.imshow(img_valid[idxconf[2]])
plt.show()
print(t_valid[idxconf[2]])

#Неверно определено
plt.imshow(img_valid[idxnconf[0]])
plt.show()
print("Ожидание: ",t_valid[idxnconf[0]]," Реальность: ",np.argmax(y[idxnconf[0], :]))

plt.imshow(img_valid[idxnconf[1]])
plt.show()
print("Ожидание: ",t_valid[idxnconf[1]]," Реальность: ",np.argmax(y[idxnconf[1], :]))

plt.imshow(img_valid[idxnconf[2]])
plt.show()
print("Ожидание: ",t_valid[idxnconf[2]]," Реальность: ",np.argmax(y[idxnconf[2], :]))