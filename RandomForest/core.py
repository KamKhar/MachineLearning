import numpy as np
from sklearn.datasets import load_digits

class Node:
    def __init__(self):
        self.right = None
        self.left = None
        self.split_ind = None
        self.split_val = None
        self.splitcoef = None
        self.splitbase = None
        self.entropy_val = None
        self.T = None

class DT:
    def __init__(self,max_depth,min_entropy,min_elem,max_features):
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_elem = min_elem
        self.max_features = max_features
        self.root = Node()
        self.root_depth = 1

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

def data_set1(x,t,tr,val):
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

def tar2onehotenc(t,k):
    tar = np.zeros((len(t),k))
    for i in range(len(t)):
        tar[i,t[i]] = 1
    return tar

def psi():
    return np.random.randint(0,64)

def tay(arr):
    t1 = arr.min()
    t2 = arr.max()
    return np.random.uniform(t1,t2,18)

def countElClass(arr,numclases):
    cEl = np.zeros(numclases)
    for i in arr:
        cEl[i] += 1
    return cEl

def shen(tar,num_clas):
    entr = 0.0
    l = len(tar)
    arr2 = countElClass(tar,num_clas)
    for i in arr2:
        if(i != 0):
            entr += ((i/l)*np.log(i/l))
    entr *= -1
    return entr

def jiny(tar,num_clas):
    entr = 0.0
    l = len(tar)
    arr2 = countElClass(tar,num_clas)
    for i in arr2:
        if(i != 0):
            entr += ((i/l)**2)
    entr = 1 - entr
    return entr

def inf_gain(ti,ti0,ti1,num_clas , type = 0):
    if type == 0:
        en0 = shen(ti0,num_clas)
        en1 = shen(ti1,num_clas)
        en = shen(ti,num_clas)
    elif type == 1:
        en0 = jiny(ti0, num_clas)
        en1 = jiny(ti1, num_clas)
        en = jiny(ti, num_clas)
    l = len(ti)
    l0 = len(ti0)
    l1 = len(ti1)
    return en - (((l0/l)* en0) + ((l1/l)* en1) )

def split(dig,tar,ax,t):
    dig_left = np.zeros((0,len(dig[0,:])))
    dig_right = np.zeros((0,len(dig[0,:])))
    tar_left = np.array([],np.int8)
    tar_right = np.array([],np.int8)
    l = len(dig[:, 0])
    for i in range(l):
        if (dig[i, ax] >= t):
            dig_right = np.append(dig_right, [dig[i, :]], axis = 0)
            tar_right =np.append(tar_right, tar[i])
        else:
            dig_left = np.append(dig_left, [dig[i, :]], axis = 0)
            tar_left = np.append(tar_left, tar[i])
    return dig_left,dig_right,tar_left,tar_right

def termarr(tar,num_clas):
    l = len(tar)
    if (l == 0):
        return np.zeros(10)
    return countElClass(tar, num_clas) / l

def findindex(dig,tay,ax):
    left = np.array([],np.int8)
    right = np.array([],np.int8)
    l = len(dig[:,0])
    for i in range(l):
        if(dig[i,ax] < tay):
            left = np.append(left,i)
        else:
            right = np.append(right,i)
    return right,left

def buildTreePar(dig,tar,nod,tree,depth,type = 0):
    if type == 0:
        nod.entropy_val = shen(tar,10)
    elif type == 1:
        nod.entropy_val = jiny(tar, 10)
    if(depth >= tree.max_depth or nod.entropy_val <= tree.min_entropy or len(tar)<=tree.min_elem):
        nod.T = termarr(tar,10)
    else:
        ax_max = 0
        tay_max = 0.0
        IG_max = 0
        ind_leftM = np.array([])
        ind_rightM = np.array([])
        for i in range(tree.max_features):
            ax = psi()
            t=tay(dig[:,ax])
            for j in t:
                ind_right,ind_left = findindex(dig,j,ax)
                IG = inf_gain(tar,tar[ind_right],tar[ind_left],10,type)
                if(IG > IG_max):
                    ax_max = ax
                    tay_max = j
                    IG_max = IG
                    ind_leftM = ind_left
                    ind_rightM = ind_right
        if (len(ind_leftM) == 0):
            d_left = []
            tar_left = []
        else:
            d_left = dig[ind_leftM, :]
            tar_left = tar[ind_leftM]
        if (len(ind_rightM) == 0):
            d_right = []
            tar_right = []
        else:
            d_right = dig[ind_rightM, :]
            tar_right = tar[ind_rightM]
        nod.split_ind = ax_max
        nod.split_val = tay_max
        depth += 1
        nod.left = Node()
        nod.right = Node()
        buildTreePar(d_left,tar_left,nod.left,tree,depth,type)
        buildTreePar(d_right, tar_right, nod.right, tree,depth,type)

def test(dig,tree):
    l = len(dig[:,0])
    res = np.zeros(l,np.int8)
    for i in range(l):
        node = tree.root
        while (node.right != None and node.left != None):
            if(dig[i,node.split_ind] < node.split_val):
                node = node.left
            else:
                node = node.right
        res[i] = np.argmax(node.T)
    return res

def test1(dig,tree):
    node = tree.root
    while (node.right != None and node.left != None):
        if(dig[node.split_ind] < node.split_val):
            node = node.left
        else:
            node = node.right
    res= node.T
    return res

def test1NotPar(dig,tree):
    node = tree.root
    while (node.right != None and node.left != None):
        if(dig[node.split_ind]@node.splitcoef < node.split_val):
            node = node.left
        else:
            node = node.right
    res= node.T
    return res

def testForestarrNotPar(treearr,x_test):
    n = len(treearr)
    arr = np.zeros((n,10))
    l = len(x_test)
    arr1 = np.zeros(l,np.int8)
    for j in range(l):
        for i in range(n):
            arr[i,:] = test1NotPar(x_test[j],treearr[i])
        arr1[j] = np.argmax(arr.sum(axis=0)/n)
    return arr1

def confusionmatx(real, find, k):
    CM = np.zeros((k, k))
    for i in range(len(find)):
        CM[real[i], find[i]] += 1
    return CM


def Accuracy(CM):
    s = 0
    for i in range(len(CM[0,:])):
        s += CM[i,i]
    return s/sum(sum(CM))

def prtre(node):
    if(node.right == None):
        print("Termnode")
    else:
        print(node.split_ind)
        prtre(node.right)
        prtre(node.left)

def buidForestPar(numtrees,MD,MEnt,Mel,MF,x_train,t_train,type = 0):
    treearr = np.array([])
    for i in range(numtrees):
        tree = DT(MD,MEnt,Mel,MF)
        buildTreePar(x_train, t_train, tree.root, tree, tree.root_depth,type)
        treearr = np.append(treearr,tree)
    return treearr

def testForestarr(treearr,x_test):
    n = len(treearr)
    arr = np.zeros((n,10))
    l = len(x_test)
    arr1 = np.zeros(l,np.int8)
    for j in range(l):
        for i in range(n):
            arr[i,:] = test1(x_test[j],treearr[i])
        arr1[j] = np.argmax(arr.sum(axis=0)/n)
    return arr1

def findindexNotPar(arr,tay):
    left = np.array([], np.int8)
    right = np.array([], np.int8)
    l = len(arr)
    for i in range(l):
        if (arr[i] < tay):
            left = np.append(left, i)
        else:
            right = np.append(right, i)
    return right, left

def psiNotPar(count):
    arr = np.random.choice(np.arange(64),count,replace=False)
    coeff = np.random.uniform(1,9,count)
    return arr,coeff

def buildTreeNotPar(dig,tar,nod,tree,depth,count,type = 0):
    if type == 0:
        nod.entropy_val = shen(tar,10)
    elif type == 1:
        nod.entropy_val = jiny(tar, 10)

    if(depth >= tree.max_depth or nod.entropy_val <= tree.min_entropy or len(tar)<=tree.min_elem):
        nod.T = termarr(tar,10)
    else:
        ax_max = []
        tay_max = 0.0
        coef_max = []
        IG_max = 0
        ind_leftM = np.array([])
        ind_rightM = np.array([])
        for i in range(tree.max_features):
            ax,coef = psiNotPar(count)
            a = np.dot(dig[:,ax],coef)
            t=tay(a)
            for j in t:
                ind_right,ind_left = findindexNotPar(a,j)
                IG = inf_gain(tar,tar[ind_right],tar[ind_left],10,type)
                if(IG > IG_max):
                    ax_max = ax
                    tay_max = j
                    coef_max = coef
                    IG_max = IG
                    ind_leftM = ind_left
                    ind_rightM = ind_right
        if(len(ind_leftM) == 0):
            d_left = []
            tar_left = []
        else:
            d_left = dig[ind_leftM,:]
            tar_left = tar[ind_leftM]
        if(len(ind_rightM) == 0):
            d_right = []
            tar_right = []
        else:
            d_right = dig[ind_rightM,:]
            tar_right = tar[ind_rightM]
        nod.split_ind = ax_max
        nod.split_val = tay_max
        nod.splitcoef = coef_max
        depth += 1
        nod.left = Node()
        nod.right = Node()
        buildTreeNotPar(d_left,tar_left,nod.left,tree,depth,count,type)
        buildTreeNotPar(d_right, tar_right, nod.right, tree,depth,count,type)

def buidForestNotPar(numtrees,MD,MEnt,Mel,MF,x_train,t_train,count,type = 0):
    treearr = np.array([])
    for i in range(numtrees):
        tree = DT(MD,MEnt,Mel,MF)
        buildTreeNotPar(x_train, t_train, tree.root, tree, tree.root_depth,count,type)
        treearr = np.append(treearr,tree)
    return treearr

possible_functions = [lambda x:x**0,lambda x:x**1,lambda x:x**2,lambda x:x**3,
                      lambda x:x**4,lambda x:x**5,lambda x:x**6,lambda x:x**7,
                      lambda x:x**8,lambda x:x**9,lambda x:x**10,lambda x:np.sin(x),
                      lambda x:np.cos(x)]

def psiBase(count):
    arr1 = np.random.choice(np.arange(64), count, replace=False)
    arr = np.random.choice(np.arange(13), count, replace=False)
    return arr1,arr

def getvalue(x,arr1,arr,count):
    a = np.zeros(len(x[:,0]))
    for i in range(count):
        a += possible_functions[arr[i]](x[:,arr1[i]])
    return a

def getvaluevar(x,arr1,arr,count):
    a = 0.0
    for i in range(count):
        a += possible_functions[arr[i]](x[arr1[i]])
    return a

def buildTreeBase(dig,tar,nod,tree,depth,count,type = 0):
    if type == 0:
        nod.entropy_val = shen(tar,10)
    elif type == 1:
        nod.entropy_val = jiny(tar, 10)

    if(depth >= tree.max_depth or nod.entropy_val <= tree.min_entropy or len(tar)<=tree.min_elem):
        nod.T = termarr(tar,10)
    else:
        ax_max = []
        tay_max = 0.0
        IG_max = 0
        basef_max = []
        ind_leftM = np.array([])
        ind_rightM = np.array([])
        for i in range(tree.max_features):
            ax,basef = psiBase(count)
            a = getvalue(dig,ax,basef,count)
            t=tay(a)
            for j in t:
                ind_right,ind_left = findindexNotPar(a,j)
                IG = inf_gain(tar,tar[ind_right],tar[ind_left],10,type)
                if(IG > IG_max):
                    ax_max = ax
                    tay_max = j
                    basef_max = basef
                    IG_max = IG
                    ind_leftM = ind_left
                    ind_rightM = ind_right
        if(len(ind_leftM) == 0):
            d_left = []
            tar_left = []
        else:
            d_left = dig[ind_leftM,:]
            tar_left = tar[ind_leftM]
        if(len(ind_rightM) == 0):
            d_right = []
            tar_right = []
        else:
            d_right = dig[ind_rightM,:]
            tar_right = tar[ind_rightM]
        nod.split_ind = ax_max
        nod.split_val = tay_max
        nod.splitbase = basef_max
        depth += 1
        nod.left = Node()
        nod.right = Node()
        buildTreeBase(d_left,tar_left,nod.left,tree,depth,count,type)
        buildTreeBase(d_right, tar_right, nod.right, tree,depth,count,type)

def buidForestBase(numtrees,MD,MEnt,Mel,MF,x_train,t_train,count,type = 0):
    treearr = np.array([])
    for i in range(numtrees):
        tree = DT(MD,MEnt,Mel,MF)
        buildTreeBase(x_train, t_train, tree.root, tree, tree.root_depth,count,type)
        treearr = np.append(treearr,tree)
    return treearr

def test1Base(dig,tree):
    node = tree.root
    while (node.right != None and node.left != None):
        a = getvaluevar(dig,node.split_ind,node.splitbase,len(node.split_ind))
        if(a < node.split_val):
            node = node.left
        else:
            node = node.right
    res= node.T
    return res

def testForestarrBase(treearr,x_test):
    n = len(treearr)
    arr = np.zeros((n,10))
    l = len(x_test)
    arr1 = np.zeros(l,np.int8)
    for j in range(l):
        for i in range(n):
            arr[i,:] = test1Base(x_test[j],treearr[i])
        arr1[j] = np.argmax(arr.sum(axis=0)/n)
    return arr1

def buildTreeAllAx(dig,tar,nod,tree,depth,type = 0):
    if type == 0:
        nod.entropy_val = shen(tar,10)
    elif type == 1:
        nod.entropy_val = jiny(tar, 10)
    if(depth >= tree.max_depth or nod.entropy_val <= tree.min_entropy or len(tar)<=tree.min_elem):
        nod.T = termarr(tar,10)
    else:
        ax_max = 0
        tay_max = 0.0
        IG_max = 0
        ind_leftM = np.array([])
        ind_rightM = np.array([])
        for i in range(64):
            ax = i
            t=tay(dig[:,ax])
            for j in t:
                ind_right,ind_left = findindex(dig,j,ax)
                IG = inf_gain(tar,tar[ind_right],tar[ind_left],10,type)
                if(IG > IG_max):
                    ax_max = ax
                    tay_max = j
                    IG_max = IG
                    ind_leftM = ind_left
                    ind_rightM = ind_right
        if (len(ind_leftM) == 0):
            d_left = []
            tar_left = []
        else:
            d_left = dig[ind_leftM, :]
            tar_left = tar[ind_leftM]
        if (len(ind_rightM) == 0):
            d_right = []
            tar_right = []
        else:
            d_right = dig[ind_rightM, :]
            tar_right = tar[ind_rightM]
        nod.split_ind = ax_max
        nod.split_val = tay_max
        depth += 1
        nod.left = Node()
        nod.right = Node()
        buildTreeAllAx(d_left,tar_left,nod.left,tree,depth,type)
        buildTreeAllAx(d_right, tar_right, nod.right, tree,depth,type)

def bagging(x_train,t_train):
    N = len(t_train)
    ind = np.arange(N)
    ind_prm = np.random.permutation(ind)
    x = ind_prm[:np.int32(0.6*N)]
    x1 = x_train[x, :]
    t1 = t_train[x]
    return x1,t1

def buidForestAllAx(numtrees,MD,MEnt,Mel,MF,x_train,t_train,type = 0):
    treearr = np.array([])
    for i in range(numtrees):
        tree = DT(MD,MEnt,Mel,MF)
        x,t = bagging(x_train,t_train)
        buildTreeAllAx(x, t, tree.root, tree, tree.root_depth,type)
        treearr = np.append(treearr,tree)
    return treearr



digits = load_digits()
X1 = digits.data
tar = digits.target
X = standartisation(X1)
img = digits.images
x_train,t_train,x_valid,t_valid,img_valid = data_set(X,tar,img,0.8)

"""Дерево с параллельным разделением"""
treearr = buidForestPar(10,8,0.05,25,8,x_train,t_train,1)
"""8 аргумент функции  testForestarrPar  определяет используется энтропия или Джини"""
arr1 = testForestarr(treearr,x_train)
CM1 = confusionmatx(t_train,arr1 ,10)
print("CM for train set")
print(CM1)
Ac = Accuracy(CM1)
print("Accuracy for train set:",Ac)

arr = testForestarr(treearr,x_valid)
CM = confusionmatx(t_valid,arr ,10)
print("CM for test set")
print(CM)
Ac1 = Accuracy(CM)
print("Accuracy for test set:",Ac1)


"""Дерево с непараллельным разделением"""
#treearr = buidForestNotPar(10,8,0.05,25,8,x_train,t_train,6,1)
"""8 аргумент функции testForestarrNotPar -  количество переменных в гиперплоскости," \
 9 аргумент  определяет используется энтропия или Джини"""
# arr1 = testForestarrNotPar(treearr,x_train)
# CM1 = confusionmatx(t_train,arr1 ,10)
# print("CM for train set")
# print(CM1)
# Ac = Accuracy(CM1)
# print("Accuracy for train set:",Ac)
#
# arr = testForestarrNotPar(treearr,x_valid)
# CM = confusionmatx(t_valid,arr ,10)
# print("CM for test set")
# print(CM)
# Ac1 = Accuracy(CM)
# print("Accuracy for test set:",Ac1)

"""Дерево с базовыми функциями"""
#treearr = buidForestBase(10,8,0.05,25,8,x_train,t_train,6,1)
"""8 аргумент функции testForestarrNotPar -  количество функция," \
 9 аргумент  определяет используется энтропия или Джини"""
# arr1 = testForestarrBase(treearr,x_train)
# CM1 = confusionmatx(t_train,arr1 ,10)
# print("CM for train set")
# print(CM1)
# Ac = Accuracy(CM1)
# print("Accuracy for train set:",Ac)
#
# arr = testForestarrBase(treearr,x_valid)
# CM = confusionmatx(t_valid,arr ,10)
# print("CM for test set")
# print(CM)
# Ac1 = Accuracy(CM)
# print("Accuracy for test set:",Ac1)

"""Дерево с полным перебором параметров с параллельным разделением """
#treearr = buidForestAllAx(10,8,0.05,25,8,x_train,t_train,1)
"""8 аргумент функции  testForestarrPar  определяет используется энтропия или Джини"""
# arr1 = testForestarr(treearr,x_train)
# CM1 = confusionmatx(t_train,arr1 ,10)
# print("CM for train set")
# print(CM1)
# Ac = Accuracy(CM1)
# print("Accuracy for train set:",Ac)
#
# arr = testForestarr(treearr,x_valid)
# CM = confusionmatx(t_valid,arr ,10)
# print("CM for test set")
# print(CM)
# Ac1 = Accuracy(CM)
# print("Accuracy for test set:",Ac1)

"""Валидиция"""
# bestdepth = 0
# best_entropy = 0
# best_elem = 0
# best_features = 0
# best_numtr = 0
# Max_Ac = 0
#
# x_train,t_train,x_valid,t_valid,x_test,t_test = data_set1(X,tar,0.8,0.1)
#
# for i in range(15):
#     depth = np.random.randint(5,15)
#     entropy = np.random.uniform(0.001,0.1)
#     elem = np.random.randint(10,35)
#     features = np.random.randint(5,15)
#     numtr = np.random.randint(7,16)
#
#     treearr = buidForestPar(numtr, depth, entropy, elem, features, x_train, t_train, 1)
#     arr = testForestarr(treearr, x_valid)
#     CM = confusionmatx(t_valid,arr ,10)
#     Ac1 = Accuracy(CM)
#
#     if(Ac1 > Max_Ac):
#         best_entropy = entropy
#         best_elem = elem
#         bestdepth = depth
#         best_features = features
#         best_numtr = numtr
#         Max_Ac = Ac1
#
# arr = testForestarr(treearr,x_test)
# CM = confusionmatx(t_test,arr ,10)
# print("CM for test set")
# print(CM)
# Ac1 = Accuracy(CM)
# print("Accuracy for test set:",Ac1)

"""Отдельное дерево с параллельным разбиением"""
#tree = DT(10,0.0001,25,8)
#buildTreePar(x_train,t_train,tree.root,tree,tree.root_depth)
#find = test(x_valid,tree)
#CM = confusionmatx(t_valid,find,10)
#print(CM)
#Ac = Accuracy(CM)
#print(Ac)
