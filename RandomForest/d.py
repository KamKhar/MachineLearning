import numpy as np
from sklearn.datasets import load_digits

class Node:
    def __init__(self):
        self.right = None
        self.left = None
        self.split_ind = None
        self.split_val = None
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

def inf_gain(ti,ti0,ti1,num_clas):
    en0 = jiny(ti0,num_clas)
    en1 = jiny(ti1,num_clas)
    en = jiny(ti,num_clas)
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

def buildTree(dig,tar,nod,tree,depth):
    nod.entropy_val = jiny(tar,10)
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
                IG = inf_gain(tar,tar[ind_right],tar[ind_left],10)
                if(IG > IG_max):
                    ax_max = ax
                    tay_max = j
                    IG_max = IG
                    ind_leftM = ind_left
                    ind_rightM = ind_right
        d_left = dig[ind_leftM,:]
        d_right = dig[ind_rightM,:]
        nod.split_ind = ax_max
        nod.split_val = tay_max
        depth += 1
        nod.left = Node()
        nod.right = Node()
        buildTree(d_left,tar[ind_leftM],nod.left,tree,depth)
        buildTree(d_right, tar[ind_rightM], nod.right, tree,depth)

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

def buidForest(numtrees,x_train,t_train):
    treearr = np.array([])
    for i in range(numtrees):
        tree = DT(8,0.05,25,8)
        buildTree(x_train, t_train, tree.root, tree, tree.root_depth)
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
digits = load_digits()
X1 = digits.data
tar = digits.target
X = standartisation(X1)
img = digits.images
x_train,t_train,x_valid,t_valid,img_valid = data_set(X,tar,img,0.8)


#tree = DT(10,0.0001,25,8)
#buildTree(x_train,t_train,tree.root,tree,tree.root_depth)

treearr = buidForest(10,x_train,t_train)
arr = testForestarr(treearr,x_train)
CM1 = confusionmatx(t_train,arr ,10)
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

#find = test(x_valid,tree)
#CM = confusionmatx(t_valid,find,10)
#print(CM)
#Ac = Accuracy(CM)
#print(Ac)
