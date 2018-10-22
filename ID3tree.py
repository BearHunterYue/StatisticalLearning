import numpy as np
from math import log
import copy
import pickle
# calculate the entropy of the dataset
def entropy(data):
    number=len(data)
    labelcount={}
    entropynumber=0.0
    for line in data:
        labelcount[line[-1]]=labelcount.get(line[-1],0)+1
    for key in labelcount:
        pro=float(labelcount[key])/number
        entropynumber-=pro*log(pro,2)
    return entropynumber

# splite the data according to the given axis and value
def splitData(data,axis,value):
    returndata=[]
    for line in data:
        if line[axis] == value:
            #if data is a array ,the type must be the same
            #then the int will be changed into str,the judge will fail
            #so here ues the list is better
            #str(line[axis])==str(value):
            returnVec=line[:axis]
            returnVec.extend(line[axis+1:])
            #returnVec=copy.deepcopy(line)
            returndata.append(returnVec)
    return returndata

# choose the best feature from the data
def chooseBestFeatures(data):
    numberOffea=len(data[0])-1
    baseEntropy=entropy(data)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numberOffea):
        feaValue=[example[i] for example in data]
        feaValue=set(feaValue)
        newEntropy=0.0
        for value in feaValue:
            subData=splitData(data,i,value)
            pro=len(subData)/float(len(data))
            newEntropy=newEntropy+pro*entropy(subData)
        infoGain=baseEntropy-newEntropy
        if infoGain>=bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

# classify
def classify(classList):
    classDict={}
    for value in classList:
        classDict[value]=classDict.get(value,0)+1
    classDict=sorted(classDict.items(),key=lambda x:x[1],reverse=True)
    return classDict[0][0]

# create the tree ,data includes the label in the last column
# label is the features' name list
def createTree(data,labels):
    classList=[example[-1] for example in data]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(data[0])==1:
        return classify(classList)
    bestFea=chooseBestFeatures(data)
    bestFeaLabel=labels[bestFea]
    myTree={bestFeaLabel:{}}
    del(labels[bestFea])
    feaValues=[example[bestFea] for example in data]
    feaValues=set(feaValues)
    for value in feaValues:
        subLabels=labels[:]
        subData=splitData(data,bestFea,value)
        myTree[bestFeaLabel][value]=createTree(subData,subLabels)
    return myTree

# judeg function
def classifyTree(tree,feaLabels,testVec):
    firstFea=list(tree.keys())[0]
    secondDict=tree[firstFea]
    index=feaLabels.index(firstFea)
    for value in secondDict.keys():
        if value==testVec[index]:
            if type(secondDict[value]).__name__=='dict':
                classLabel=classifyTree(secondDict[value],feaLabels,testVec)
            else:
                classLabel=secondDict[value]
    return classLabel

# save the tree into a file
def storeTree(tree,filename):
    f=open(filename,'wb')
    pickle.dump(tree,f,0)
    f.close()

# load the tree from file
def grabTree(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

if __name__== "__main__":
    data = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels=['no surfacing','flippers']
    labels1=copy.deepcopy(labels)
    print(entropy(data))
    print(splitData(data,0,0))
    print(type(chooseBestFeatures(data)))
    classList=[vector[2] for vector in data]
    print(classify(classList))
    tree=createTree(data,labels)
    print(tree)
    print(classifyTree(tree,labels1,[1,1]))
    storeTree(tree,'tree.txt')
    print(grabTree('tree.txt'))

    # test for the data lenses
    with open('D:/研二文件/研二上/机器学习/机器学习实战代码和数据/machinelearninginaction/Ch03/lenses.txt') as f:
        lensesData=[line.strip().split('\t') for line in f.readlines()]
    lensesLabel=['age','prescript','astigmatic','tearRate']
    lensesTree=createTree(lensesData,lensesLabel)
    print(lensesTree)
