# achieve the NaiveBayes algorithm
import numpy as np
import math
import re
from KNN import divideData
# generate the test data
def loadData():
    postingList = [['my', 'dog', 'has', 'flea', \
                    'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', \
                    'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute',
                    'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', \
                    'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList,classVec

# generate the vocabulary list
def createVocList(dataSet):
    vocSet=set([])
    for data in dataSet:
        vocSet=vocSet | set(data)
    return list(vocSet)

# change the words list to a feature vector
def words2Vector(vocList,featureVector):
    returnVec=[0]*len(vocList)
    for word in featureVector:
        if word in vocList:
            returnVec[vocList.index(word)]=1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return returnVec

# change the words matrix to a feature matrix
def data2matrix(vocList,dataSet):
    returnMatrix=[]
    for data in dataSet:
        returnMatrix.append(words2Vector(vocList,data))
    return returnMatrix

#train the NB model,it's different from the real Naive Bayes
def trainNB0(dataSet,labels):
    dataNumber=len(dataSet)
    featureNumber=len(dataSet[0])
    pAubsive=sum(labels)/float(dataNumber)
    p0Num=np.ones(featureNumber)
    p1Num=np.ones(featureNumber)
    p0Total=2.0
    p1Total=2.0
    for i in range(dataNumber):
        if labels[i]==1:
            p1Num+=dataSet[i]
            p1Total+=sum(dataSet[i])
        else:
            p0Num+=dataSet[i]
            p0Total+=sum(dataSet[i])
    p0Vect=np.zeros(featureNumber)
    p1Vect=np.zeros(featureNumber)
    for i in range(featureNumber):
        p0Vect[i]=math.log(p0Num[i]/p0Total)
        p1Vect[i]=math.log(p1Num[i]/p0Total)
    return pAubsive,p0Vect,p1Vect

#train the NB model,it's the real Naive Bayes with Laplace smoothing
def trainNB1(dataSet,labels):
    dataNumber=len(dataSet)
    featureNumber=len(dataSet[0])
    pAubsive=(sum(labels)+1)/float(dataNumber+2)
    p0Num=np.ones((2,featureNumber))
    p1Num=np.ones((2,featureNumber))
    p0Total=2.0
    p1Total=2.0
    for i in range(dataNumber):
        if labels[i]==1:
            p1Num[1]+=dataSet[i]
            p1Total+=1
        else:
            p0Num[1]+=dataSet[i]
            p0Total+=1
    p1Num[0]=p1Total-p1Num[1]
    p0Num[0]=p0Total-p0Num[1]
    p1Num=p1Num/p1Total
    p0Num=p0Num/p0Total
    p0Vect=np.zeros((2,featureNumber))
    p1Vect=np.zeros((2,featureNumber))
    #print(p1Num)
    #print(p0Num)
    for i in range(p0Vect.shape[0]):
        for j in range(p0Vect.shape[1]):
            p0Vect[i][j]=math.log(p0Num[i][j])
            p1Vect[i][j]=math.log(p1Num[i][j])
    return pAubsive,p0Vect,p1Vect

def classifyNB(featureVector,pAb,p0Vec,p1Vec):
    p0=sum(featureVector*p0Vec)+math.log(1.0-pAb)
    p1=sum(featureVector*p1Vec)+math.log(pAb)
    if p0>=p1:
        return 0
    else:
        return 1

def classifyNB1(featureVector,pAb,p0Vec,p1Vec):
    featureMatrx=np.zeros((2,len(featureVector)))
    featureMatrx[1]=np.array(featureVector)
    featureMatrx[0]=1-featureMatrx[1]

    p0=sum(featureMatrx[0]*p0Vec[0])+sum(featureMatrx[1]*p0Vec[1])+math.log(1.0-pAb)
    p1=sum(featureMatrx[0]*p1Vec[0])+sum(featureMatrx[1]*p1Vec[1])+math.log(pAb)
    #print(p0,p1)
    if p0>=p1:
        return 0
    else:
        return 1
# parse the text and return a word list
def textPasrse(text):
    listOfTokens=re.split(r'\W*',text)
    return [word.lower() for word in listOfTokens if len(word)>2]

# test the algorithm using spam data
def spamTest():
    dataSet=[]
    labels=[]
    # loading data
    for i in range(1,26):
        with open('D:/研二文件/研二上/机器学习/机器学习实战代码和数据/machinelearninginaction/Ch04/email/ham/%d.txt' % i,errors='ignore') as f:
            data=f.read()
        dataSet.append(textPasrse(data))
        labels.append(0)
    for i in range(1,26):
        with open('D:/研二文件/研二上/机器学习/机器学习实战代码和数据/machinelearninginaction/Ch04/email/spam/%d.txt' % i,errors='ignore') as f:
            data=f.read()
        dataSet.append(textPasrse(data))
        labels.append(1)
    # generate vocabulary bag
    vocList=createVocList(dataSet)
    # transform the word list to a matrix
    dataMatrix=data2matrix(vocList,dataSet)
    # divide the data into test and trian randomly
    testData,trainData=divideData(np.array(dataMatrix),labels,0.2)
    testVec=testData[0]
    testLab=testData[1]
    # train the model
    pA,p0,p1=trainNB1(trainData[0],trainData[1])
    # calculate the error rate
    error=0
    for i in range(len(testLab)):
        if classifyNB1(testVec[i],pA,p0,p1) != testLab[i]:
           error+=1
    print('error count is %d total count is %d error rate is %f' %(error,len(testLab),(float(error)/len(testLab))))
    return True

if __name__ == '__main__':
    # dataSet,labels=loadData()
    # vocList=createVocList(dataSet)
    # print(dataSet)
    # print(labels)
    # print(vocList)
    # print(dataSet[0])
    # print(words2Vector(vocList,dataSet[0]))
    #
    # dataMatrix=data2matrix(vocList,dataSet)
    # pAb,p0Vec,p1Vec=trainNB1(dataMatrix,labels)
    #
    # print(pAb)
    # print(p0Vec)
    # print(p1Vec)
    #
    # testData1=['love','my','dalmation']
    # testData2=['stupid','garbage']
    # testData1Vec=words2Vector(vocList,testData1)
    # testData2Vec=words2Vector(vocList,testData2)
    # print(classifyNB1(testData1Vec,pAb,p0Vec,p1Vec))
    # print(classifyNB1(testData2Vec, pAb, p0Vec, p1Vec))
    spamTest()
