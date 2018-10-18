# use KNN to classify handwriting 0-9
# attention! os have the open function it will conflict with the file function open
import numpy as np
import os
# the function transform the 32X32 img txt into the 1X1024 vector
def img2vector(filename):
    vector=np.zeros((1,1024))
    with open(filename) as f:
        for i in range(32):
            line=f.readline()
            for j in range(32):
                vector[0,0+32*i+j]=int(line[j])
    return vector

# the function transform all txt into a array in the dirname and give the list of label
def readtxt(dirname):
    label=[]
    strlist=os.listdir(dirname)
    m=len(strlist)
    data=np.zeros((m,1024))
    for i in range(m):
        filename=strlist[i]
        classint=int(filename.split('_')[0])
        label.append(classint)
        data[i]=img2vector(dirname+'/'+filename)
    return data,label

def kNN(x,data,label,k):
    dataSize=data.shape[0]
    diffMat=np.tile(x,(dataSize,1))-data
    sqMat=diffMat**2
    sqDistance=sqMat.sum(axis=1)
    distance=sqDistance**0.5
    distance=distance.argsort()
    classCount={}
    for i in range(k):
        votalabel=label[distance[i]]
        classCount[votalabel]=classCount.get(votalabel,0)+1
    classCount=sorted(classCount.items(),key=lambda x:x[1],reverse=True)
    return classCount[0][0]

if __name__ == '__main__':
    testData,testLabel=readtxt("D:/研二文件/研二上/机器学习/机器学习实战代码和数据/machinelearninginaction/Ch02/testDigits")
    trainData,trainLabel = readtxt("D:/研二文件/研二上/机器学习/机器学习实战代码和数据/machinelearninginaction/Ch02/trainingDigits")
    errorCount = 0
    for i in range(testData.shape[0]):
        predict=kNN(testData[i],trainData,trainLabel,3)
        if predict != testLabel[i]:
            errorCount=errorCount+1
    print('the total error rate is: %f' %(errorCount/len(testLabel)))
    print(errorCount,len(testLabel))