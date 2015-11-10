# -*- coding: cp936 -*-
# 1：约会网站
from numpy import *
import operator
from os import listdir

#创造数据集
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#第一个kNN分类器  inX-测试数据 dataSet-样本数据  labels-标签 k-邻近的k个样本
def classify0(inX,dataSet, labels, k):
    #计算距离
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1))- dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances **0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    #选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #排序
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]


# 将文本记录到转换numPy的解析程序
def file2matrix(filename):
    #打开文件并得到文件行数
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    #创建返回的numPy矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index =0
    #解析文件数据到列表
    for line in arrayOLines:
        line = line.strip()
        listFormLine = line.split('\t')
        returnMat[index,:] = listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index += 1
    return returnMat, classLabelVector

#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0);
    maxVals = dataSet.max(0);
    ranges = maxVals - minVals;
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals


#测试代码
def datingClassTest():
    hoRatio = 0.10    #测试数据占的百分比
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print 'the classifier came back with: %d, the real answer is: %d' %(classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f " % (errorCount/float(numTestVecs))
    

#输入某人的信息，便得出对对方喜欢程度的预测值
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels,3)
    print 'You will probably like this person: ', resultList[classifierResult - 1]
    

# 2：手写识别系统
#将一个32*32的二进制图像矩阵转换成1*1024的向量

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


#手写识别系统测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')   #获取目录内容
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]              #分割得到标签  从文件名解析得到分类数据
        fileStr = fileNameStr.split('.')[0]
        classStr = int(fileStr.split('_')[0])
        hwLabels.append(classStr)                 #测试样例标签
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print 'the classifier came back with: %d, the real answer is: %d' % (classifierResult, classStr)
        if(classifierResult != classStr): errorCount += 1.0
    print "\nthe total numbers of errors is : %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
        



