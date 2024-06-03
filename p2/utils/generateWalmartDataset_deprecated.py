import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#Returns a numpy array with size nrows x ncolumns-1. nrows and ncolums are the rows and columns of the dataset
#the Date column is skipped (ncolumns-1)
def readData(fname):
    with open(fname) as f:
        fileData = f.read()
  
    lines = fileData.split("\n")
    header = lines[0].split(",")
    lines = lines[1:] 
    #print(header) 
    #print("Data rows: ", len(lines))

    rawData = np.zeros((len(lines), len(header)-1)) #skip the Date column

    for i, aLine in enumerate(lines):       
        splittedLine = aLine.split(",")[:]
        rawData[i, 0] = splittedLine[0]
        rawData[i, 1:] = [float(x) for x in splittedLine[2:]] 

    return rawData


#Returns the train, validation and test data, normalized. It also returns the standard deviation of Weekly_Sales
#Each list has a size equal to the number of stores
#For each store there is a list of size trainNSaples (valNSamples or testNSamples) x nColums-1 (the store id is skipped)
#Columns: Weekly_Sales,Holiday_Flag,Temperature,Fuel_Price,CPI,Unemployment
def splitTrainTestValidation(rawData, valPercent, testPercent):
    listStore = np.unique(rawData[:, 0])
    totalSamples = np.zeros(len(listStore))
    
    for i, storeId in enumerate(listStore):
        totalSamples[i] = np.count_nonzero(rawData[:, 0] == storeId)
    
    trainNSamples = np.floor((1 - testPercent - valPercent) * totalSamples)
    valNSamples = np.floor(valPercent * totalSamples)
    
    tmpTrain = np.zeros((int(np.sum(trainNSamples)), len(rawData[0])))

    store = -1
    counter = 0
    counterTrain = 0
    storeDictTrain = dict(zip(listStore, trainNSamples))
    storeDictVal = dict(zip(listStore, valNSamples))
    
    for i, aLine in enumerate(rawData):
        if store != aLine[0]:
            store = int(aLine[0])
            counter = 0
        if(counter < storeDictTrain.get(store)):
            tmpTrain[counterTrain] = rawData[i][:]
            counterTrain += 1
            counter += 1

    meanData = tmpTrain.mean(axis=0)
    stdData = tmpTrain.std(axis=0)
    rawNormData = (rawData - meanData) / stdData

    allTrain = list()
    allVal = list()
    allTest = list()
    store = -1
    counter = 0
    for i, aLine in enumerate(rawNormData):
        splittedLine = [float(x) for x in aLine[1:]]  # skip store id
        if store != rawData[i][0]:
            if i != 0:
                allTrain.append(storeDataTrain)
                allVal.append(storeDataVal)
                allTest.append(storeDataTest)
            store = int(rawData[i][0])
            storeDataTrain = list()
            storeDataVal = list()
            storeDataTest = list()
            counter = 0

        trainLimit = storeDictTrain.get(store)
        valLimit = trainLimit + storeDictVal.get(store)
        
        if(counter < trainLimit):
            storeDataTrain.append(splittedLine)
        elif(counter < valLimit):
            storeDataVal.append(splittedLine)
        else:
            storeDataTest.append(splittedLine)
        counter += 1

        if i == len(rawNormData)-1:
            allTrain.append(storeDataTrain)
            allVal.append(storeDataVal)
            allTest.append(storeDataTest)

    return allTrain, allVal, allTest, stdData[1]  # std of Weekly_Sales


#generates a time series given the input and ouput data, the sequence length and the batch size
#seqLength is the number of weeks (observations) of data to be used as input
#the target will be the weekly sales in 2 weeks
def generateTimeSeries(data, wSales, seqLength, batchSize):   
    sampling_rate = 1 #keep all the data points 
    weeksInAdvance = 3
    delay = sampling_rate * (seqLength + weeksInAdvance - 1) #the target will be the weekly sales in 2 weeks
    
    dataset = keras.utils.timeseries_dataset_from_array(
        data[:-delay],
        targets=wSales[delay:],
        sampling_rate=sampling_rate,
        sequence_length=seqLength,
        shuffle=True,
        batch_size=batchSize,
        start_index=0)
    
    return dataset


def printTimeSeriesList(theList):
    print('list length', len(theList))
    print('First element')
    input, target = theList[0]
    print([float(x) for x in input.numpy().flatten()], [float(x) for x in target.numpy().flatten()])
    print('Last element')
    input, target = theList[-1]
    print([float(x) for x in input.numpy().flatten()], [float(x) for x in target.numpy().flatten()])


#returns the training and test time series
#it also returns the standard deviation of Weekly_Sales, and the number of input features
def generateTrainTestData(fileName, valPercent, testPercent, seqLength, batchSize):
    rawData = readData(os.path.join(fileName))
    allTrain, allVal, allTest, stdSales = splitTrainTestValidation(rawData, valPercent, testPercent)
    #return allTrain, allVal, allTest
    for i in range(len(allTrain)):
        tmp_train = generateTimeSeries(np.array(allTrain[i]), np.array(allTrain[i])[:,0], seqLength, batchSize)
        tmp_val = generateTimeSeries(np.array(allVal[i]), np.array(allVal[i])[:,0], seqLength, batchSize)
        tmp_test = generateTimeSeries(np.array(allTest[i]), np.array(allTest[i])[:,0], seqLength, batchSize)

        if i == 0:
            train_dataset = tmp_train
            val_dataset = tmp_val
            test_dataset = tmp_test
        else:
            train_dataset = train_dataset.concatenate(tmp_train)
            val_dataset = val_dataset.concatenate(tmp_val)
            test_dataset = test_dataset.concatenate(tmp_test)
    
    return train_dataset, val_dataset, test_dataset, stdSales, np.shape(allTrain)[2]


