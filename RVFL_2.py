import numpy as np
import random
import h5py
from function import RVFL_train_val
from option import option as op

dataset_name = "wine_quality_white"

temp = h5py.File("examples/UCI data python/"+dataset_name+"_R.mat")
data = np.array(temp['data']).T

data = data[:, 1:]
dataX = data[:, 0:-1]

dataX_mean = np.mean(dataX, axis=0)
dataX_std = np.std(dataX, axis=0)
dataX = (dataX-dataX_mean)/dataX_std

dataY = data[:, -1]
dataY = np.expand_dims(dataY, 1)

temp = h5py.File("examples/UCI data python/" + dataset_name + "_conxuntos.mat")
index1 = np.array(temp['index1']).astype(np.int32) - 1
index2 = np.array(temp['index2']).astype(np.int32) - 1
index1 = np.squeeze(index1, axis=1)
index2 = np.squeeze(index2, axis=1)
trainX = dataX[index1, :]
trainY = dataY[index1, :]
testX = dataX[index2, :]
testY = dataY[index2, :]



# initialization
# N:hidden neurons
# C:determine the lambda in ridge regression
# S:range of parameter selection
MAX_acc = np.zeros([8, 1])
Best_N = np.zeros([8, 1]).astype(np.int32)
Best_C = np.zeros([4, 1])
Best_S = np.zeros([8, 1])
S = np.linspace(-5, 5, 21)
# op1-dir+sigmoid+ridge
# op2-no dir+sigmoid+ridge
# op3-dir+tribas+ridge
# op4-no dir+tribas+ridge
# op5-dir+sigmoid+moore
# op6-no dir+sigmoid+moore
# op7-dir+tribas+moore
# op8-no dir+tribas+moore
option1 = op()
option2 = op()
option3 = op()
option4 = op()
option5 = op()
option6 = op()
option7 = op()
option8 = op()

# parameter tuning
for s in range(0, S.size):

    for N in range(3, 204, 20):

        for C in range(-5, 15):

            Scale = np.power(2, S[s])

            option1.N = N
            option1.C = 2 ** C
            option1.Scale = Scale
            option1.Scalemode = 3
            option1.link = 1
            option1.ActivationFunction = 'sigmoid'

            option2.N = N
            option2.C = 2 ** C
            option2.Scale = Scale
            option2.Scalemode = 3
            option2.link = 0
            option2.ActivationFunction = 'sigmoid'

            option3.N = N
            option3.C = 2 ** C
            option3.Scale = Scale
            option3.Scalemode = 3
            option3.link = 1
            option3.ActivationFunction = 'tribas'

            option4.N = N
            option4.C = 2 ** C
            option4.Scale = Scale
            option4.Scalemode = 3
            option4.link = 0
            option4.ActivationFunction = 'tribas'

            option5.N = N
            option5.mode = 2
            option5.Scale = Scale
            option5.Scalemode = 3
            option5.link = 1
            option5.ActivationFunction = 'sigmoid'

            option6.N = N
            option6.mode = 2
            option6.Scale = Scale
            option6.Scalemode = 3
            option6.link = 0
            option6.ActivationFunction = 'sigmoid'

            option7.N = N
            option7.mode = 2
            option7.Scale = Scale
            option7.Scalemode = 3
            option7.link = 1
            option7.ActivationFunction = 'tribas'

            option8.N = N
            option8.mode = 2
            option8.Scale = Scale
            option8.Scalemode = 3
            option8.link = 0
            option8.ActivationFunction = 'tribas'

            train_accuracy1, test_accuracy1 = RVFL_train_val(trainX, trainY, testX, testY, option1)
            train_accuracy2, test_accuracy2 = RVFL_train_val(trainX, trainY, testX, testY, option2)
            train_accuracy3, test_accuracy3 = RVFL_train_val(trainX, trainY, testX, testY, option3)
            train_accuracy4, test_accuracy4 = RVFL_train_val(trainX, trainY, testX, testY, option4)
            train_accuracy5, test_accuracy5 = RVFL_train_val(trainX, trainY, testX, testY, option5)
            train_accuracy6, test_accuracy6 = RVFL_train_val(trainX, trainY, testX, testY, option6)
            train_accuracy7, test_accuracy7 = RVFL_train_val(trainX, trainY, testX, testY, option7)
            train_accuracy8, test_accuracy8 = RVFL_train_val(trainX, trainY, testX, testY, option8)

            if test_accuracy1 > MAX_acc[0]:
                # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                MAX_acc[0] = test_accuracy1
                Best_N[0] = N
                Best_C[0] = C
                Best_S[0] = Scale

            if test_accuracy2 > MAX_acc[1]:
                # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                MAX_acc[1] = test_accuracy2
                Best_N[1] = N
                Best_C[1] = C
                Best_S[1] = Scale

            if test_accuracy3 > MAX_acc[2]:
                # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                MAX_acc[2] = test_accuracy3
                Best_N[2] = N
                Best_C[2] = C
                Best_S[2] = Scale

            if test_accuracy4 > MAX_acc[3]:
                # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                MAX_acc[3] = test_accuracy4
                Best_N[3] = N
                Best_C[3] = C
                Best_S[3] = Scale

            if test_accuracy5 > MAX_acc[4]:
                # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                MAX_acc[4] = test_accuracy5
                Best_N[4] = N
                Best_S[4] = Scale

            if test_accuracy6 > MAX_acc[5]:
                # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                MAX_acc[5] = test_accuracy6
                Best_N[5] = N
                Best_S[5] = Scale

            if test_accuracy7 > MAX_acc[6]:
                # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                MAX_acc[6] = test_accuracy7
                Best_N[6] = N
                Best_S[6] = Scale

            if test_accuracy8 > MAX_acc[7]:
                # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                MAX_acc[7] = test_accuracy8
                Best_N[7] = N
                Best_S[7] = Scale

temp = h5py.File("examples/UCI data python/" + dataset_name + "_conxuntos_kfold.mat")
index = []
for i in range(8):
    index_temp = np.array([temp[element[i]][:] for element in temp['index']]).astype(np.int32) - 1
    index_temp = np.squeeze(index_temp, axis=0)
    index_temp = np.squeeze(index_temp, axis=1)
    index.append(index_temp)

ACC_CV = np.zeros([4, 8])

for i in range(4):
    trainX = dataX[index[2 * i], :]
    trainY = dataY[index[2 * i], :]
    testX = dataX[index[2 * i + 1], :]
    testY = dataY[index[2 * i + 1], :]

    option1.N = Best_N[0, 0]
    option1.C = 2 ** Best_C[0, 0]
    option1.Scale = Best_S[0, 0]
    option1.Scalemode = 3
    option1.link = 1
    option1.ActivationFunction = 'sigmoid'

    option2.N = Best_N[1, 0]
    option2.C = 2 ** Best_C[1, 0]
    option2.Scale = Best_S[1, 0]
    option2.Scalemode = 3
    option2.link = 0
    option2.ActivationFunction = 'sigmoid'

    option3.N = Best_N[2, 0]
    option3.C = 2 ** Best_C[2, 0]
    option3.Scale = Best_S[2, 0]
    option3.Scalemode = 3
    option3.link = 1
    option3.ActivationFunction = 'tribas'

    option4.N = Best_N[3, 0]
    option4.C = 2 ** Best_C[3, 0]
    option4.Scale = Best_S[3, 0]
    option4.Scalemode = 3
    option4.link = 0
    option4.ActivationFunction = 'tribas'

    option5.N = Best_N[4, 0]
    option5.mode = 2
    option5.Scale = Best_S[4, 0]
    option5.Scalemode = 3
    option5.link = 1
    option5.ActivationFunction = 'sigmoid'

    option6.N = Best_N[5, 0]
    option6.mode = 2
    option6.Scale = Best_S[5, 0]
    option6.Scalemode = 3
    option6.link = 0
    option6.ActivationFunction = 'sigmoid'

    option7.N = Best_N[6, 0]
    option7.mode = 2
    option7.Scale = Best_S[6, 0]
    option7.Scalemode = 3
    option7.link = 1
    option7.ActivationFunction = 'tribas'

    option8.N = Best_N[7, 0]
    option8.mode = 2
    option8.Scale = Best_S[7, 0]
    option8.Scalemode = 3
    option8.link = 0
    option8.ActivationFunction = 'tribas'

    train_accuracy1, ACC_CV[i, 0] = RVFL_train_val(trainX, trainY, testX, testY, option1)
    train_accuracy2, ACC_CV[i, 1] = RVFL_train_val(trainX, trainY, testX, testY, option2)
    train_accuracy3, ACC_CV[i, 2] = RVFL_train_val(trainX, trainY, testX, testY, option3)
    train_accuracy4, ACC_CV[i, 3] = RVFL_train_val(trainX, trainY, testX, testY, option4)
    train_accuracy5, ACC_CV[i, 4] = RVFL_train_val(trainX, trainY, testX, testY, option5)
    train_accuracy6, ACC_CV[i, 5] = RVFL_train_val(trainX, trainY, testX, testY, option6)
    train_accuracy7, ACC_CV[i, 6] = RVFL_train_val(trainX, trainY, testX, testY, option7)
    train_accuracy8, ACC_CV[i, 7] = RVFL_train_val(trainX, trainY, testX, testY, option8)

print(np.mean(ACC_CV, axis=0))
