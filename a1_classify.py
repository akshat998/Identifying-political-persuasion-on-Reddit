from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import argparse
import sys
import os
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold
from scipy import stats
import csv
import timeit



def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    total = 0.0
    correct = 0.0
    for i in range(C.shape[0]):     # ith row
        for j in range(C.shape[1]):     # jth column
            total += C[i][j]
            if i == j:
                correct += C[i][j]
    if total == 0:
        return 0.0
    return correct / total

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''

    correct = [0,0,0,0]
    count = [0,0,0,0]
    res = [0,0,0,0]
    for i in range(C.shape[0]): # For class i
        for j in range(C.shape[1]):
            count[i] += C[i][j]
            if j == i:
                correct[i] += C[i][j]
    if count[0] != 0:
        res[0] = correct[0]/count[0]
    if count[1] != 0:
        res[1] = correct[1]/count[1]
    if count[2] != 0:
        res[2] = correct[2]/count[2]
    if count[3] != 0:
        res[3] = correct[3]/count[3]
    return res


def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    correct = [0,0,0,0]
    count = [0,0,0,0]
    res = [0,0,0,0]
    for i in range(C.shape[0]): # For class i
        for j in range(C.shape[1]):
            count[j] += C[i][j]
            if j == i:
                correct[j] += C[i][j]
    if count[0] != 0:
        res[0] = correct[0]/count[0]
    if count[1] != 0:
        res[1] = correct[1]/count[1]
    if count[2] != 0:
        res[2] = correct[2]/count[2]
    if count[3] != 0:
        res[3] = correct[3]/count[3]
    return res


def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    print('Section 3.1')
    iBest = 0

    results = []
    csv_data = np.zeros((5, 26))

    feats = np.load(filename)
    feats = feats[feats.files[0]]       #(40000,174)

    X = feats[...,:-1]       # first 173 element for all 40,000 inputs -> input
    y = feats[...,-1]        # last column of feats -> label
    # Splitting data into 80% and 20%
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.20, random_state=1)     # Seed value is one

    # 1. SVC
    clf = SVC(kernel = 'linear',max_iter=10000)
    clf.fit(X_train, y_train)
    y_pred1 = clf.predict(X_test)
    y_true = y_test
    c1 = confusion_matrix(y_true, y_pred1)

    #print(c1)
    acc1 = accuracy(c1)
    rec1 = recall(c1)
    prec1 = precision(c1)

    results.append(acc1)
    csv_data[0][0] = 1
    csv_data[0][1] = acc1
    csv_data[0][2:6] = rec1
    csv_data[0][6:10] = prec1
    csv_data[0][10:] = c1.reshape((1,16))

    #print('accuracy for test 1 is: ' + str(acc1))
    #print('recall for test 1 is: ' + str(rec1))
    #print('precision for test 1 is: ' + str(prec1))

    # 2. SVC radial
    clf = SVC(kernel='rbf', max_iter=10000, gamma=2)  # default is rbf
    clf.fit(X_train, y_train)
    y_pred2 = clf.predict(X_test)
    y_true = y_test
    c2 = confusion_matrix(y_true, y_pred2)
    #print(c2)
    acc2 = accuracy(c2)
    rec2 = recall(c2)
    prec2 = precision(c2)

    results.append(acc2)
    csv_data[1][0] = 2
    csv_data[1][1] = acc2
    csv_data[1][2:6] = rec2
    csv_data[1][6:10] = prec2
    csv_data[1][10:] = c2.reshape((1, 16))

    #print('accuracy for test 2 is: ' + str(acc2))
    #print('recall for test 2 is: ' + str(rec2))
    #print('precision for test 2 is: ' + str(prec2))


    # 3. RandomForestClassifier, Nerual Net, performance may be different for each time fo trainning
    clf = RandomForestClassifier(max_depth=5, n_estimators=10)
    clf.fit(X_train, y_train)

    y_pred3 = clf.predict(X_test)
    y_true = y_test
    c3 = confusion_matrix(y_true, y_pred3)

    #print(c3)
    acc3 = accuracy(c3)
    rec3 = recall(c3)
    prec3 = precision(c3)

    results.append(acc3)
    csv_data[2][0] = 3
    csv_data[2][1] = acc3
    csv_data[2][2:6] = rec3
    csv_data[2][6:10] = prec3
    csv_data[2][10:] = c3.reshape((1, 16))

    #print('accuracy for test 3 is: ' + str(acc3))
    #print('recall for test 3 is: ' + str(rec3))
    #print('precision for test 3 is: ' + str(prec3))


    # 4. MLPClassifier, Nerual Net, performance may be different for each time fo trainning
    clf = MLPClassifier(alpha=0.05)
    clf.fit(X_train, y_train)

    y_pred4 = clf.predict(X_test)
    y_true = y_test
    c4 = confusion_matrix(y_true, y_pred4)

    #print(c4)
    acc4 = accuracy(c4)
    rec4 = recall(c4)
    prec4 = precision(c4)

    results.append(acc4)
    csv_data[3][0] = 4
    csv_data[3][1] = acc4
    csv_data[3][2:6] = rec4
    csv_data[3][6:10] = prec4
    csv_data[3][10:] = c4.reshape((1, 16))

    #print('accuracy for test 4 is: ' + str(acc4))
    #print('recall for test 4 is: ' + str(rec4))
    #print('precision for test 4 is: ' + str(prec4))


    # 5. AdaBoostClassifier
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)

    y_pred5 = clf.predict(X_test)
    y_true = y_test
    c5 = confusion_matrix(y_true, y_pred5)

    #print(c5)
    acc5 = accuracy(c5)
    rec5 = recall(c5)
    prec5 = precision(c5)

    results.append(acc5)
    csv_data[4][0] = 5
    csv_data[4][1] = acc5
    csv_data[4][2:6] = rec5
    csv_data[4][6:10] = prec5
    csv_data[4][10:] = c5.reshape((1, 16))

    #print('accuracy for test 5 is: ' + str(acc5))
    #print('recall for test 5 is: ' + str(rec5))
    #print('precision for test 5 is: ' + str(prec5))

    iBest = results.index(max(results))+1

    # Writing results into a1_3.1.csv file
    with open( './a1_3.1.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in csv_data:
            writer.writerow(line)

    # Adding commentary to the sixth line of this file
    # For SVC method, we give it a maximum iteration to upper bound the calculation. By testing using different values,
    # we can find out that as the number of iteration increases, the overall accuracy increases, until it reaches around
    # 27 percent. When the iteration number is small, there is a sign of overfitting: the precision and recall for
    # predicting type Center and Right is zero, and as the number of iteration increases, the performance increases.
    # However, when the number of iteration increases to 10,000, the same sign of overfitting occurs again, while the
    # overall accuracy stays around 27%.

    return (X_train, X_test, y_train, y_test, iBest)


def class32(X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''

    # 1K, 5K, 10K, 15K, 20K
    size = [1000, 5000, 10000, 15000, 20000]

    if iBest == 1:
        clf = SVC(kernel='linear', max_iter=10000)
    if iBest == 2:
        clf = SVC(kernel='rbf', max_iter=10000, gamma=2)  # default is rdf
    if iBest == 3:
        clf = RandomForestClassifier(max_depth=5, n_estimators=10)
    if iBest == 4:
        clf = MLPClassifier(alpha=0.05)
    if iBest == 5:
        clf = AdaBoostClassifier()

    acc = []
    for i in size:
        train_set = X_train[:i]
        train_label = y_train[:i]
        test_set = X_test[:i]
        test_true = y_test[:i]
        clf.fit(train_set, train_label)
        test_predict = clf.predict(test_set)
        c = confusion_matrix(test_true, test_predict)
        acc.append(accuracy(c))

    # Writing results into a1_3.2.csv file
    with open('./a1_3.2.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(acc)

    # Comment on the 2nd row
    # 1K,   5K,     10K,    15K,    20K
    # 0.389,0.4696,0.48475,0.47775,0.48125
    # The performance is expected to increase as the training set size increases, as it is supposed to be able to pick
    # up more features exist in the training data. However, as the sample size increase from 10K to 15K, the overall
    # performance decreased. This could be noise, or may be a sign of over-fitting, because the model may learned too
    # much specific patterns of the training data. As the size of training set keeps increasing, the performance
    # increased again, this may because the increase of training set size reduced the noise exists in the model.

    X_1k = X_train[:1000]
    y_1k = y_train[:1000]

    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''

    k_list = [5, 10, 20, 30, 40, 50]
    result_1K = []
    result_32K = []

# 3.3.1
    # Finding the best k for the 1K training set
    #print('1 K data set')
    for v in k_list:
        line = []
        selector = SelectKBest(f_classif, k = v)
        X_new = selector.fit_transform(X_1k, y_1k)
        pp = sorted(selector.pvalues_)
        #print(pp)
        line.append(v)
        line += pp[0:v]
        result_1K.append(line)

    for e in result_1K[0][1:6]:
        itemindex = np.where(selector.pvalues_ == e)
        print(itemindex)
    '''
    (array([16]),)
    (array([0]),)
    (array([149]),)
    (array([128]),)
    (array([21]),)
    '''
    # Finding the best k for the 32k training set
    # write line 1-6 in a1_3.3.csv,  for each line, write number of k , pk
    #print('32 K data set')
    for v in k_list:
        line = []
        selector = SelectKBest(f_classif, k = v)
        X_new = selector.fit_transform(X_train, y_train)
        pp = sorted(selector.pvalues_)
        #print(pp)
        line.append(v)
        line += pp[0:v]
        result_32K.append(line)

    '''
    # Finding index of feature that are of most significance
    for e in result_32K[0][1:6]:
        itemindex = np.where(selector.pvalues_ == e)
        print(itemindex)
        
    (array([  0,  16, 163]),)
    (array([  0,  16, 163]),)
    (array([  0,  16, 163]),)
    (array([142]),)
    (array([21]),)
    '''

# 3.3.2
    if iBest == 1:
        clf = SVC(kernel='linear', max_iter=10000)
    if iBest == 2:
        clf = SVC(kernel='rbf', max_iter=10000, gamma=2)  # default is rdf
    if iBest == 3:
        clf = RandomForestClassifier(max_depth=5, n_estimators=10)
    if iBest == 4:
        clf = MLPClassifier(alpha=0.05)
    if iBest == 5:
        clf = AdaBoostClassifier()

    # use the best k=5 features, train 1k
    selector = SelectKBest(f_classif, k=5)
    X_new = selector.fit_transform(X_1k, y_1k)
    X_test_new = selector.transform(X_test)
    clf.fit(X_new, y_1k)
    y_pred1K = clf.predict(X_test_new)
    c_1K = confusion_matrix(y_test, y_pred1K)
    acc_1K = accuracy(c_1K)

    # use the best k=5 features, train 32k
    selector = SelectKBest(f_classif, k=5)
    X_new = selector.fit_transform(X_train, y_train)
    X_test_new = selector.transform(X_test)
    clf.fit(X_new, y_train)
    y_pred32K = clf.predict(X_test_new)
    c_32K = confusion_matrix(y_test, y_pred32K)
    acc_32K = accuracy(c_32K)

    # Writing csv files
    with open('./a1_3.3.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in result_32K:                     # Write the results for 32K data into
            writer.writerow(line)
        writer.writerow([acc_1K, acc_32K])          # On line 7, write  accuracy for 1K, accuracy for 32K


    # 3.3.3
    # (a). Line 8: What features, if any, are chosen at both the low and high(er) amounts of input data? Also
# provide a possible explanation as to why this might be.
    '''
    1 K data set
    (array([16]),)
    (array([0]),)
    (array([149]),)
    (array([128]),)
    (array([21]),)
    32 K data set
    (array([  0,  16, 163]),)
    (array([  0,  16, 163]),)
    (array([  0,  16, 163]),)
    (array([142]),)
    (array([21]),)
    '''

    # (b). Line 9: Are p-values generally higher or lower given more or less data? Why or why not?
    '''
    1 K data set
    [1.0594693216719177e-18, 2.2755949500449372e-13, 2.4012552770811349e-13,...]
    32 K data set
    [0.0, 0.0, 0.0, 1.4143545537221312e-298, 2.2959328207557922e-296, 1.0829095234436538e-295, ...]
    '''

    # (c). Line 10: Name the top 5 features chosen for the 32K training case. Hypothesize as to why those particular
# features might differentiate the classes.
    '''
    1 K data set
    (array([16]),)
    (array([0]),)
    (array([149]),)
    (array([128]),)
    (array([21]),)
    32 K data set
    (array([  0,  16, 163]),)
    (array([  0,  16, 163]),)
    (array([  0,  16, 163]),)
    (array([142]),)
    (array([21]),)
    '''


def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
    '''
    # Set timer
    start = timeit.default_timer()

    kf = KFold(n_splits=5, random_state=1, shuffle=True)

    feats = np.load(filename)
    feats = feats[feats.files[0]]  # (40000,174)

    X = feats[..., :-1]  # first 173 element for all 40,000 inputs -> input
    y = feats[..., -1]  # last column of feats -> label

    output = np.zeros((5,5))

    # Count time
    stop = timeit.default_timer()
    print('Starting the folding')
    print(stop - start)

    f = 0   # counter for fold
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for classifier in range(5):
            print('Now working on classifier ' + str(classifier))
            if classifier == 0:
                clf = SVC(kernel='linear', max_iter=10000)
            if classifier == 1:
                clf = SVC(kernel='rbf', max_iter=10000, gamma = 2)  # default is rdf
            if classifier == 2:
                clf = RandomForestClassifier(max_depth=5, n_estimators=10)
            if classifier == 3:
                clf = MLPClassifier(alpha=0.05)
            if classifier == 4:
                clf = AdaBoostClassifier()

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            c = confusion_matrix(y_test, y_pred)
            output[f][classifier] = accuracy(c)     #adding to the output array

        stop = timeit.default_timer()
        print('Done with ' + str(f+1) + ' fold')
        print(stop - start)
        f += 1

    iBest = i-1
    # h[:,1]    the 2nd coloumn only of np array h

    p_values = []
    for column in range(output.shape[1]):
        if column != iBest:
            S = stats.ttest_rel(output[:,column], output[:,iBest])
            #print(output[:,column])
            #print(output[:, iBest])
            #print(S)
            p_values.append(S[1])

    with open('./a1_3.4.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in output:  # Write the results for 32K data into
            writer.writerow(line)
        writer.writerow(p_values)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input file')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # TODO : complete each classification experiment, in sequence.

    filename = args.input
    #feats = np.load(args.input)         #feats.npz
    #feats = feats[feats.files[0]]       #(40000,174)

    (X_train, X_test, y_train, y_test, iBest) = class31(filename)

    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)

    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)

    class34(filename, iBest)


