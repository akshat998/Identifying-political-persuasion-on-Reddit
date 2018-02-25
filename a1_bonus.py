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
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from scipy import stats
import csv
import timeit
import json


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


def classifers(filename):
    ''' This function performs experiment 3.1

    Parameters
       filename : string, the name of the npz file from Task 2

    '''
    start = timeit.default_timer()

    RandomForestClassifier_data = np.zeros((15, 26))
    MLPClassifier_data = np.zeros((8,26))
    AdaBoostClassifier_data = np.zeros((6,26))
    GaussianNB_data = np.zeros((1,26))
    DecisionTreeClassifier_data = np.zeros((3,26))


    feats = np.load(filename)
    feats = feats[feats.files[0]]  # (40000,174)

    X = feats[..., :-1]  # first 173 element for all 40,000 inputs -> input
    y = feats[..., -1]  # last column of feats -> label
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.20, random_state=1)  # Seed value is one


    # 1. RandomForestClassifier, Nerual Net, performance may be different for each time fo trainning
    max_depth = range(1,16)

    for d in max_depth:

        print('Working on depth '+str(d))
        stop = timeit.default_timer()
        print(stop - start)

        clf = RandomForestClassifier(max_depth = d, n_estimators = 10)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_true = y_test
        c = confusion_matrix(y_true, y_pred)

        print(c)
        acc = accuracy(c)
        rec = recall(c)
        prec = precision(c)

        RandomForestClassifier_data[d-1][0] = d
        RandomForestClassifier_data[d-1][1] = acc
        RandomForestClassifier_data[d-1][2:6] = rec
        RandomForestClassifier_data[d-1][6:10] = prec
        RandomForestClassifier_data[d-1][10:] = c.reshape((1, 16))

        print('accuracy for test is: ' + str(acc))
        print('recall for test is: ' + str(rec))
        print('precision for test is: ' + str(prec))


    # 2. MLPClassifier, Nerual Net, performance may be different for each time fo trainning
    alpha_ist = [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
    for a in alpha_ist:

        print('Working on alpha ' + str(a))
        stop = timeit.default_timer()
        print(stop - start)

        clf = MLPClassifier(alpha = a)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_true = y_test
        c = confusion_matrix(y_true, y_pred)

        print(c)
        acc = accuracy(c)
        rec = recall(c)
        prec = precision(c)

        MLPClassifier_data[alpha_ist.index(a)][0] = a
        MLPClassifier_data[alpha_ist.index(a)][1] = acc
        MLPClassifier_data[alpha_ist.index(a)][2:6] = rec
        MLPClassifier_data[alpha_ist.index(a)][6:10] = prec
        MLPClassifier_data[alpha_ist.index(a)][10:] = c.reshape((1, 16))

        print('accuracy for test is: ' + str(acc))
        print('recall for test is: ' + str(rec))
        print('precision for test is: ' + str(prec))


    # 3. AdaBoostClassifier

    learning_rate_data = [0.1, 0.5, 0.8, 1.0, 1.5, 2.0]
    for r in learning_rate_data:
        print('Working on learning rate ' + str(r))
        stop = timeit.default_timer()
        print(stop - start)

        clf = AdaBoostClassifier(learning_rate = r)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_true = y_test
        c = confusion_matrix(y_true, y_pred)

        print(c)
        acc = accuracy(c)
        rec = recall(c)
        prec = precision(c)

        AdaBoostClassifier_data[learning_rate_data.index(r)][0] = r
        AdaBoostClassifier_data[learning_rate_data.index(r)][1] = acc
        AdaBoostClassifier_data[learning_rate_data.index(r)][2:6] = rec
        AdaBoostClassifier_data[learning_rate_data.index(r)][6:10] = prec
        AdaBoostClassifier_data[learning_rate_data.index(r)][10:] = c.reshape((1, 16))

        print('accuracy for test is: ' + str(acc))
        print('recall for test is: ' + str(rec))
        print('precision for test is: ' + str(prec))

    # 4. GaussianNB
    stop = timeit.default_timer()
    print(stop - start)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_true = y_test
    c = confusion_matrix(y_true, y_pred)

    print(c)
    acc = accuracy(c)
    rec = recall(c)
    prec = precision(c)

    GaussianNB_data[0][0] = 0
    GaussianNB_data[0][1] = acc
    GaussianNB_data[0][2:6] = rec
    GaussianNB_data[0][6:10] = prec
    GaussianNB_data[0][10:] = c.reshape((1, 16))

    print('accuracy for test is: ' + str(acc))
    print('recall for test is: ' + str(rec))
    print('precision for test is: ' + str(prec))

    # 5. DecisionTreeClassifier
    max_features_data = ['sqrt', 'log2', None]
    stop = timeit.default_timer()
    print(stop - start)
    for m in max_features_data:
        clf = DecisionTreeClassifier(random_state=0, max_features = m)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_true = y_test
        c = confusion_matrix(y_true, y_pred)

        print(c)
        acc = accuracy(c)
        rec = recall(c)
        prec = precision(c)

        DecisionTreeClassifier_data[max_features_data.index(m)][0] = max_features_data.index(m)
        DecisionTreeClassifier_data[max_features_data.index(m)][1] = acc
        DecisionTreeClassifier_data[max_features_data.index(m)][2:6] = rec
        DecisionTreeClassifier_data[max_features_data.index(m)][6:10] = prec
        DecisionTreeClassifier_data[max_features_data.index(m)][10:] = c.reshape((1, 16))

        print('accuracy for test is: ' + str(acc))
        print('recall for test is: ' + str(rec))
        print('precision for test is: ' + str(prec))


    # Writing results into a1_bonus.csv file
    with open('./a1_bonus_classifiers.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in RandomForestClassifier_data:
            writer.writerow(line)
        for line in MLPClassifier_data:
            writer.writerow(line)
        for line in AdaBoostClassifier_data:
            writer.writerow(line)
        for line in GaussianNB_data:
            writer.writerow(line)
        for line in DecisionTreeClassifier_data:
            writer.writerow(line)



def adding_features(filename, preproc):
    '''
    :param filename:   an numpy array, output from part 2, stored in feats.npz
    :param preproc:    preprocessed result from part 1
    :return:           updated new feature array with the two updated features
    '''

    # loading array feat.npz
    feats_old = np.load(filename)
    feats_old = feats_old[feats_old.files[0]]  # (40000,174)

    # Loading preproc to add new features into the np.array feats
    data = json.load(open(preproc))
    feats_new = np.zeros((len(data), 173 + 1 + 4))  #added score and controversiality

    for i in range(feats_new.shape[0]):
        feats_new[i][:173] = feats_old[i][:173]
        feats_new[i][173] = data[i]['controversiality']
        feats_new[i][174] = data[i]['score']
        feats_new[i][175] = data[i]['body'].count('clinton')
        feats_new[i][176] = data[i]['body'].count('obama')
        feats_new[i][177] = feats_old[i][173]


    # feats_new is the new numpy array with dimension (40000,176), with the addition of the two features
    # the last column of array feats_new is the category
    return feats_new


def class33(feats):
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

    X = feats[..., :-1]  # first 173 element for all 40,000 inputs -> input
    y = feats[..., -1]  # last column of feats -> label
    # Splitting data into 80% and 20%
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.20, random_state=1)  # Seed value is one
    np.random.seed(0)
    selected_indices = np.random.choice(len(X_train),1000,replace=False)
    X_1k = np.take(X_train, selected_indices, axis=0)
    y_1k = np.take(y_train, selected_indices)
    iBest = 5   # Taken result from previous part, the addition of the result should not be affecting the classifier by to much
    k_list = [5, 10, 20, 30, 40, 50]
    result_1K = []
    result_32K = []


    # 3.3.1
    # Finding the best k for the 1K training set
    # print('1 K data set')
    for v in k_list:
        line = []
        selector = SelectKBest(f_classif, k=v)
        X_new = selector.fit_transform(X_1k, y_1k)
        pp = sorted(selector.pvalues_)

        print(selector.pvalues_[-4:])

        # print(pp)
        line.append(v)
        line += pp[0:v]
        result_1K.append(line)

    for e in result_1K[1][1:6]:
        itemindex = np.where(selector.pvalues_ == e)
        print(itemindex)
    '''
    old:
    (array([16]),)
    (array([0]),)
    (array([149]),)
    (array([128]),)
    (array([21]),)
   
    new: 
    (array([0]),)
    (array([16]),)
    (array([74]),)
    (array([128]),)
    (array([153]),)
    '''

    # Finding the best k for the 32k training set
    # write line 1-6 in a1_3.3.csv,  for each line, write number of k , pk
    # print('32 K data set')
    for v in k_list:
        line = []
        selector = SelectKBest(f_classif, k=v)
        X_new = selector.fit_transform(X_train, y_train)

        pp = sorted(selector.pvalues_)
        print(selector.pvalues_[-4:])


        # print(pp)
        line.append(v)
        line += pp[0:v]
        result_32K.append(line)


    # Finding index of feature that are of most significance
    for e in result_32K[1][1:6]:
        itemindex = np.where(selector.pvalues_ == e)
        print(itemindex)

    '''
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
    with open('./a1_bonus_adding_features.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in result_32K:  # Write the results for 32K data into
            writer.writerow(line)
        writer.writerow([acc_1K, acc_32K])  # On line 7, write  accuracy for 1K, accuracy for 32K



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input file')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument("-f", "--file", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()

    # This function can be run using, python3 a1_bonus.py -i feats.py -f preproc.json
    # It will output two files: a1_bonus_classifiers.csv
    #                           a1_bonus_adding_features.csv
    # both files are annotated, and a1_bonus_classifiers.csv's direct output will be in the same format as in 3.1 and 3.3

    filename = args.input
    preproc = args.file

    classifers(filename)    # Output file 'a1_bonous.csv'

    # Adding new features to feats aray
    feats_new = adding_features(filename, preproc)
    # Test the feats array with additional features on part 3.3
    class33(feats_new)

