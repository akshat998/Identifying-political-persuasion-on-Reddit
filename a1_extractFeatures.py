import numpy as np
import sys
import argparse
import os
import json
import string

import csv
import re


#path = '/u/cs401/Wordlists/'
path = '/Users/ouutsuyuki/cdf/csc401/Wordlists/'
file = open(path + "BristolNorms+GilhoolyLogie.csv", "r")
reader = csv.reader(file)
word1 = []
AoA = []
IMG = []
FAM = []
for line in reader:
    word1.append(line[1])
    AoA.append(line[3])
    IMG.append(line[4])
    FAM.append(line[5])

file = open(path + "Ratings_Warriner_et_al.csv", "r")
reader = csv.reader(file)
word2 = []
V = []
A = []
D = []
for line in reader:
    word2.append(line[1])
    V.append(line[2])
    A.append(line[5])
    D.append(line[8])

#np.mean(list)
#numpy.std(a,ddof = 1)
#["foo", "bar", "baz"].index("bar")

def extract1( comment ):

    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''

    # Make word_lists and groups

    # Creating array feats: 1*173
    feats = np.zeros((1,173))

    body = re.compile("([\w]+|[\W]+)/(?=[\w]+|[\W]+)").findall(comment)    # left word
    lemma = re.compile("(?=[\w]+|[\W]+)/([\w]+|[\W]+)").findall(comment)  # right side
    #1. Number of First-person pronouns  -PRP
    # I, me, my, mine, we, us, our, ours
    l = ['i','me','my','mine','we','us','our','ours']
    res = re.compile(r'\b(' + r'|'.join(l) + r')\b').findall(comment)
    feats[0][0] = len(res)

    #2. Number of second-person pronouns
    # you, your, yours, u, ur, urs
    l = ['you', 'your', 'yours', 'u', 'ur', 'urs']
    res = re.compile(r'\b(' + r'|'.join(l) + r')\b').findall(comment)
    feats[0][1] = len(res)

    #3. Number of third-person pronouns
    # he, him, his, she, her, hers, it, its, they, them, their, theirs
    l =['he', 'him', 'his', 'she','her','hers','it','its','they','them','their','theirs']
    res = re.compile(r'\b(' + r'|'.join(l) + r')\b').findall(comment)
    feats[0][2] = len(res)

    #4. Number of coordinating conjunctions
    feats[0][3] = lemma.count('CC')

    #5. Number of past-tense verbs
    feats[0][4] = lemma.count('VBD')

    #6. Number of future-tense verbs
    l = ['\'ll', 'will', 'gonna']
    res = re.compile(r'\b(' + r'|'.join(l) + r')\b').findall(comment)
    feats[0][5] += len(res)
    res = re.compile(r"go/VBG to/TO [\w]+/VB").findall(comment)
    feats[0][5] += len(res)

    #7. Number of commas
    left = re.compile("(?=/)[\S]+").sub('', comment)
    feats[0][6] = left.count(',')

    #8. Number of multi-character punctuation tokens
    for e in body:
        if e[0] in string.punctuation and len(e)>1:
            feats[0][7] += 1

    #9. Number of common nouns
    feats[0][8] = lemma.count('NN') + lemma.count('NNS')

    #10. Number of proper nouns
    feats[0][9] = lemma.count('NNP') + lemma.count('NNPS')

    #11. Number of adverbs
    feats[0][10] += lemma.count('RB') + lemma.count('RBR') + lemma.count('RBS')

    #12. Number of wh- words
    feats[0][11] += lemma.count('WDT') +  lemma.count('WP') + lemma.count('WP$') + lemma.count('WRB')

    #13. Number of slang acronyms
    slang = ['smh','fwb','lmfao', 'lms','tbh','ro','wtf', 'bff', 'wyd','lylc','brb','atm','imao','sml','btw','bw',
             'imho', 'fyi','ppl','sob','ttyl','imo','ltr','thx','kk','omg','omfg','ttys','afn', 'bbs','cya','ez','f2f',
             'gtr','ic','jk','k','ly','ya','nm','np','plz','ru','so','tc','tmi','ym','ur','u','sol','gml']
    for e in body:
        if e in slang:
            feats[0][12] += 1

    #14. Number of words in uppercase (3 letters long)
    for e in body:
        if e.isupper and len(e)>3:
            feats[0][13] += 1

    #15. Average length of sentences, in tokens
    sen = comment.count('\n')+1
    feats[0][14] = len(body)/sen

    #16. Average length of tokens, excluding punctuation-only tokens, in characters
    n = 0
    t = 0
    for e in body:
        if e[0] not in string.punctuation:
            n += 1
            t += len(e)
    if n != 0:
        feats[0][15] = t/n

    #17. Number of sentences.
    feats[0][16] = comment.count('\n') + 1

    # 18-23 Average and std of AoA, IMG FAM
    # np.mean(list)
    # numpy.std(a,ddof = 1)
    # ["foo", "bar", "baz"].index("bar")
    sAoA = []
    sIMG = []
    sFAM = []
    for e in body:
        if e in word1:
            i = word1.index(e)
            sAoA.append(int(AoA[i]))
            sIMG.append(int(IMG[i]))
            sFAM.append(int(FAM[i]))
    if len(sAoA) >0:
        feats[0][17] = np.mean(sAoA)
        feats[0][20] = np.std(sAoA)
    #if len(sIMG) > 0:
        feats[0][18] = np.mean(sIMG)
        feats[0][21] = np.std(sIMG)
    #if len(sFAM) > 0:
        feats[0][19] = np.mean(sFAM)
        feats[0][22] = np.std(sFAM)

    # 24-29 Average and std of V, D, A
    sV = []
    sD = []
    sA = []
    for e in body:
        if e in word2:
            i = word2.index(e)
            sV.append(float(V[i]))
            sD.append(float(D[i]))
            sA.append(float(A[i]))
    if len(sV)>0:
        feats[0][23] = np.mean(sV)
        feats[0][26] = np.std(sV)
    if len(sD) > 0:
        feats[0][24] = np.mean(sD)
        feats[0][27] = np.std(sD)
    if len(sA) > 0:
        feats[0][25] = np.mean(sA)
        feats[0][28] = np.std(sA)

    return feats

def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))
    #print(feats.shape)
    # TODO: your code here2212

    # Read all ID files and creat a list of each ID list for each Alt, Right, Left and Center
    #path = '/u/cs401/A1/feats/'
    path = '/Users/ouutsuyuki/cdf/csc401/feats/'

    Alt_data = np.load(path+'Alt_feats.dat.npy')        # <class 'numpy.ndarray'>   (200272, 144)
    Alt_ID = open(path+'Alt_IDs.txt','r').read().split('\n')        # List of IDs

    Right_data = np.load(path + 'Right_feats.dat.npy')  # <class 'numpy.ndarray'>   (200272, 144)
    Right_ID = open(path + 'Right_IDs.txt', 'r').read().split('\n')  # List of IDs

    Left_data = np.load(path + 'Left_feats.dat.npy')  # <class 'numpy.ndarray'>   (200272, 144)
    Left_ID = open(path + 'Left_IDs.txt', 'r').read().split('\n')  # List of IDs

    Center_data = np.load(path + 'Center_feats.dat.npy')  # <class 'numpy.ndarray'>   (200272, 144)
    Center_ID = open(path + 'Center_IDs.txt', 'r').read().split('\n')  # List of IDs

    # for each data[i], use Alt_ID.index(data[i]['id'])

    # main, call extract1 on each datum, and add the results (+ the class) to the feats array.
    for i in range(feats.shape[0]):
        feats[i][:-1] = extract1(data[i]['body'])
        if data[i]['cat'] == 'Left':
            feats[i][-1] = 0
            index = Left_ID.index(data[i]['id'])
            feats[i][29:-1] = Left_data[index][:]

        if data[i]['cat'] == 'Center':
            feats[i][-1] = 1
            index = Center_ID.index(data[i]['id'])
            feats[i][29:-1] = Center_data[index][:]

        if data[i]['cat'] == 'Right':
            feats[i][-1] = 2
            index = Right_ID.index(data[i]['id'])
            feats[i][29:-1] = Right_data[index][:]

        if data[i]['cat'] == 'Alt':
            feats[i][-1] = 3
            index = Alt_ID.index(data[i]['id'])
            feats[i][29:-1] = Alt_data[index][:]

        if i%100 == 0:
            print(i)

    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

