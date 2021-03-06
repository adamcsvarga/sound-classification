# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:26:45 2015

Main script for running sound recognition experiments.

@author: vurga
"""

from random import randint
import extract_features as ef, numpy as np, train_models as tm, pandas as pd
import os

def get_file_list(dirname):
    filenames = []
    for f in os.listdir(dirname):
        if f.endswith('.wav'):
            filenames.append(f)
    
    return filenames
    
def get_labels(filenames):
    labels, infos = [], []
    for filename in filenames:
        label = filename.split('-')[-1].split('.')[0]
        labels.append(label)
        age, gender = filename.split('-')[-2], filename.split('-')[-3]
        infos.append([age, gender])
    
    # convert to categorical integers
    l = pd.Categorical(pd.Series(labels))
    labels = l.labels
    
    return labels, infos

def create_training_data(filenames, labels, trainlist):
    with open(trainlist, 'r') as source:
        trainlines = source.readlines()
        
    train_files, train_labels = [], []
    test_files , test_labels = filenames, labels
    for trainline in trainlines:
        try:
            index = filenames.index(trainline[:-1])
            train_files.append(filenames[index])
            train_labels.append(labels[index])
            del test_files[index]
            del test_labels[index]
        except:
            pass
    
    return train_files, train_labels, test_files, test_labels
    

def split_data(filenames, train_prop=0.7):
    with open('train.list', 'w'):
        pass
    with open('test.list', 'w'):
        pass
    
    full_length = len(filenames)
    while len(filenames) > full_length * (1 - train_prop):
        try:
            index = randint(0, len(filenames))
            with open('train.list', 'a+') as target:
                target.write(filenames[index] + '\n')
            del filenames[index]
        except:
            pass
        
def extract(trainfiles, testfiles, trainlabels, testlabels):
    train_mfcc, test_mfcc = [], []
    
    for trainfile in trainfiles:
        m = ef.get_mfcc('wav_samples/' + trainfile, num_coeffs=26, \
        deltas=True, ddeltas=True)
        a = ef.compute_avg(m, num_coeffs=26, deltas=True, ddeltas=True)
        train_mfcc.append(a)
    
    for testfile in testfiles:
        m = ef.get_mfcc('wav_samples/' + testfile, num_coeffs=26, \
        deltas=True, ddeltas=True)
        a = ef.compute_avg(m, num_coeffs=26, deltas=True, ddeltas=True)
        test_mfcc.append(a)
      
    np.save('train.mfcc', np.asarray(train_mfcc))
    np.save('test.mfcc', np.asarray(test_mfcc))
    np.save('train.labels', np.asarray(trainlabels))
    np.save('test.labels', np.asarray(testlabels))
    return train_mfcc, test_mfcc
    
def nodk(trainset, testset, trainlabels, testlabels):
    trainclean = []
    trainlabclean = []
    testclean = []
    testlabclean = []
    
    print(len(trainset), len(trainlabels))
    print(len(testset), len(testlabels))
    for i in range(0, len(trainset)):
        if trainlabels[i] != 1:
            trainclean.append(trainset[i])
            trainlabclean.append(trainlabels[i])
    for i in range(0, len(testset)):
        if testlabels[i] != 1:
            testclean.append(testset[i])
            testlabclean.append(testlabels[i])
            
            
    return np.asarray(trainclean), np.asarray(testclean),\
    np.asarray(trainlabclean), np.asarray(testlabclean)
    
if __name__ == '__main__':
    # get file list & info from filenames
    filelist = get_file_list('wav_samples')
    target_labels, extra_info = get_labels(filelist)
    
    # Train - test splitting
    #split_data(filelist)
    train, train_targets, test, test_targets = \
    create_training_data(filelist, target_labels, 'train.list')
    
    #feature extraction (saving to files)    
    train_set, test_set = extract(train, test, train_targets, test_targets)
 
    # load data
    train_set = np.load('train.mfcc.npy')
    train_labels = np.load('train.labels.npy')
    train_infos = np.load('train.infos.npy')
    test_set = np.load('test.mfcc.npy')
    test_labels = np.load('test.labels.npy')
    test_infos = np.load('test.infos.npy')
    
    # linear regression
    lpreds = tm.linreg(train_set, train_labels, test_set)
    print(tm.eval(lpreds, test_labels))
    
    # ridge regression
    preds = tm.ridgereg(train_set, train_labels, test_set)
    print(tm.eval(preds, test_labels))
    
    # lasso
    preds = tm.lasso(train_set, train_labels, test_set)
    print(tm.eval(preds, test_labels))
    
    # lars lasso
    preds = tm.larslasso(train_set, train_labels, test_set)
    print(tm.eval(preds, test_labels))
      
    # bayesian ridge regression
    preds = tm.bayesridgereg(train_set, train_labels, test_set)
    print(tm.eval(preds, test_labels))
    
    # polynomial lars lasso
    preds = tm.polyreg(train_set, train_labels, test_set)
    print(tm.eval(preds, test_labels))
    
    # SVM
    preds = tm.svmtrain(train_set, train_labels, test_set)
    print(tm.eval(preds, test_labels))
    
    # stochastic gradient descent
    preds = tm.sgd(train_set, train_labels, test_set)
    print(tm.eval(preds, test_labels))
    
    # nearest centroid
    preds = tm.ncentr(train_set, train_labels, test_set)
    print(tm.eval(preds, test_labels))
    
    # Gaussian process
    preds = tm.gauss(train_set, train_labels, test_set)
    print(tm.eval(preds, test_labels))
    
    # Gaussian Naive Bayes
    preds = tm.nb(train_set, train_labels, test_set)
    print(tm.eval(preds, test_labels))
    
    # decision tree
    preds = tm.dt(train_set, train_labels, test_set)
    print(tm.eval(preds, test_labels))
    
    # LDA + SVM
    train_set_lda, test_set_lda = tm.lda(train_set, train_labels, test_set)
    preds = tm.svmtrain(train_set_lda, train_labels, test_set_lda)
    print(train_labels)
    print(preds, test_labels)
    print(tm.eval(preds, test_labels))
    print(test_set)
    
    # append extra infos (age, gender)
    # age
    train_set_age = np.column_stack((train_set, train_infos[:,0]))
    test_set_age = np.column_stack((test_set, test_infos[:,0]))
    train_set_age_lda, test_set_age_lda = tm.lda(train_set_age,\
    train_labels, test_set_age)
    preds = tm.svmtrain(train_set_age_lda, train_labels, test_set_age_lda)
    print(tm.eval(preds, test_labels))
    # gender
    train_set_gnd = np.column_stack((train_set, train_infos[:,1]))
    test_set_gnd = np.column_stack((test_set, test_infos[:,1]))
    train_set_gnd_lda, test_set_gnd_lda = tm.lda(train_set_gnd,\
    train_labels, test_set_gnd)
    preds = tm.svmtrain(train_set_gnd_lda, train_labels, test_set_gnd_lda)
    print(tm.eval(preds, test_labels))
    
    # remove 'don't know'
    train_set_clean, test_set_clean, train_labels_clean,\
    test_labels_clean = nodk(train_set, test_set, train_labels, test_labels)
    train_set_clean_lda, test_set_clean_lda = tm.lda(train_set_clean,\
    train_labels_clean, test_set_clean)
    preds = tm.svmtrain(train_set_clean_lda, train_labels_clean,\
    test_set_clean_lda)
    print(tm.eval(preds, test_labels_clean))
    print(len(test_set), len(test_set_clean))
