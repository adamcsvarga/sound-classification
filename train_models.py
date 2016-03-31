# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:43:56 2015

Training and evaluating different models

@author: vurga
"""

from sklearn import linear_model, svm, gaussian_process, tree, hmm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
import numpy as np, math

def eval(hyps, reals):
    correct = 0.0
    for i in range(0, len(hyps)):
        if math.ceil(hyps[i]) == reals[i]:
            correct += 1.0
    
    return correct / len(hyps)

def linreg(train, labels, test):
    clf = linear_model.LinearRegression()
    clf.fit(train, labels)
    return clf.predict(test)
    
def ridgereg(train, labels, test):
    clf = linear_model.Ridge(alpha=0.5)
    clf.fit(train, labels)
    return clf.predict(test)
    
def lasso(train, labels, test):
    clf = linear_model.Lasso(alpha=0.1, max_iter=10000)
    clf.fit(train, labels)
    return clf.predict(test)
    
def larslasso(train, labels, test):
    clf = linear_model.LassoLars(alpha=0.1)
    clf.fit(train, labels)
    return clf.predict(test)
    
def bayesridgereg(train, labels, test):
    clf = linear_model.BayesianRidge()
    clf.fit(train, labels)
    return clf.predict(test)
    
def polyreg(train, labels, test):
    poly = PolynomialFeatures(degree=2)
    transformed_train = poly.fit_transform(train)
    transformed_test = poly.fit_transform(test)
    return larslasso(transformed_train, labels, transformed_test)
    
def svmtrain(train, labels, test):
    clf = svm.SVC()
    clf.fit(train, labels)
    return clf.predict(test)
    
def sgd(train, labels, test):
    clf = linear_model.SGDClassifier(loss='hinge', penalty='l2')
    clf.fit(train, labels)
    return clf.predict(test)
    
def ncentr(train, labels, test):
    clf = NearestCentroid()
    clf.fit(train, labels)
    return clf.predict(test)
    
def gauss(train, labels, test):
    gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4,\
    thetaU=1e-1)
    gp.fit(train, labels)
    return gp.predict(test)

def nb(train, labels, test):
    gnb = GaussianNB()
    return gnb.fit(train, labels).predict(test)
    
def dt(train, labels, test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(train, labels)
    return clf.predict(test)
    
def lda(train, labels, test):
    lda = LDA(n_components=7)
    return lda.fit(train, labels).transform(train),\
    lda.fit(train, labels).transform(test)
    
def hmmtrain(train, labels, test):
    model = hmm.GaussianHMM(1, 'full')
    model.fit([train])
    return model.predict(test)