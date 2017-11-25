#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:32:19 2017

@author: eric.hensleyibm.com
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier

def lin_svc():
    pipe = Pipeline([
        ('a_preprocess', RobustScaler()),
        ('b_reduce', PCA(iterated_power=7, random_state = 86, n_components = 1)),
        ('c_classify', SVC(kernel = 'linear', probability = True, random_state = 46, C = .04))
    ])
    return ('SVC-lin', pipe)

def rbf_svc():
    pipe = Pipeline([
        ('a_preprocess', RobustScaler()),
        ('b_reduce', PCA(iterated_power=7, random_state = 86, n_components = 3)),
        ('c_classify', SVC(kernel = 'rbf', probability = True, random_state = 46, C = 1, gamma =1))
    ])
    return ('SVC-rbf', pipe)

def poly_svc():
    pipe = Pipeline([
        ('a_preprocess', StandardScaler()),
        ('b_reduce', PCA(iterated_power=7, random_state = 86, n_components = 2)),
        ('c_classify', SVC(kernel = 'poly', random_state = 46, degree = 2, C =.4, gamma = .05, probability = True))
    ])
    return ('SVC-poly', pipe)

def GausProc():
    pipe = Pipeline([
        ('a_preprocess', RobustScaler()),
        ('b_reduce', PCA(iterated_power=7, random_state = 86, n_components = 1)),
        ('c_classify', GaussianProcessClassifier(1.0 * RBF(1.0)))
    ])
    return ('Gauss', pipe)

def light_gbc():
    pipe = Pipeline([
        ('a_preprocess', StandardScaler()),
        ('b_reduce', PCA(iterated_power=7, random_state = 86, n_components = 11)),
        ('c_classify', lgb.LGBMClassifier(random_state = 46, learning_rate = .243, n_estimators = 100, num_leaves = 18, max_depth = 10))
    ])
    return ('LGBC', pipe)

def knn():
    pipe = Pipeline([
        ('a_preprocess', MinMaxScaler()),
        ('b_reduce', PCA(iterated_power=7, random_state = 86, n_components = 3)),
        ('c_classify', KNeighborsClassifier(weights='distance', p=1, n_neighbors = 95))
    ])
    return ('KNN', pipe)

def QDA():
    pipe = Pipeline([
        ('a_preprocess', StandardScaler()),
        ('b_reduce', PCA(iterated_power=7, random_state = 86, n_components = 10)),
        ('c_classify', QuadraticDiscriminantAnalysis())
    ])
    return ('QDA', pipe)

def naive_bayes():
    pipe = Pipeline([
        ('a_preprocess', StandardScaler()),
        ('b_reduce', PCA(iterated_power=7, random_state = 86, n_components = 12)),
        ('c_classify', GaussianNB())
    ])
    return ('naiive bayes', pipe)

def RandomForrest():
    pipe = Pipeline([
        ('a_preprocess', StandardScaler()),
        ('b_reduce', PCA(iterated_power=7, random_state = 86, n_components = 9)),
        ('c_classify', RandomForestClassifier(random_state = 46, max_features = None, max_depth = 15, min_samples_split = 45))
    ])
    return ('Random Forrest', pipe)

def allmodels():
    models = [lin_svc(), rbf_svc(), poly_svc(), GausProc(), light_gbc(), knn(), QDA(), naive_bayes(), RandomForrest()]
    return models

def tuned_ensemble():
    model = VotingClassifier(estimators = [poly_svc(), light_gbc(), knn(), QDA(), naive_bayes(), RandomForrest()], weights = [ 9.52981016,  0.14208127,  9.47070708,  9.33068956,  6.89142325, 1.09667858], voting = 'soft')
    return model