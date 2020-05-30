import glob
import math
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from copy import deepcopy
import logging
import scipy.optimize as opt
import os
from .screener import load_data_all, load_data_single
from .. import arguments
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error



class MLP(object):

    def __init__(self, args, Xin, Xout):
        self.args = args
        self.Xin = Xin
        self.Xout = Xout
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.Xin, self.Xout, test_size=0.1, random_state=1)

        self.hyper_parameters = {'hidden_layer_sizes':[(64,), (128,), (256,)], 
                                 'alpha': [1e-3, 1e-4, 1e-5], 
                                 'learning_rate_init':[1e-2, 1e-3, 1e-4]}

        # self.hyper_parameters = {'hidden_layer_sizes':[(128,)], 
        #                          'alpha': [1e-4], 
        #                          'learning_rate_init':[1e-3]}

        self.model = MLPRegressor(activation='logistic', max_iter=200, random_state=1, 
                                  tol=1e-5, verbose=False, n_iter_no_change=100)
    

    def cross_validation(self):
        self.Grid = GridSearchCV(self.model, self.hyper_parameters, cv=5, scoring='neg_mean_squared_error', verbose=1)
        self.Grid.fit(self.X_train, self.y_train) 
        print ("Best hyper params", self.Grid.best_params_)
        print ("Best score = {}".format(self.Grid.best_score_))
        print("Summary", self.Grid.cv_results_)
        np.save('plots/new_data/numpy/hyper/best_hyper.npy', self.Grid.best_params_)
        np.save('plots/new_data/numpy/hyper/summary.npy', self.Grid.cv_results_) 


    def model_fit(self):
        best_params = np.load('plots/new_data/numpy/hyper/best_hyper.npy', allow_pickle='TRUE').item()
        self.model.set_params(**best_params)
        self.model.set_params(verbose=True)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)

        MSE_test = mean_squared_error(self.y_test, y_pred)
        print(MSE_test)

        np.save('saved_weights/coefs.npy', self.model.coefs_)
        np.save('saved_weights/intercepts.npy', self.model.intercepts_)
        print(self.model.coefs_)
        print(self.model.intercepts_)
        print(self.model.coefs_[0].shape)
        print(self.model.coefs_[1].shape)
        print(self.model.intercepts_[0].shape)
        print(self.model.intercepts_[1].shape)


    def model_exp(self):
        regr = MLPRegressor(hidden_layer_sizes=(256,), activation='logistic', solver='adam', alpha=0,
                            batch_size=32, learning_rate_init=1e-2, max_iter=1000, random_state=1, 
                            tol=1e-9, verbose=True, n_iter_no_change=100).fit(self.X_train, self.y_train)

        y_pred = regr.predict(self.X_test)
        MSE_test = mean_squared_error(self.y_test, y_pred)
        print(MSE_test)
        np.save('saved_weights/coefs.npy', regr.coefs_)
        np.save('saved_weights/intercepts.npy', regr.intercepts_)


def polynomial_regression(args):
    Xin, Xout = load_data_all(args, rm_dup=False, middle=False)
    trainer = Trainer(args, Xin, Xout)
    degrees = np.arange(1, 10, 1)
    train_MSE_tosave = []
    test_MSE_tosave = []
    poly_degree = []
    for d in degrees:
        train_MSE, test_MSE = trainer.polynomial_regression(d)
        train_MSE_tosave.append(train_MSE)
        test_MSE_tosave.append(test_MSE)
        poly_degree.append(d)
    np.save('plots/new_data/numpy/polynomial/train_MSE.npy',
            np.asarray(train_MSE_tosave))
    np.save('plots/new_data/numpy/polynomial/test_MSE.npy',
            np.asarray(test_MSE_tosave))
    np.save('plots/new_data/numpy/polynomial/poly_degree.npy',
            np.asarray(poly_degree))


def training(args):
    Xin, Xout = load_data_all(args, rm_dup=False, middle=False)
    trainer = Trainer(args, Xin, Xout)
    trainer.shuffle_data(K=10, test_ratio=0.)
    error = trainer.train(k=1, hidden_dim=256, lr=1e-2, bsize=32)


def run(args):
    Xin =  np.load('saved_data_sobol/Xin.npy')
    Xout =  np.load('saved_data_sobol/Xout.npy')
    mlp_model = MLP(args, Xin, Xout)
    mlp_model.model_exp()


if __name__ == '__main__':
    args = arguments.args
    run(args)
