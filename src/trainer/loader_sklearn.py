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
from sklearn.linear_model import LinearRegression


class MLP(object):

    def __init__(self, args, Xin, Xout):
        self.args = args
        X_mean = Xin.mean(0)
        Xin = Xin - X_mean
        np.save('saved_weights/mean.npy', X_mean)
        self.Xin = Xin
        self.Xout = Xout
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.Xin, self.Xout, test_size=0.1, random_state=1)

        self.hyper_parameters = {'hidden_layer_sizes': [(64,), (128,), (256,)],
                                 'alpha': [1e-3, 1e-4, 1e-5],
                                 'learning_rate_init': [1e-2, 1e-3, 1e-4]}

        # self.hyper_parameters = {'hidden_layer_sizes':[(128,)],
        #                          'alpha': [1e-4],
        #                          'learning_rate_init':[1e-3]}

        self.model = MLPRegressor(activation='logistic', max_iter=200, random_state=1,
                                  tol=1e-5, verbose=False, n_iter_no_change=100)

    def cross_validation(self):
        self.Grid = GridSearchCV(
            self.model, self.hyper_parameters, cv=5, scoring='neg_mean_squared_error', verbose=1)
        self.Grid.fit(self.X_train, self.y_train)
        print("Best hyper params", self.Grid.best_params_)
        print("Best score = {}".format(self.Grid.best_score_))
        print("Summary", self.Grid.cv_results_)
        np.save('plots/new_data/numpy/hyper/best_hyper.npy',
                self.Grid.best_params_)
        np.save('plots/new_data/numpy/hyper/summary.npy', self.Grid.cv_results_)

    def model_fit(self):
        best_params = np.load(
            'plots/new_data/numpy/hyper/best_hyper.npy', allow_pickle='TRUE').item()
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
                            batch_size=32, learning_rate_init=1e-2, max_iter=2000, random_state=1,
                            tol=1e-9, verbose=True, n_iter_no_change=1000).fit(self.X_train, self.y_train)

        y_pred = regr.predict(self.X_test)
        MSE_test = mean_squared_error(self.y_test, y_pred)
        print(MSE_test)
        np.save('saved_weights/coefs.npy', regr.coefs_)
        np.save('saved_weights/intercepts.npy', regr.intercepts_)
        # plt.plot(regr.loss_curve_)
        # plt.show()


class PolyReg(object):

    def __init__(self, args, Xin, Xout):
        self.args = args
        Xin = Xin - Xin.mean(0)
        self.Xin = Xin
        self.Xout = Xout
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.Xin, self.Xout, test_size=0.1, random_state=1)

    def polynomial_features(self, degree):
        category_training = self.X_train[:, -1:]
        category_test = self.X_test[:, -1:]
        transform_training = self.X_train[:, :-1]
        transform_test = self.X_test[:, :-1]

        poly = PolynomialFeatures(degree)
        transform_training = poly.fit_transform(transform_training)
        transform_test = poly.fit_transform(transform_test)
        self.X_training_poly = np.concatenate(
            (transform_training, category_training), axis=1)
        self.X_test_poly = np.concatenate(
            (transform_test, category_test), axis=1)
        self.degree = degree

    def linear_regression_sklearn(self):
        regr = LinearRegression(fit_intercept=False).fit(self.X_training_poly, self.y_train)
        y_pred_train = regr.predict(self.X_training_poly)
        y_pred_test = regr.predict(self.X_test_poly)
        MSE_train = mean_squared_error(self.y_train, y_pred_train)
        MSE_test = mean_squared_error(self.y_test, y_pred_test)
        print("Sklern polynomial regression degree {} training MSE {}, test MSE {}\n".format(
            self.degree, MSE_train, MSE_test))

        return MSE_train, MSE_test

    def linear_regression_custom(self):
        print(np.linalg.matrix_rank(self.X_training_poly))
        print(np.linalg.matrix_rank(
            np.matmul(self.X_training_poly.transpose(), self.X_training_poly)))
        print(self.X_training_poly.shape[1])

        # Analytical form
        tmp = np.linalg.inv(np.matmul(self.X_training_poly.transpose(), self.X_training_poly))
        w = np.matmul(np.matmul(tmp, self.X_training_poly.transpose()),
                      np.expand_dims(self.y_train, axis=1))
        y_predicted_training = np.matmul(self.X_training_poly, w).flatten()
        y_predicted_test = np.matmul(self.X_test_poly, w).flatten()

        MSE_train = np.sum(
            (y_predicted_training - self.y_train)**2) / len(self.y_train)
        MSE_test = np.sum((y_predicted_test - self.y_test)**2) / len(self.y_test)
        print("Custom polynomial regression degree {} training MSE {}, test MSE {}\n".format(
            self.degree, MSE_train, MSE_test))

        return MSE_train, MSE_test


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


def MLP_regression(args):
    Xin = np.load('saved_data_sobol/Xin.npy')
    Xout = np.load('saved_data_sobol/Xout.npy')
    mlp_model = MLP(args, Xin, Xout)
    mlp_model.model_exp()

def polynomial_regression(args):
    Xin = np.load('saved_data_sobol/Xin.npy')
    Xout = np.load('saved_data_sobol/Xout.npy')
    poly_model = PolyReg(args, Xin, Xout)
    degrees = np.arange(1, 14, 1)
    train_MSE_tosave = []
    test_MSE_tosave = []
    poly_degree = []
    for d in degrees:
        poly_model.polynomial_features(degree=d)
        train_MSE, test_MSE  = poly_model.linear_regression_custom()
        poly_model.linear_regression_sklearn()
        train_MSE_tosave.append(train_MSE)
        test_MSE_tosave.append(test_MSE)
        poly_degree.append(d)


if __name__ == '__main__':
    args = arguments.args
    MLP_regression(args)
