import glob
import math
import numpy as np
import argparse
import sys
import pickle
import time
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
                                 'batch_size': [32, 64, 128],
                                 'learning_rate_init': [1e-1, 1e-2, 1e-3]}

        self.model = MLPRegressor(activation='logistic', solver='adam', alpha=0,
                                  max_iter=1000, random_state=1,
                                  tol=1e-9, verbose=False, n_iter_no_change=1000)

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
        cv_summary = np.load('plots/new_data/numpy/hyper/summary.npy').item()

        print(best_params)

        params = cv_summary['params']
        scores = cv_summary['mean_test_score']
        for i in range(len(params)):
            print("param is {} and MSE_validation is {:.6f}".format(
                params[i], -scores[i]))

        self.model.set_params(**best_params)
        self.model.set_params(verbose=True)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)

        MSE_test = mean_squared_error(self.y_test, y_pred)
        print(MSE_test)

        np.save('saved_weights/coefs.npy', self.model.coefs_)
        np.save('saved_weights/intercepts.npy', self.model.intercepts_)
        pickle.dump(self.model, open('saved_weights/model.sav', 'wb'))
        print(self.model.coefs_[0].shape)
        print(self.model.coefs_[1].shape)
        print(self.model.intercepts_[0].shape)
        print(self.model.intercepts_[1].shape)

    def model_exp(self):
        print("start")
        start = time.time()

        # regr = MLPRegressor(hidden_layer_sizes=(256,), activation='logistic', solver='adam', alpha=0,
        #                     batch_size=32, learning_rate_init=1e-2, max_iter=2000, random_state=1,
        # tol=1e-9, verbose=True, n_iter_no_change=1000).fit(self.X_train,
        # self.y_train)
        regr = MLPRegressor(hidden_layer_sizes=(128,), activation='logistic', solver='adam', alpha=0,
                            batch_size=64, learning_rate_init=1e-2, max_iter=1000, random_state=1,
                            tol=1e-9, verbose=False, n_iter_no_change=1000).fit(self.X_train, self.y_train)

        y_pred = regr.predict(self.X_test)
        MSE_test = mean_squared_error(self.y_test, y_pred)
        print(MSE_test)
        end = time.time()
        print("time spent {}".format(end - start))
        # np.save('saved_weights/coefs.npy', regr.coefs_)
        # np.save('saved_weights/intercepts.npy', regr.intercepts_)
        # pickle.dump(regr, open('saved_weights/model.sav', 'wb'))
        # plt.plot(regr.loss_curve_)
        # plt.show()


class PolyReg(object):

    def __init__(self, args, Xin, Xout):
        self.args = args
        self.X_mean = Xin.mean(0)
        self.Xin = Xin - self.X_mean
        self.Xout = Xout
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.Xin, self.Xout, test_size=0.1, random_state=1)

    @classmethod
    def poly_features_flexible(cls, degree, X):
        # TODO: Bad implementation
        poly = PolynomialFeatures(degree)
        transform_X = X[:, :-1]
        category_X = X[:, -1:]
        transform_X = poly.fit_transform(transform_X)
        X_poly = np.concatenate((transform_X, category_X), axis=1)
        return X_poly

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
        regr = LinearRegression(fit_intercept=False).fit(
            self.X_training_poly, self.y_train)
        y_pred_train = regr.predict(self.X_training_poly)
        y_pred_test = regr.predict(self.X_test_poly)
        MSE_train = mean_squared_error(self.y_train, y_pred_train)
        MSE_test = mean_squared_error(self.y_test, y_pred_test)
        print("Sklern polynomial regression degree {} training MSE {}, test MSE {}\n".format(
            self.degree, MSE_train, MSE_test))

        pickle.dump(regr, open('saved_weights/poly.sav', 'wb'))

        return MSE_train, MSE_test

    def linear_regression_custom(self):
        print(self.X_training_poly.shape[1])
        print(np.linalg.matrix_rank(self.X_training_poly))
        print(np.linalg.matrix_rank(
            np.matmul(self.X_training_poly.transpose(), self.X_training_poly)))

        # Analytical form
        tmp = np.linalg.inv(
            np.matmul(self.X_training_poly.transpose(), self.X_training_poly))
        w = np.matmul(np.matmul(tmp, self.X_training_poly.transpose()),
                      np.expand_dims(self.y_train, axis=1))
        y_predicted_training = np.matmul(self.X_training_poly, w).flatten()
        y_predicted_test = np.matmul(self.X_test_poly, w).flatten()

        MSE_train = np.sum(
            (y_predicted_training - self.y_train)**2) / len(self.y_train)
        MSE_test = np.sum((y_predicted_test - self.y_test)
                          ** 2) / len(self.y_test)
        print("Custom polynomial regression degree {} training MSE {}, test MSE {}\n".format(
            self.degree, MSE_train, MSE_test))

        return MSE_train, MSE_test


def MLP_regression(args):
    Xin = np.load('saved_data_sobol/Xin.npy')
    Xout = np.load('saved_data_sobol/Xout.npy')
    mlp_model = MLP(args, Xin, Xout)
    # mlp_model.model_fit()
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
        train_MSE, test_MSE = poly_model.linear_regression_custom()
        poly_model.linear_regression_sklearn()
        train_MSE_tosave.append(train_MSE)
        test_MSE_tosave.append(test_MSE)
        poly_degree.append(d)
    np.save('plots/new_data/numpy/polynomial/train_MSE.npy',
            np.asarray(train_MSE_tosave))
    np.save('plots/new_data/numpy/polynomial/test_MSE.npy',
            np.asarray(test_MSE_tosave))
    np.save('plots/new_data/numpy/polynomial/poly_degree.npy',
            np.asarray(poly_degree))

    # Run the best again
    poly_model.polynomial_features(degree=10)
    poly_model.linear_regression_sklearn()


if __name__ == '__main__':
    args = arguments.args
    MLP_regression(args)
    # polynomial_regression(args)
