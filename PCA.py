# -*- coding: utf-8 -*-
import numpy as np

def normalize(X):
    """Normalisasi dataset
    Arg:
        X: ndarray, dataset
    
    Return:
        Xbar: mengembalikan dataset yang telah di normalisasi sehingga
        memiliki standar deviasi 1 dan rata rata 0.
    """
    mu =  np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    std_filled = std.copy()
    #supaya kalo misal std nya 0 nanti mean nya ga NaN
    std_filled[std==0] = 1
    #biar rata-rata nya 0 dikurangin rata rata diri sendiri
    Xbar = (X - mu) / std_filled
    return Xbar

def eig(S):
    """ Menghitung Eigenvalues dan Eigenvector dari covarians yang diberikan
    Arg:
        X: ndarray, matriks covarians

    Return:
        (eigvals,eigvecs): ndarray, mengembalikan eigenvalues dan eigenvectors
    """
    eigvals, eigvecs = np.linalg.eig(S)
    k = np.argsort(eigvals)[::-1]
    return eigvals[k], eigvecs[:,k]

class PCA:
    """ Principal Component Analysis (PCA)

    Arg:
    n_components = int
    """
    def __init__(self, num_components):
        self.num_components = num_components

    def fit(self, X):
        """Fit model dengan X.
        Arg:
            X : array dengan bentuk (n_samples, n_features)

        Return:
            self : object.
        """
        S = np.cov(X.T)
        self.eig_vals, self.eig_vecs = eig(S)
        self.eig_vals, self.eig_vecs = self.eig_vals[:self.num_components], self.eig_vecs[:, :self.num_components]
        self.explained_variance_ = self.eig_vals
        self.components_ = (self.eig_vecs).T
        return self

    def transform(self, X):
        """Mengaplikasikan dimensionality reduction pada X.
        Arg:
            X : array dengan bentuk (n_samples, n_features)

        Return:
            X_new : (n_samples, n_components).
        """
        transformed = (self.eig_vecs).T.dot(X.T)
        return transformed.T