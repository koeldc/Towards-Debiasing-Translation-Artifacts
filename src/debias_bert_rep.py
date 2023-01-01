import argparse
import numpy as np
import scipy
import time

from collections import Counter
from sklearn import cluster, metrics, model_selection
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import (LogisticRegression, Perceptron, SGDClassifier,
                                  SGDRegressor)
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from torch import utils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Dict, List

import debias
import pytorch_lightning as pl
import torch

import pandas as pd
import pickle


def load_dataset(path: str) -> object:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def load_dictionary(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    k2v, v2k = {}, {}
    for line in lines:
        k, v = line.strip().split("\t")
        v = int(v)
        k2v[k] = v
        v2k[v] = k
    return k2v, v2k

def get_projection_matrix(num_clfs: int, X_train: np.ndarray, Y_train_translationese: np.ndarray, X_dev: np.ndarray, Y_dev_translationese: np.ndarray, Y_train_task: Optional[np.ndarray]=None, Y_dev_task: Optional[np.ndarray]=None, dim: int=300) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    is_autoregressive = True
    min_acc = 0.
    dim = 768
    n = num_clfs
    start = time.time()
    TYPE = "svm"
    if TYPE == "sgd":
        tr_clf = SGDClassifier
        params = {'loss': 'hinge', 'penalty': 'l2', 'fit_intercept': False, 'class_weight': None, 'n_jobs': 32}
    else:
        tr_clf = LinearSVC
        params = {'penalty': 'l2', 'C': 0.01, 'fit_intercept': True, 'class_weight': None, "dual": False}

    P, rowspace_projections, Ws = debias.get_debiasing_projection(tr_clf, params, X_train, Y_train_translationese, X_dev, Y_dev_translationese, Y_train_task, Y_dev_task, is_autoregressive=is_autoregressive, min_acc=min_acc, dim=dim, n=n)

    if is_autoregressive:
        return P, rowspace_projections, Ws
    else:
        return P

def load_data(emb_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    de_reps = np.load(emb_path + "de_cls.npy")
    en_reps = np.load(emb_path + "en_cls.npy")
    en_labels = np.ones(en_reps.shape[0], dtype=int)
    de_labels = np.zeros(de_reps.shape[0], dtype=int)
    author_data = np.concatenate((de_reps, en_reps), axis=0)
    author_labels = np.concatenate((de_labels, en_labels))
    author_train_dev, author_test, author_train_dev_label, author_test_label = sklearn.model_selection.train_test_split(
        author_data, author_labels, test_size=0.3, random_state=0
    )
    author_train, author_dev, author_train_label, author_dev_label = sklearn.model_selection.train_test_split(
        author_train_dev, author_train_dev_label, test_size=0.3, random_state=0
    )
    return author_train, author_dev, author_test, author_train_label, author_dev_label, author_test_label


def train_origin_classifier(X_train: np.ndarray, y_train: np.ndarray, X_dev: np.ndarray, y_dev: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(
        warm_start=True, penalty="l2", fit_intercept=False, verbose=5, solver="saga", random_state=23, max_iter=7
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_origin_classifier(clf: LogisticRegression, X_train: np.ndarray, y_train: np.ndarray, X_dev: np.ndarray, y_dev: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
    print(f"OG translationese classifier score on train: {clf.score(X_train, y_train)}")
    print(f"OG translationese classifier score on dev: {clf.score(X_dev, y_dev)}")
    print(f"ORIGINAL translationese classifier score on test: {clf.score(X_test, y_test)}")

        
def train_debiased_classifier(X_train: np.ndarray, y_train: np.ndarray, X_dev: np.ndarray, y_dev: np.ndarray, num_clfs: int, dim: int) -> Union[LogisticRegression, MLPClassifier]:
    P, _, _ = get_projection_matrix(num_clfs, X_train, y_train, X_dev, y_dev, dim=dim)
    X_train_debiased = np.matmul(X_train, P)
    X_dev_debiased = np.matmul(X_dev, P)
    if MLP:
        clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, alpha=1e-4,
                            solver='sgd', verbose=10, tol=1e-4, random_state=1,
                            learning_rate_init=.1)
        clf.fit(X_train_debiased, y_train)
    else:
        clf = LogisticRegression(warm_start=True, penalty='l2', fit_intercept=False, verbose=5, solver='saga', random_state=23, max_iter=7)
        clf.fit(X_train_debiased, y_train)
    return clf

def evaluate_debiased_classifier(clf: Union[LogisticRegression, MLPClassifier], X_train: np.ndarray, y_train: np.ndarray, X_dev: np.ndarray, y_dev: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, P: np.ndarray) -> None:
    X_train_debiased = np.matmul(X_train, P)
    X_dev_debiased = np.matmul(X_dev, P)
    X_test_debiased = np.matmul(X_test, P)
    print(f"DEBIASED translationese classifier score on train: {clf.score(X_train_debiased, y_train)}")
    print(f"DEBIASED translationese classifier score on dev: {clf.score(X_dev_debiased, y_dev)}")
    print(f"DEBIASED translationese classifier score on test: {clf.score(X_test_debiased, y_test)}")
    
def main(num_clfs: int, out_path: str, emb_path: str) -> None:
    X_train, X_dev, X_test, y_train, y_dev, y_test = load_data(emb_path)
    clf_original = train_original_classifier(X_train, y_train, X_dev, y_dev)
    evaluate_original_classifier(clf_original, X_train, y_train, X_dev, y_dev, X_test, y_test)
    clf_debiased = train_debiased_classifier(X_train, y_train, X_dev, y_dev, num_clfs, 768)
    P, _, _ = get_projection_matrix(num_clfs, X_train, y_train, X_dev, y_dev, dim=768)
    evaluate_debiased_classifier(clf_debiased, X_train, y_train, X_dev, y_dev, X_test, y_test, P)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", help="num of egs in de/en (for toy experiments). Defaults to entire en-de datasets", default=-1, type=int)
    parser.add_argument("--nclfs", help="num INLP iterations", default=300, type=int)
    parser.add_argument("--out", help="path to en/de debiased embeddings", default="../mono/debiasing_output/en_de/")
    args = parser.parse_args()
    main(args.nclfs, args.out, args.emb)

