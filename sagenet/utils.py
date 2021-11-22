import os
import csv
import numpy as np
import pandas as pd
# from graspologic.simulations import sbm
from sklearn.metrics import *
from sklearn.covariance import GraphicalLassoCV, graphical_lasso, LedoitWolf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import igraph as ig 
import networkx as nx
from community import community_louvain
import umap
import torch
import torch_geometric.data as geo_dt
from sklearn.utils.extmath import fast_logdet
import scanpy as sc
import pandas as pd
import igraph as ig
import numpy as np



def load_anndata(input_dir, tag='train', class_col='class_', ext='.h5ad', **kwargs):
    full_path = os.path.join(input_dir, tag) + ext
    print(full_path)
    dt = sc.read(full_path)
    # comm = dt.var.community
    # ind = (comm!=max(comm)).values.tolist()
    # dt.uns['adj'] = dt.uns['adj'][ind, :] 
    # dt.uns['adj'] = dt.uns['adj'][:, ind] 
    # ind = comm[ind].index.tolist()
    # dt = dt[:, ind]
    dt = dt[dt.obs[class_col].sort_values().index, :]
    dt.obs.class_ = dt.obs[class_col]
    return dt.X, (dt.obs.class_.values.astype('long')-1), dt.uns['adj'].toarray(), (dt.obs.class_.unique().astype('long')-1), dt.var.ID.values


def load_adj(path):
    """
        loads the adjacency matrix.

        Parameters
        ----------
        path: str
           Path to the directory including `adj.txt` file.

        Returns
        -------
        adj : The adjacency matrix
        num_nodes : Number of nodes
    """
    full_path = os.path.join(path, 'adj.txt')
    num_nodes = -1
    adj = []
    with open(full_path, mode='r') as txt_file:
        for row in txt_file:
            row = row.split(",")
            num_nodes += 1
            if num_nodes == 0:
                continue
            adj.append([float(row[i]) for i in range(0, len(row))])

    adj = np.asarray(adj)
    return adj, num_nodes

def load_names(path):
    full_path = os.path.join(path, 'feature_names.txt')
    if not os.path.isfile(full_path):
        return None
    features = []
    with open(full_path, mode='r') as txt_file:
        for row in txt_file:
            features.append(row.strip('\n'))
    return features

def glasso(data, alphas=5, n_jobs=None, mode='cd'):
    """
        Estimates the graph with graphical lasso finding the best alpha based on cross validation

        Parameters
        ----------
        data: numpy ndarray
            The input data for to reconstruct/estimate a graph on. Features as columns and observations as rows.
        alphas: int or array-like of shape (n_alphas,), dtype=float, default=5
            Non-negative. If an integer is given, it fixes the number of points on the grids of alpha to be used. If a list is given, it gives the grid to be used. 
        Returns
        -------
        adjacency matrix : the estimated adjacency matrix.
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    cov = GraphicalLassoCV(alphas=alphas, n_jobs=n_jobs).fit(data)
    precision_matrix = cov.get_precision()
    adjacency_matrix = precision_matrix.astype(bool).astype(int)
    adjacency_matrix[np.diag_indices_from(adjacency_matrix)] = 0
    return adjacency_matrix

def glasso_R(data, alphas, mode='cd'):
    """
        Estimates the graph with graphical lasso based on its implementation in R.

        Parameters
        ----------
        data: numpy ndarray
            The input data for to reconstruct/estimate a graph on. Features as columns and observations as rows.
        alphas: float
            Non-negative regularization parameter of the graphical lasso algorithm.
        Returns
        -------
        adjacency matrix : the estimated adjacency matrix.
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    _ , n_samples = data.shape
    cov_emp = np.dot(data.T, data) / n_samples
    covariance, precision_matrix = graphical_lasso(emp_cov=cov_emp, alpha=alphas, mode=mode)
    adjacency_matrix = precision_matrix.astype(bool).astype(int)
    adjacency_matrix[np.diag_indices_from(adjacency_matrix)] = 0
    return adjacency_matrix

def lw(data, alphas):
    """
        Estimates the graph with Ledoit-Wolf estimator.

        Parameters
        ----------
        data: numpy ndarray
            The input data for to reconstruct/estimate a graph on. Features as columns and observations as rows.
        alphas: float
            The threshold on the precision matrix to determine edges.
        Returns
        -------
        adjacency matrix : the estimated adjacency matrix.
    """
    alpha=alphas
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    cov = LedoitWolf().fit(data)
    precision_matrix = cov.get_precision()
    n_features, _ = precision_matrix.shape
    mask1 = np.abs(precision_matrix) > alpha
    mask0 = np.abs(precision_matrix) <= alpha
    adjacency_matrix = np.zeros((n_features,n_features))
    adjacency_matrix[mask1] = 1
    adjacency_matrix[mask0] = 0
    adjacency_matrix[np.diag_indices_from(adjacency_matrix)] = 0
    return adjacency_matrix



def ebic(covariance, precision, n_samples, n_features, gamma=0):
    """
    Extended Bayesian Information Criteria for model selection.
    When using path mode, use this as an alternative to cross-validation for
    finding lambda.
    See:
        "Extended Bayesian Information Criteria for Gaussian Graphical Models"
        R. Foygel and M. Drton, NIPS 2010
    Parameters
    ----------
    covariance : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance (sample covariance)
    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the model to be tested
    n_samples :  int
        Number of examples.
    n_features : int
        Dimension of an example.
    gamma : (float) in (0, 1)
        Choice of gamma=0 leads to classical BIC
        Positive gamma leads to stronger penalization of large graphs.
    Returns
    -------
    ebic score (float).  Caller should minimized this score.
    """
    l_theta = -np.sum(covariance * precision) + fast_logdet(precision)
    l_theta *= n_features / 2.

    # is something goes wrong with fast_logdet, return large value
    if np.isinf(l_theta) or np.isnan(l_theta):
        return 1e10

    mask = np.abs(precision.flat) > np.finfo(precision.dtype).eps
    precision_nnz = (np.sum(mask) - n_features) / 2.0  # lower off diagonal tri

    return -2.0 * l_theta \
        + precision_nnz * np.log(n_samples) \
        + 4.0 * precision_nnz * np.log(n_features) * gamma
    


def compare_graphs(A, Ah):
    """
        Compares a (adjacency) matrix with a reference (adjacency) matrix.

        Parameters
        ----------
        A: numpy ndarray
            The reference (adjacency) matrix.
        Ah: numpy ndarray
            The (adjacency) matrix to compare with the reference matrix.
        Returns
        -------
        TPR : true positive rate.
        TNR : true negative rate.
        FPR : false positive rate.
        FNR : false negative rate.
        accuracy : accuracy.
    """
    TP = np.sum(A[A==1] == Ah[A==1]) # true positive rate
    TN = np.sum(A[A==0] == Ah[A==0]) # true negative rate
    FP = np.sum(A[A==0] != Ah[A==0]) # false positive rate
    FN = np.sum(A[A==1] != Ah[A==1]) # false negative rate
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1_score  = 2 * precision * recall / (precision + recall)
    accuracy = (TP+TN)/(TP+FP+TN+FN)
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    FPR = FP/(FP+TN)
    FNR = FN/(FN+TP)
    BA = (TPR+TNR)/2
    return round(TPR,4), round(TNR,4), round(FPR,4), round(FNR,4), round(accuracy,4), round(BA,4) 

def compare_graphs_eigv(A, Ah):
    """
        Compares a (adjacency) matrix with a reference (adjacency) matrix based on the spectral norm.

        Parameters
        ----------
        A: numpy ndarray
            The reference (adjacency) matrix.
        Ah: numpy ndarray
            The (adjacency) matrix to compare with the reference matrix.
        Returns
        -------
        The norm-2 distance of the two matrices.
    """
    return round(np.sqrt(np.sum((np.linalg.eigvals(A)-np.linalg.eigvals(Ah))**2)),2)



def compute_metrics(y_true, y_pred):
    """
        Computes prediction quality metrics.

        Parameters:
        ----------
        y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) labels.

        y_pred : 1d array-like, or label indicator array / sparse matrix
            Predicted labels, as returned by a classifier.

        Returns:
        --------
        accuracy : accuracy
        conf_mat : confusion matrix
        precision : weighted precision score
        recall : weighted recall score
        f1 : weighted f1 score
    """
    accuracy  = accuracy_score(y_true, y_pred)
    conf_mat  = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall    = recall_score(y_true, y_pred, average='weighted')
    f1        = f1_score(y_true, y_pred, average='weighted')
    return accuracy, conf_mat, precision, recall, f1




def get_dataloader(graph, X, y, batch_size=1,undirected=True, shuffle=True, num_workers=0):
    """
        Converts a graph and a dataset to a dataloader.
        
        Parameters:
        ----------
        graph : igraph object
            The underlying graph to be fed to the graph neural networks.

        X : numpy ndarray
            Input dataset with columns as features and rows as observations.

        y : numpy ndarray
            Class labels.

        batch_size: int, default=1
            The batch size.

        undirected: boolean
            if the input graph is undirected (symmetric adjacency matrix).

        Returns:
        --------
        dataloader : a pytorch-geometric dataloader. All of the graphs will have the same connectivity (given by the input graph),
        but the node features will be the features from X.
    """
    n_obs, n_features = X.shape
    rows, cols = np.where(graph == 1)
    edges      = zip(rows.tolist(), cols.tolist())
    sources    = []
    targets    = []
    for edge in edges:
        sources.append(edge[0])
        targets.append(edge[1])
        if undirected:
            sources.append(edge[0])
            targets.append(edge[1])
    edge_index  = torch.tensor([sources,targets],dtype=torch.long)

    list_graphs = []
    y = y.tolist()
    # print(y)
    for i in range(n_obs):
        y_tensor = torch.tensor(y[i])
        X_tensor = torch.tensor(X[i,:]).view(X.shape[1], 1).float()
        data     = geo_dt.Data(x=X_tensor, edge_index=edge_index, y=y_tensor)
        list_graphs.append(data.coalesce())

    dataloader = geo_dt.DataLoader(list_graphs, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
    return dataloader


def save_np_txt(ndarray, path, colnames=None, rownames=None):
    df = pd.DataFrame(data=ndarray, index=rownames, columns=colnames)
    df.to_csv(path, sep='\t', index=True, header=True)



def kullback_leibler_divergence(X):

    """
    Finds the pairwise Kullback-Leibler divergence
    matrix between all rows in X.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Array of probability data. Each row must sum to 1.

    Returns
    -------
    D : ndarray, shape (n_samples, n_samples)
        The Kullback-Leibler divergence matrix. A pairwise matrix D such that D_{i, j}
        is the divergence between the ith and jth vectors of the given matrix X.

    Notes
    -----
    Based on code from Gordon J. Berman et al.
    (https://github.com/gordonberman/MotionMapper)

    References:
    -----------
    Berman, G. J., Choi, D. M., Bialek, W., & Shaevitz, J. W. (2014). 
    Mapping the stereotyped behaviour of freely moving fruit flies. 
    Journal of The Royal Society Interface, 11(99), 20140672.
    """

    X_log = np.log(X)
    X_log[np.isinf(X_log) | np.isnan(X_log)] = 0

    entropies = -np.sum(X * X_log, axis=1)

    D = np.matmul(-X, X_log.T)
    D = D - entropies
    D = D / np.log(2)
    D *= (1 - np.eye(D.shape[0]))

    return D

def multinomial_rvs(n, p):
    """
    Sample from the multinomial distribution with multiple p vectors.

    * n must be a scalar.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """
    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out

