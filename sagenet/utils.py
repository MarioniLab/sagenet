import os
import csv
import numpy as np
from sklearn.metrics import *
from gglasso.helper.data_generation import generate_precision_matrix, group_power_network, sample_covariance_matrix
from gglasso.problem import glasso_problem
from gglasso.helper.basic_linalg import adjacency_matrix
import torch
import torch_geometric.data as geo_dt 
from sklearn.utils.extmath import fast_logdet
import numpy as np
from scipy import sparse
import warnings
from torch.nn import Softmax



def glasso(adata, lambda_low=-10, lambda_high=-1, mode='cd'):
    """
        Recustructs the gene-gene interaction network based on gene expressions in `.X` using a guassian graphical model estimated by `glasso`. 

        Parameters
        ----------
        adata: `AnnData` 
            The annotated data matrix of shape `n_obs × n_vars`. Rows correspond to cells and columns to genes.
        alphas: int or array-like of shape (n_alphas,), dtype=`float`, default=`5`
            Non-negative. If an integer is given, it fixes the number of points on the grids of alpha to be used. If a list is given, it gives the grid to be used. 
        n_jobs: int, default `None`
            Non-negative. number of jobs.

        Returns
        -------
        adds an `csr_matrix` matrix under key `adj` to `.varm`.

        References
        -----------
        Friedman, J., Hastie, T., & Tibshirani, R. (2008). 
        Sparse inverse covariance estimation with the graphical lasso. 
        Biostatistics, 9(3), 432-441.
    """
    N = adata.shape[1]
    scaler = StandardScaler()
    data = scaler.fit_transform(adata.X)
    S    = empirical_covariance(data)
    P    = glasso_problem(S, N, latent = False, do_scaling = True)
    # lambda1_range = np.logspace(-0.1, -1, 10)
    lambda1_range = np.logspace(-10, -1,10)
    modelselect_params = {'lambda1_range': lambda1_range}
    P.model_selection(modelselect_params = modelselect_params, method = 'eBIC', gamma = 0.1, tol=1e-7)
    sol = P.solution.precision_
    P.solution.calc_adjacency(t = 1e-4)
    save_adata(adata, attr='varm', key='adj', data=sparse.csr_matrix(P.solution.precision_))



def compute_metrics(y_true, y_pred):
    """
        Computes prediction quality metrics.

        Parameters
        ----------
        y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) labels.

        y_pred : 1d array-like, or label indicator array / sparse matrix
            Predicted labels, as returned by a classifier.

        Returns
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




def get_dataloader(graph, X, y, batch_size=1, undirected=True, shuffle=True, num_workers=0):
    """
        Converts a graph and a dataset to a dataloader.
        
        Parameters
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

        shuffle: boolean, default = `True`
            Wheather to shuffle the dataset to be passed to `torch_geometric.data.DataLoader`.

        num_workers: int, default = 0
            Non-negative. Number of workers to be passed to `torch_geometric.data.DataLoader`.


        Returns
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



def kullback_leibler_divergence(X):

    """Finds the pairwise Kullback-Leibler divergence
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

        References
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
    """Sample from the multinomial distribution with multiple p vectors.

        Parameters
        ----------
        n : int
            must be a scalar >=1
        p : numpy ndarray 
            must an n-dimensional 
            he last axis of p holds the sequence of probabilities for a multinomial distribution.
        
        Returns
        -------
        D : ndarray
            same shape as p
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

def save_adata(adata, attr, key, data):
    """updates an attribute of an `AnnData` object

        Parameters
        ----------
        adata : `AnnData` 
            The annotated data matrix of shape `n_obs × n_vars`. Rows correspond to cells and columns to genes.
        attr : str
            must be an attribute of `adata`, e.g., `obs`, `var`, etc.
        key : str
            must be a key in the attr
        data : non-specific
            the data to be updated/placed

    """
    obj = getattr(adata, attr)
    obj[key] = data


def prob_con(adata, overwrite=False, inplace=True):
    if "prob_concatenated" in adata.obsm.keys():
        warnings.warn("obsm['prob_concatenated'] already exists!")
        if not overwrite:
            return adata
        else:
            warnings.warn("overwriting obsm['prob_concatenated'].")
            del adata.obsm["prob_concatenated"]
    # Get a list of obsm matrices with names starting with "prob"
    prob_matrices = [matrix_name for matrix_name in adata.obsm.keys() if matrix_name.startswith("prob")]
    # Define a function to concatenate two matrices
    def concatenate_matrices(matrix1, matrix2):
        return np.concatenate((matrix1, matrix2), axis=1)
    # Use functools.reduce to concatenate all matrices in prob_matrices
    if prob_matrices:
        concatenated_matrix = reduce(concatenate_matrices, [adata.obsm[matrix] for matrix in prob_matrices])
        adata.obsm["prob_concatenated"] = concatenated_matrix
        if inplace:
            save_adata(adata_q, attr='obsm', key='spatial', data=concatenated_matrix)
            return None
    else:
        warnings.warn("No 'prob' matrices found in the AnnData object.")
    if not inplace:
        return adata_q


def map2ref(adata_ref, adata_q, k=10, key='spatial_pred', inplace=True):
    if "spatial" not in adata_ref.obsm.keys():
        raise Exception("adata_ref.obsm['spatial'] does not exist. Necessary for spatial mapping.")
    if "prob_concatenated" not in adata_ref.obsm.keys():
        warnings.warn("obsm['prob_concatenated'] does not exsit for adata_ref. Calculating obsm['prob_concatenated'].")
        prob_con(adata_ref)
    if "prob_concatenated" not in adata_q.obsm.keys():
        warnings.warn("obsm['prob_concatenated'] does not exsit for adata_q. Calculating obsm['prob_concatenated'].")
        prob_con(adata_q)
    ref_embeddings = adata_ref.obsm['prob_concatenated']
    kdtree_r1 = cKDTree(ref_embeddings)
    target_embeddings = adata_q.obsm['prob_concatenated']
    distances, indices = kdtree_r1.query(target_embeddings, k=k)
    m             = Softmax(dim=1)
    probs         = m(-torch.tensor(distances))
    dist          = torch.distributions.categorical.Categorical(probs=probs)
    idx           = dist.sample().numpy()
    indices       = indices[np.arange(len(indices)), idx]
    adata_q.obsm['spatial'] = adata_ref.obsm['spatial'][indices] 
    if inplace:
        save_adata(adata_q, attr='obsm', key='spatial', data= adata_ref.obsm['spatial'][indices])
    else:
        return adata_q
    # swap.obs['sink']