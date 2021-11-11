import os
import csv
import numpy as np
import pandas as pd
# from graspologic.simulations import sbm
from sklearn.metrics import *
# from sklearn.covariance import GraphicalLassoCV, graphical_lasso, LedoitWolf
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

def load_classes(path, type, max_labels=None, **kwargs):
    """
        loads the classes.

        Parameters
        ----------
        path: str
            The path to the directory including `classes_train.txt` or `classes_test.txt`.
            The file should be a table with a column named `'class_'`.
        type: str
            The type of the dataset, used to read the file `classes_{type}.txt`.
        Returns
        -------
        labels : a list containing the labels 
        one_hot_labels : a numpy ndarray with the one hot encoded labels.
        num_graphs : number of samples
        num_classes : number of distinct classes
        nans : number of NA classes
    """
    full_path = os.path.join(path, 'cell_meta_{type}.txt'.format(type=type))
    classes = pd.read_csv(full_path)
    nans = pd.isna(classes['class_']).values
    classes.dropna(axis=0, inplace=True)
    classes['id'], classes_ = pd.factorize(classes.class_)
    labels = classes['id'].values
    labels -= (np.min(labels) - 1)
    # labels = classes['id'].values.astype(int)
    if (max_labels is None) or max_labels >= np.max(labels):
        num_classes = np.max(labels)
        num_graphs = labels.shape[0]
        labels -= np.ones(shape=(num_graphs,), dtype=int)
        one_hot_labels = np.zeros((num_graphs, num_classes))
        one_hot_labels[np.arange(num_graphs), labels] = 1
        return labels, one_hot_labels, num_graphs, num_classes, nans, classes_
    else:
        num_classes = max_labels
        num_graphs = labels.shape[0]
        for_one_hot = np.where(labels <= max_labels, labels, 0)
        labels = np.where(labels <= max_labels, labels, max_labels + 1)
        labels -= np.ones(shape=(num_graphs,), dtype=int)
        one_hot_labels = np.zeros((num_graphs, num_classes))
        one_hot_labels[np.arange(num_graphs), for_one_hot] = 1
        return labels, one_hot_labels, num_graphs, max_labels + 1, nans, classes_

def load_features(path, type, is_binary=False, **kwargs):
    """
        loads the input features.

        Parameters
        ----------
        path: str
            The path to the directory including `data_train.txt` or `data_test.txt`.
        type: str
            The type of the dataset, used to read the file `data_{type}.txt`.
        Returns
        -------
        features : a numpy ndarray with features as columns and samples as rows.
    """
    full_path = os.path.join(path, 'data_{type}.txt'.format(type=type))
    num_nodes = -1
    features = []
    with open(full_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            num_nodes += 1
            if num_nodes == 0:
                continue
            if is_binary:
                features.append([1 if float(row[i]) > 0 else 0 for i in range(0, len(row))])
            else:
                features.append([float(row[i]) for i in range(0, len(row))])
    features = np.asarray(features)
    features = features.T
    return features

def gen_syn_data(
    n_classes=3,
    n_obs_train=200,
    n_obs_test=100,
    n_features=10,
    n_edges=3,
    n_char_features=10,
    signal=[0, 0],
    diff_coef=[0.1, 0.1],
    noise=[0.2, 0.2],
    n_communities=5,
    probs=[0.5, 0.1],
    n_iter=3,
    model='BA',
    syn_method="sign",
    random_seed=1996):
    """
        Generates synthetic training and test datasets based on an underlying random graph model.
        Each class is defined by a set of characteristic features. 
        Each feature starts with random values. For each observation, the characteristic features of its class are increased by "signal".
        Then, values on each node are altered based on the synthetic method used.

        Parameters:
        ----------
        n_classes: int
            Number of classes
        n_obs_train: int 
            Number of observations per class for the training dataset
        n_obs_test: int 
            Number of observations per class for the test dataset
        n_features: int
            Number of features, each corresponding to a node in the graph
        n_char_features: int
            Number of features that are specific to each class
        signal: [float, float]
            The level of initial signal for the characteristic features, for training and test dataset respectively. 
            Only used when `syn_method == 'diffusion'` or `syn_method == 'activation'`.
        diff_coef: [float, float]
            How much each value transmits its value over the edges, for training and test dataset respectively.
            Only used when `syn_method == 'diffusion'`.
        noise: [float, float]
            (Gaussian) Noise level added at the end of the information passing, for training and test dataset respectively. 
        n_communities: int
            Number of graph communities for the Stochastic Block Model. Used only when `model == 'SBM'`.
        probs: [float, float]
            Probability of intra and inter cluster edges for the Stochastic Block Model. Used only when `model == 'SBM'`.
        model: str
            The random graph generation model. Can be `'BA'` for Barabási–Albert, `'ER'` for Erdős–Rényi, or `'SBM'` for Stochastic Block Model.
        syn_method: str
            The message passing synthetic process. Can be:
                `'diffusion'` for diffusing information over edges based on the difference on the end nodes.
                `'activation'` for activating a characteristic node based on its neighbors.
                `'sign'` for changing the sign of a characteristic node based on the average sign of its neighbors.

        Returns
        -------
        X_train : a numpy ndarray with features generated for the training dataset.
        y_train : a numpy ndarray with labels generated for the training dataset.
        adj_train : the adjacency matrix of the graph generated for the training dataset.
        X_test : a numpy ndarray with features generated for the test dataset.
        y_test : a numpy ndarray with labels generated for the test dataset.
        adj_test : the adjacency matrix of the graph generated for the test dataset.
    """
    np.random.seed(random_seed)
    if model=='ER':
        # Generate a random graph with the Erdos-Renyi model.
        graph_train = graph_test = ig.Graph.Erdos_Renyi(n=n_features, m=n_edges*n_features, directed=False)
        adj_train = adj_test = np.array(graph_train.get_adjacency().data)
    elif model=="BA":
        # Generate a scale-free graph with the Barabasi-Albert model.
        graph_train  = graph_test = ig.Graph.Barabasi(n_features, n_edges, directed=False)
        adj_train = adj_test = np.array(graph_train.get_adjacency().data)
    elif model=='SBM':
        # Generate a random graph with the stochastic block matrix model.
        n = [n_features // n_communities] * n_communities
        p = np.full((n_communities, n_communities), probs[1])
        adj_train = sbm(n=n, p=p)
        adj_test  = sbm(n=n, p=p)
        graph_train = ig.Graph.Adjacency(adj_train.tolist())
        graph_test  = ig.Graph.Adjacency(adj_test.tolist())
    elif model=='linear':
        g = ig.Graph()
        edges = [(i, i+1) for i in range(n_features-1)]
        g.add_vertices(n_features)
        g.add_edges(edges)
        graph_train = graph_test = g
        adj_train = np.array(g.get_adjacency().data)
        adj_test  = np.array(g.get_adjacency().data)
    else:
        print("Unrecognized random graph generation model. Please use ER, BA, linear, or SBM.")
    X_train = []
    y_train = []
    X_test  = []
    y_test  = []
    char_feat = dict()
    if syn_method=="sign":
        for c in range(n_classes):
            # Draw the features which define this class
            char_features = np.random.choice(n_features,size=n_char_features,replace=False)
            char_feat[c] = char_features
            for i in range(n_obs_train):
                # Start from a random vector
                features = np.random.normal(0, 1, n_features)
                features_next = np.copy(features)
                for f in char_features:
                    s=0
                    for neighbor in graph_train.neighbors(f):
                        s+=features[neighbor]
                    #set the sign to the average sign of the neighbours
                    features_next[f] = np.sign(s)* np.abs(features[f])
                features = features_next
                # Add additional noise
                if noise[0] > 0:
                    features += np.random.normal(0, noise[0], n_features)
                X_train.append(features)
                y_train.append(c)

            for i in range(n_obs_test):
                # Start from a random vector
                features = np.random.normal(0, 1, n_features)
                features_next = np.copy(features)
                for f in char_features:
                    s=0
                    for neighbor in graph_train.neighbors(f):
                        s+=features[neighbor]
                    # Set the sign to the average sign of the neighbours
                    features_next[f] = np.sign(s)*  np.abs(features[f])
                features = features_next
                # Add additional noise
                if noise[1] > 0:
                    features += np.random.normal(0, noise[1], n_features)
                X_test.append(features)
                y_test.append(c)
    elif syn_method=="diffusion":
        for c in range(n_classes):
            signal[0] = np.random.normal(signal[0], 1, 1)
            signal[1] = np.random.normal(signal[1], 1, 1)
            # Draw the features which define this class
            char_features = np.random.choice(n_features,size=n_char_features,replace=False)
            char_feat[c] = char_features
            for i in range(n_obs_train):
                # Start from a random vector
                features = np.abs(np.random.normal(0, 1, n_features))
                # Increase the value for the characteristic features
                features[char_features] += np.abs(np.random.normal(signal[0], 1, n_char_features))
                features = features / np.linalg.norm(features)
                # Diffuse values through the graph
                for it in range(n_iter):
                    features_next = np.copy(features)
                    for e in graph_train.es:
                        features_next[e.target]+= (features[e.source] - features[e.target]) * diff_coef[0]
                        features_next[e.source]+= (features[e.target] - features[e.source]) * diff_coef[0]
                    features = features_next
                if noise[0] > 0:
                    features += np.random.normal(0, noise[0], n_features)
                X_train.append(features)
                y_train.append(c)

            for i in range(n_obs_test):
                # Start from a random vector
                features = np.abs(np.random.normal(0, 1, n_features))
                # Increase the value for the characteristic features
                features[char_features] += np.abs(np.random.normal(signal[1], 1, n_char_features))
                features = features / np.linalg.norm(features)
                # Diffuse values through the graph
                for it in range(n_iter):
                    features_next = np.copy(features)
                    for e in graph_test.es:
                        features_next[e.target]+= (features[e.source] - features[e.target]) * diff_coef[1]
                        features_next[e.source]+= (features[e.target] - features[e.source]) * diff_coef[1]
                    features = features_next
                if noise[1] > 0:
                    features += np.random.normal(0, noise[1], n_features)
                X_test.append(features)
                y_test.append(c)
    
    elif syn_method=="activation":
        for c in range(n_classes):
            # Draw the features which define this class
            char_features = np.random.choice(n_features,size=n_char_features,replace=False)
            char_feat[c] = char_features
            for i in range(n_obs_train):
                # Start from a random vector
                features = np.random.normal(0, 1, n_features)
                features_next = np.copy(features)
                for f in char_features:
                    s=0
                    degree=0
                    for neighbor in graph_train.neighbors(f):
                        s+=features[neighbor]
                        degree+=1
                    degree = max(degree,1)
                    features_next[f] = np.random.normal(s/degree * signal[0],0.2) 

                features = features_next
                if noise[0] > 0:
                    features += np.random.normal(0, noise[0], n_features)
                X_train.append(features)
                y_train.append(c)

            for i in range(n_obs_test):
                # Start from a random vector
                features = np.random.normal(0, 1, n_features)            
                features_next = np.copy(features)
                for f in char_features:
                    s=0
                    degree=0
                    for neighbor in graph_train.neighbors(f):
                        s+=features[neighbor]
                        degree+=1
                    degree = max(degree,1)
                    features_next[f] = np.random.normal(s/degree * signal[1],0.2) 

                features = features_next
                if noise[1] > 0:
                    features += np.random.normal(0, noise[1], n_features)
                X_test.append(features)
                y_test.append(c)
    else:
        print("Unrecognized synthetic dataset generation method!")
    train_idx = np.random.permutation(len(y_train)) - 1
    X_train   = np.array(X_train)[train_idx, :]
    y_train   = np.array(y_train)[train_idx]
    test_idx  = np.random.permutation(len(y_test)) - 1
    X_test    = np.array(X_test)[test_idx, :]
    y_test    = np.array(y_test)[test_idx]

    return np.absolute(X_train), y_train, adj_train, \
        np.absolute(X_test), y_test, adj_test, char_feat


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


def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Parameters:
    ----------
    g : networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition : dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos : dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):


    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

def draw_graph(adjacency_matrix, node_color=None):
    """
        Draw a modular graph, also color its nodes based on the node communities detected by the Louvain algorithm.

        Parameters:
        ----------
        adjacency_matrix : numpy ndarray
            The adjacency matrix.

        node_color : list, default=None
            A list of colors to color nodes. If `None` the nodes will be colored based on the node communities detected by the Louvain algorithm.


        Returns:
        --------
        pos : dict mapping int node -> (float x, float y)
            node positions

    """
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    g = nx.Graph()
    g.add_edges_from(edges)
    partition = community_louvain.best_partition(g)
    pos = community_layout(g, partition)
    if node_color == None:
      node_color = list(partition.values())
    nx.draw(g, pos, node_color=node_color, node_size=10); 
    return list(partition.values())

def plot_lowDim(data, labels=None, title=None):
    """
        Visualizes the 2-dimensional embedding of a dataset based on the UMAP algorithm.

        Parameters:
        ----------
        data : numpy ndarray
            The input data with features as columns and observations as rows.

        labels : list
            The point labels. Used to color the points in the 2-dimensional plot.

    """
    reducer   = umap.UMAP()
    embedding = reducer.fit_transform(data)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
    plt.title(title)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')


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



def sample_vec(vec, n):
    """
        Subsample a vector uniformly from each level. Used to subsample datasets with several classes in a balanced manner.
        
        Parameters:
        ----------
        vec : numpy ndarray
            The vector to sample from.

        n : int
            Number of samples per level.

        Returns:
        --------
        to_ret : a numpy array including indices of the selected subset.
    """
    vec_list = vec.tolist()
    vec_list = set(vec_list)
    to_ret = np.array([], dtype='int')
    for val in vec_list:
        ii = np.where(vec == val)[0] 
        index = np.random.choice(ii, n)
        to_ret = np.append(to_ret, index)
    return to_ret

def save_np_txt(ndarray, path, colnames=None, rownames=None):
    df = pd.DataFrame(data=ndarray, index=rownames, columns=colnames)
    df.to_csv(path, sep='\t', index=True, header=True)




