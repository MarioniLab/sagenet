import os
from utils import *

class Dataset():
    def __init__(self, 
        tag,
        input_dir= 'data_input', 
        output_dir='data_input',
        random_seed=1996):
        self.input_dir  = os.path.join(input_dir, tag)
        self.output_dir = os.path.join(output_dir, tag)
        self.X_train  = None
        self.y_train  = None
        self.A_train  = None
        self.Ah_train = None
        self.X_test   = None
        self.y_test   = None
        self.A_test   = None
        self.Ah_test  = None
        self.classes  = None
        self.features = None
        self.char_features = None
        self.seed = random_seed

    def create_syn(self, **kwargs):
        self.X_train, self.y_train, self.A_train,\
            self.X_test, self.y_test, self.A_test, self.char_features = gen_syn_data(random_seed=self.seed, **kwargs)

    # def load(self, train_tag = 'train', test_tag = 'test', **kwargs):
    #     self.X_train = load_features(self.input_dir, train_tag, **kwargs)
    #     self.y_train = load_classes(self.input_dir, train_tag, **kwargs)[0]
    #     self.X_test  = load_features(self.input_dir, test_tag, **kwargs)
    #     self.y_test  = load_classes(self.input_dir, test_tag, **kwargs)[0]
    #     self.classes = load_classes(self.input_dir, train_tag, **kwargs)[5]
    #     self.features = load_names(self.input_dir)

    def load(self, train_tag = 'train', test_tag = 'test', train_class_col = 'class_', test_class_col = 'class_', **kwargs):
        print(train_tag)
        self.X_train, self.y_train, self.Ah_train, self.classes, self.features = \
            load_anndata(self.input_dir, tag=train_tag, class_col = train_class_col, **kwargs)

        self.X_test, self.y_test, self.Ah_test, _, _ = \
            load_anndata(self.input_dir, tag=test_tag, class_col = test_class_col, **kwargs)

    def subsample(self, n_obs_train=None, n_obs_test=None):
        if n_obs_train is not None:
            train_indices = sample_vec(self.y_train, n_obs_train)
            self.y_train = self.y_train[train_indices]
            self.X_train = self.X_train[train_indices, :]
        if n_obs_test is not None:
            test_indices = sample_vec(self.y_test, n_obs_test)
            self.y_test = self.y_test[test_indices]
            self.X_test = self.X_test[test_indices, :]

    def create_graph(self, method='glasso_R', alphas=5, n_jobs=None, mode='cd'):
        """
        Infer the graph (typically using graphical lasso) based on the data.
        """
        #TODO: add **kwargs
        if method == 'glasso':
            self.Ah_train = glasso(self.X_train, alphas, n_jobs, mode)
            self.Ah_test  = glasso(self.X_test, alphas, n_jobs, mode)
        elif method=="glasso_R":
            self.Ah_train = glasso_R(self.X_train, alphas, mode)
            self.Ah_test  = glasso_R(self.X_test, alphas, mode)
        elif method == "lw":
            self.Ah_train = lw(self.X_train, alphas)
            self.Ah_test  = lw(self.X_test, alphas)

    def create_noisy_true_graph(self,FPR,FNR):
        """
        Start from the true graph, and remove and add edges.
        FPR is the proportion of edges, present in the initial graph, that will be removed.
        FNR is the proportion of edges in the final graph that were not present in the initial graph.
        """
        rows, cols = np.where(self.A_train == 1)
        edges = zip(rows.tolist(), cols.tolist())
        edges_unique = []
        for i,j in edges:
            if i<j:
                edges_unique.append((i,j))
        nb_edges = len(edges_unique)
        # Remove edges from the true graph
        nb_edges_removed = int(nb_edges * FNR)
        edges_removed_ind = np.random.choice(nb_edges,nb_edges_removed,replace=False)
        A = np.copy(self.A_train)
        for ind in edges_removed_ind:
            i,j = edges_unique[ind]
            A[i,j]=0
            A[j,i]=0
        
        # Add random edges
        nb_edges_final = int(nb_edges * (1-FNR)/(1-FPR +0.00001))
        nb_edges_added = int(nb_edges_final*FPR)
        if nb_edges_added>= A.shape[0]*A.shape[0] //2-3:
            for i in range(A.shape[0]):
                for j in range(A.shape[0]):
                    A[i,j]=1
                    A[j,i]=1
        else:
            for k in range(nb_edges_added):
                added_new = False
                while not added_new:
                    i,j = np.random.choice(A.shape[0],2,replace=False)
                    if A[i,j]==0:
                        added_new=True
                        A[i,j]=1
                        A[j,i]=1
        self.Ah_train = A

    def score_graphs(self):
        # TODO: add error trap
        return compare_graphs(self.A_train, self.Ah_train), \
            compare_graphs(self.A_test, self.Ah_test)

    def comp_test(self):
        # TODO: add error trap
        return compare_graphs(self.Ah_train, self.Ah_test)

    def optim_graphs(self):
        # TODO: add!
        pass

    def _dataloader(self, dataset = 'train',batch_size=1,use_true_graph=True, shuffle=True, num_workers=0):
        if use_true_graph:
            A = self.A_train
        else:
            A = self.Ah_train

        if dataset == 'train':
            return get_dataloader(A, self.X_train, self.y_train, shuffle=shuffle, num_workers=num_workers)
        else:
            return get_dataloader(A, self.X_test, self.y_test, shuffle=shuffle, num_workers=num_workers)

    def CV_dataloaders(self,batch_size=1,use_true_graph=True,n_splits=6,graph_method="glasso_R",alpha=0.5):
        """
        Returns a generator of pairs (dataloader_train, dataloader_val) used for cross-validations.
        """
        if use_true_graph:
            A = self.A_train
        else:
            self.create_graph(method=graph_method,alphas=alpha)
            A = self.Ah_train
        kf = KFold(n_splits=n_splits)

        for train_index, test_index in kf.split(self.X_train):
            X_train, X_val = self.X_train[train_index], self.X_train[test_index]
            y_train, y_val = self.y_train[train_index], self.y_train[test_index]
            train_dataloader = get_dataloader(A,X_train,y_train)
            val_dataloader = get_dataloader(A,X_val,y_val)
            yield (train_dataloader,val_dataloader)

    def save(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        file = open("X_train.txt", "w")
        for row in self.X_train:
            np.savetxt(file, row)
        file.close()

        file = open("y_train.txt", "w")
        for row in self.y_train:
            np.savetxt(file, row)
        file.close()

        file = open("adj_train.txt", "w")
        for row in self.A_train:
            np.savetxt(file, row)
        file.close()

        file = open("adjh_train.txt", "w")
        for row in self.Ah_train:
            np.savetxt(file, row)
        file.close()

        file = open("X_test.txt", "w")
        for row in self.X_test:
            np.savetxt(file, row)
        file.close()

        file = open("y_test.txt", "w")
        for row in self.y_test:
            np.savetxt(file, row)
        file.close()

        file = open("adj_test.txt", "w")
        for row in self.A_test:
            np.savetxt(file, row)
        file.close()

        file = open("adjh_test.txt", "w")
        for row in self.Ah_test:
            np.savetxt(file, row)
        file.close()
        
