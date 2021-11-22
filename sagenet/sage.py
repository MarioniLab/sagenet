from utils import *
from classifer import *
from model import *

class sage():
    def __init__(self, random_seed=1996):
        self.random_seed = random_seed
        self.models    = {}
        inf_genes = None
        self.num_refs = 0

    def add_ref(self, 
                        adata, 
        tag = None,

        comm_columns = 'class_',
        compute_gin  = False,
        num_workers  = 0,
        batch_size   = 32,
        epochs       = 10,
        device       = 'cpu',
        n_genes      = 10):

        ents = np.zeros(adata.shape[1])
        self.num_refs += 1

        if tag is None:
            tag = 'ref' + str(self.num_refs + 1)

        for comm in comm_columns:
            data_loader = get_dataloader(
                graph       = adata.uns['adj'].toarray(), 
                X           = adata.X, y = adata.obs[comm].values.astype('long'), 
                batch_size  = batch_size,
                shuffle     = True, 
                num_workers = num_workers
            )

            clf = Classifier(
                n_features   = adata.shape[1],
                n_classes    = (np.max(adata.obs[comm].values.astype('long'))+1),
                n_hidden_GNN = [8],
                dropout_FC   = 0.2,
                dropout_GNN  = 0.3,
                classifier   = 'TransformerConv', 
                lr           = 0.001,
                momentum     = 0.9,
                log_dir      = "runs/TransformerConv_true",
                device       = device
            ) 

            clf.fit(data_loader, epochs = epochs, test_dataloader=None,verbose=True)
            _, ent = clf.interpret(data_loader, n_features=adata.shape[1], n_classes=(np.max(adata.obs[comm].values.astype('long'))+1))
            ents  += ent
            self.models['_'.join([tag, comm])] = clf
        # ents /= len(comm_columns)
        ind = np.argsort(ents)[0:n_genes]
        return adata.var_names[ind]


    def map_query(self, adata_q):
        data_loader = get_dataloader(
                graph       = adata_q.uns['adj'].toarray(), 
                X           = adata_q.X, y = adata_q.obs['class_'].values.astype('long'), #TODO: fix this
                batch_size  = 1,
                shuffle     = False, 
                num_workers = 0
            )
        dist_mat = np.zeros((adata_q.shape[0], adata_q.shape[0]))
        for tag in self.models.keys():
            self.models[tag].net.eval()
            i = 0
            with torch.no_grad():
                for batch in data_loader:
                    x, edge_index, label = batch.x.to(device), batch.edge_index.to(device), batch.y.to('cpu')
                    outputs = self.models[tag].net(x, edge_index)
                    predicted = outputs.data.to('cpu').detach().numpy()
                    i += 1
                    if i == 1:
                        n_classes = predicted.shape[1]
                        y_pred = np.empty((0, n_classes))
                    y_pred = np.concatenate((y_pred, predicted), axis=0)
            
            y_pred = np.exp(y_pred)
            y_pred = (y_pred.T / y_pred.T.sum(0)).T
            adata_q.obs['_'.join(['pred', tag])] = np.argmax(y_pred, axis=1)
            temp = (-y_pred * np.log2(y_pred)).sum(axis = 1)
            adata_q.obs['_'.join(['ent', tag])] = np.array(temp) / np.log2(n_classes)
            y_pred_1 = (multinomial_rvs(1, y_pred).T * np.array(adata_q.obs['_'.join(['ent', tag])])).T
            y_pred_2 = (y_pred.T * (1-np.array(adata_q.obs['_'.join(['ent', tag])]))).T
            y_pred_final = y_pred_1 + y_pred_2
            kl_d = kullback_leibler_divergence(y_pred_final)
            kl_d = kl_d + kl_d.T
            kl_d /= np.linalg.norm(kl_d, 'fro')
            dist_mat += kl_d
        # adata.obsm['dist_map'] = dist_mat
        return dist_mat


    def get_lm_genes(self, adata):
        pass

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass


