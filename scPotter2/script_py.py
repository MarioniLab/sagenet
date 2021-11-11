from utils import *
from dataset import *
from model import *
from classifier import *
from captum import *
from captum.attr import IntegratedGradients
import scanpy
import pandas as pd
import igraph as ig
import numpy as np

import torch
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)
print(device)




# dataset params
n_obs_train = 100
n_obs_test  = 100 
random_seed = 1996
alphas      = 2
# NN params
epochs   = 20
lr       = 0.001
momentum = .9

# make a Dataset instance 
dataset = Dataset(tag='seqfish_mouse_embryo', input_dir='../../../data_tidy', random_seed=random_seed)
dataset.load(train_tag = 'exp_embryo1_2', test_tag = 'exp_embryo2_2')
# dataset.subsample(n_obs_train=n_obs_train, n_obs_test=n_obs_test)
n_features = dataset.X_train.shape[1]
n_classes  = len(set(dataset.y_train.tolist()))
# infer the graph for train and test
# dataset.create_graph(alphas=2, method='glasso_R', mode='cd')
# adj = load_adj('data_input/pbmc')[0]
# dataset.Ah_train = dataset.Ah_test = adj
# # dataloaders with true graphs
# train_true = dataset._dataloader('train',use_true_graph=True,batch_size=16)
# test_true  = dataset._dataloader('test',use_true_graph=True,batch_size=16)
# dataloaders with inferred graphs
train_inferred = dataset._dataloader('train', use_true_graph=False, batch_size=100, shuffle=True)
test_inferred  = dataset._dataloader('test', use_true_graph=False, batch_size=100, shuffle=False)

clf_TransformerConv = Classifier(
        n_features=n_features,
        n_classes=n_classes,
        n_hidden_GNN=[8],
        dropout_FC=0.2,
        dropout_GNN=0.3,
        classifier='TransformerConv', 
        lr=0.001,
        momentum=momentum,
        log_dir="runs/TransformerConv_true",
        device = device) 
# fit the classifier on train data
clf_TransformerConv.fit(train_inferred, epochs = 10, test_dataloader=None,verbose=True)
# evaluate the trained classifier on train data
_ = clf_TransformerConv.eval(train_inferred, verbose=True)
# _ = clf_TransformerConv.eval(test_inferred, verbose=True)
# imp_train_TransformerConv = clf_TransformerConv.interpret(train_inferred, n_features=n_features, n_classes=n_classes)
# # imp_test_TransformerConv  = clf_TransformerConv.interpret(test_inferred, n_features=n_features, n_classes=n_classes)
# imp_TransformerConv = imp_train_TransformerConv 
# save_np_txt(imp_TransformerConv, '../../output/imp.txt', colnames=dataset.classes, rownames=dataset.features)

clf_TransformerConv.net.eval()
y_pred = np.empty((0, n_classes))
with torch.no_grad():
    for batch in test_inferred:
        x, edge_index, label = batch.x.to(device), batch.edge_index.to(device), batch.y.to('cpu')
        outputs = clf_TransformerConv.net(x, edge_index)
        predicted = outputs.data.to('cpu').detach().numpy()
        y_pred = np.concatenate((y_pred, predicted), axis=0)
print(y_pred.shape)
save_np_txt(y_pred, '../../../output/preds.txt', colnames=dataset.classes)





