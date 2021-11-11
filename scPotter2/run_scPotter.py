import argparse
import json
import os
import sys
import torch

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from classifier import Classifier
from dataset import Dataset
from hyperparameters import get_hyperparams
from utils import *
import anndata as adata
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='Syn', help='Input dataset. Can either be the name of the synthetic dataset generation method, \
                                              or the path to a real dataset.')
parser.add_argument('--tag_ref', type=str, default='Syn', help='Input dataset. Can either be the name of the synthetic dataset generation method, \
                                              or the path to a real dataset.')
parser.add_argument('--tag_query', type=str, default='Syn', help='Input dataset. Can either be the name of the synthetic dataset generation method, \
                                              or the path to a real dataset.')
parser.add_argument('-i', type=str,default='../../data_tidy', help='Input dataset. Can either be the name of the synthetic dataset generation method, \
                                              or the path to a real dataset.')
parser.add_argument('--oo', type=str,default='../../output', help='Output directory.')
parser.add_argument('--classifier',type=str,default="TransformerConv",help='Type of classifier')
parser.add_argument('--n_hidden_GNN',type=int,default=0,help='Number of hidden features for the GNN. If 0, do not use a GNN.')
parser.add_argument('--n_hidden_FC',type=int,default=0,help='Number of features in the fully connected hidden layer. If 0, do not use a hidden layer.')
parser.add_argument('--n_hidden_FC2',type=int,default=0,help='Number of features in the 2nd fully connected hidden layer. If 0, do not use a 2nd hidden layer.')
parser.add_argument('--n_hidden_FC3',type=int,default=0,help='Number of features in the 3rd fully connected hidden layer. If 0, do not use a 3rd hidden layer.')
parser.add_argument('--K',type=int,default=2,help='Parameter for Chebnet GNN.')
parser.add_argument('--epochs',type=int,default=30,help='Number of training epochs.')
parser.add_argument('--seed',type=int,default=0,help='Seed for generating the synthetic dataset.')
parser.add_argument('--n_obs_train',type=int,default=1000,help='Number of observations for training (per class)')
parser.add_argument('--n_obs_test',type=int,default=1000,help='Number of observations for testing (per class)')
parser.add_argument('--imp',type=str,default='False',help='Number of observations for testing (per class)')
parser.add_argument('--train_class_col',type=str,default='class_',help='Number of observations for testing (per class)')
parser.add_argument('--test_class_col',type=str,default='class_',help='Number of observations for testing (per class)')

args = parser.parse_args()
print(args)
n_hidden_GNN = [] if args.n_hidden_GNN==0 else [args.n_hidden_GNN]

if args.n_hidden_FC==0:
  n_hidden_FC=[]
else:
  if args.n_hidden_FC2==0:
    n_hidden_FC = [args.n_hidden_FC]
  else:
    if args.n_hidden_FC3==0:
      n_hidden_FC = [args.n_hidden_FC,args.n_hidden_FC2]
    else:
      n_hidden_FC = [args.n_hidden_FC,args.n_hidden_FC2,args.n_hidden_FC3]

if not os.path.exists(args.oo):
    os.makedirs(args.oo)
args.oo = os.path.join(args.oo, args.tag)
if not os.path.exists(args.oo):
    os.makedirs(args.oo)
args.oo = os.path.join(args.oo, 'scPotter')
if not os.path.exists(args.oo):
    os.makedirs(args.oo)


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)
print('0')
# Create a synthetic dataset or load a real dataset
dataset = Dataset(tag=args.tag, input_dir=args.i, random_seed=args.seed)
dataset.load(train_tag = args.tag_ref, test_tag = args.tag_query, 
  test_class_col = args.test_class_col, train_class_col = args.train_class_col)
# dataset.subsample(n_obs_train=args.n_obs_train, n_obs_test=args.n_obs_test)
args.n_features = dataset.X_train.shape[1]
args.n_classes  = len(set(dataset.y_train.tolist()))

path_sc = os.path.join(args.i, args.tag, args.tag_query) + '.h5ad'
ad_sc = sc.read_h5ad(path_sc)
print('1')
train_dataloader = dataset._dataloader('train', use_true_graph=False, batch_size=32, shuffle=True, num_workers=0)
test_dataloader  = dataset._dataloader('test', use_true_graph=False, batch_size=32, shuffle=False, num_workers=0)



# Fit and evaluate a classifier
clf = Classifier(
        n_features=args.n_features,
        n_classes=args.n_classes,
        n_hidden_GNN=[8],
        # n_hidden_FC=[50, 10],
        dropout_GNN=0.3, 
        dropout_FC=0.2,
        # K=4,
        classifier='TransformerConv', 
        lr=.001, 
        momentum=.9,
        log_dir=None,
        device=device)


print('2')
clf.fit(train_dataloader, epochs = args.epochs, test_dataloader=None,verbose=True)

print('3')
_ = clf.eval(train_dataloader, verbose=True)
print(args.imp)
# imp = clf.interpret(train_dataloader, n_features=args.n_features, n_classes=args.n_classes)
# imp_f = "_".join(['imp', args.tag_ref, args.tag_query]) + ".txt"
# imp_f = os.path.join(args.oo, imp_f)
# print(imp_f)
# save_np_txt(imp, imp_f, colnames=dataset.classes, rownames=dataset.features)
if args.imp == 'True':
  imp_train = clf.interpret(train_dataloader, n_features=args.n_features, n_classes=args.n_classes)
  # imp_test  = clf.interpret(test_inferred, n_features=n_features, n_classes=n_classes)
  imp = imp_train 
  imp_f = "_".join(['imp', args.tag_ref, args.train_class_col]) + ".txt"
  imp_f = os.path.join(args.oo, imp_f)
  print(imp_f)
  save_np_txt(imp, imp_f, colnames=dataset.classes, rownames=dataset.features)
else:
  _ = clf.eval(test_dataloader, verbose=True)
  clf.net.eval()
  y_pred = np.empty((0, args.n_classes))
  with torch.no_grad():
      for batch in test_dataloader:
          x, edge_index, label = batch.x.to(device), batch.edge_index.to(device), batch.y.to('cpu')
          outputs = clf.net(x, edge_index)
          predicted = outputs.data.to('cpu').detach().numpy()
          y_pred = np.concatenate((y_pred, predicted), axis=0)
  y_pred = np.exp(y_pred)
  y_pred = (y_pred.T / y_pred.T.sum(0)).T
  ad_out = adata.AnnData(X = y_pred, obs = ad_sc.obs)
  ad_out.var_names = dataset.classes
  preds_f = "_".join(['preds', args.tag_ref, args.tag_query, args.train_class_col]) + ".h5ad"
  preds_f = os.path.join(args.oo, preds_f)
  # save_np_txt(y_pred, preds_f, colnames=dataset.classes)
  ad_out.write(filename=preds_f)
