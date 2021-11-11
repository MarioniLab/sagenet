from classifier import Classifier

def select_hyperparameters_CV(
    dataset,
    n_features,
    n_classes,
    n_hidden_GNN=[10],
    n_hidden_FC=[],
    K=4,
    classifier='MLP', 
    lr=.01, 
    momentum=.9,
    epochs=50,
    device='cpu',
    batch_size=16,
    dropout_rate_list=[0,0.1,0.2,0.5],
    alpha_list=[0.5,1,2,3,5],
    use_true_graph=True,
    graph_method="glasso_R"):
    """
    Select the best dropout rate and/or the best alpha parameter using cross-validation
    """

    best_alpha=0
    best_rate=0
    best_score=0
    for dropout_rate in dropout_rate_list:
        for alpha in alpha_list:
            score=0
            for train_dataloader,val_dataloader in dataset.CV_dataloaders(use_true_graph=use_true_graph,n_splits=4,batch_size=batch_size,graph_method=graph_method,alpha=alpha):
                clf = Classifier(n_features=n_features,n_classes=n_classes,classifier=classifier,K=K,n_hidden_FC=n_hidden_FC,n_hidden_GNN=n_hidden_GNN,\
                    dropout_GNN = dropout_rate, dropout_FC=dropout_rate, lr=lr,momentum=momentum,device=device)

                clf.fit(train_dataloader, epochs = epochs, test_dataloader=val_dataloader,verbose=False)
                score+= clf.eval(val_dataloader,verbose=False)[0]

            if score>best_score:
                best_score = score
                best_rate = dropout_rate
                best_alpha = alpha

    return best_rate,best_alpha

def select_alpha(n_obs):
  """ Select alpha based on the number of observations"""
  if n_obs<=250:
    return 0.5
  elif n_obs<=600:
    return 1
  elif n_obs<=1200:
    return 2
  elif n_obs<=1700:
    return 2.5
  elif n_obs<=2000:
    return 3
  elif n_obs<=3500:
    return 4
  elif n_obs<=5500:
    return 5
  elif n_obs<=6500:
    return 6
  elif n_obs<=7500:
    return 8
  elif n_obs<=9000:
    return 11
  else:
    return 15

def get_hyperparams(CV_dropout,CV_alpha,dataset,n_features,n_obs_train,n_classes,n_hidden_GNN,n_hidden_FC,K,classifier,lr,momentum,epochs,\
    device,batch_size,use_true_graph,dropout_rate,alpha):
    """Return the dropout rate and the alpha parameter, either by doing cross-validation or by using specified values."""
    if CV_alpha or CV_dropout: # run CV
        if CV_dropout:
            dropout_rate_list=[0,0.1,0.2,0.5] # grid of dropout rate values
        else:
            dropout_rate_list=[dropout_rate]
        if CV_alpha:
            alpha_list = [0.5,1,2,3,4,6] # grid of alpha values
        else:
            if alpha is None:
            #select alpha based on the number of observations
                alpha_list = [select_alpha(n_obs_train)]
            else:
                alpha_list=[alpha]
        dropout_rate,alpha = select_hyperparameters_CV(dataset=dataset,n_features=n_features,n_classes=n_classes,n_hidden_GNN=n_hidden_GNN,n_hidden_FC=n_hidden_FC,\
                K=K,classifier=classifier,lr=0.001,momentum=0.9,epochs=epochs,device=device,batch_size=batch_size,dropout_rate_list=dropout_rate_list,\
                alpha_list=alpha_list,use_true_graph=use_true_graph,graph_method="glasso_R")
        print("Selected dropout rate: " + str(dropout_rate))
        print("Selected alpha: " + str(alpha))
    else: # don't run CV
        if alpha is None:
            alpha = select_alpha(n_obs_train)
            print("Selected alpha: " + str(alpha))
    return dropout_rate,alpha

