"""Training-only feature scaling and validation-selected compact classifiers."""
import copy
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .evaluation import score
from .models import predict_head


def group_weights(labels,groups):
    """Each source family contributes equally within its class."""
    pairs=list(zip(labels,groups));counts={pair:pairs.count(pair) for pair in set(pairs)}
    weights=np.asarray([1/counts[pair] for pair in pairs],dtype=float)
    for label in [0,1]:
        mask=np.asarray(labels)==label;weights[mask]*=mask.sum()/weights[mask].sum()
    return weights


def fit_candidates(x_train,y_train,groups,x_val,y_val,seed):
    scaler=StandardScaler().fit(x_train)
    base={'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'temperature':1.}
    a=scaler.transform(x_train);b=scaler.transform(x_val)
    weights=group_weights(y_train,groups)
    choices=[];heads=[]
    def register(head,settings):
        stats=score(y_val,predict_head(x_val,head,calibrated=False))
        choices.append({**settings,'validation':stats});heads.append(head)
    for c in [.001,.01,.1,1.,10.]:
        clf=LogisticRegression(C=c,max_iter=2000,random_state=seed).fit(a,y_train,sample_weight=weights)
        register({**base,'classifier_type':'linear','coef':clf.coef_[0].tolist(),'intercept':float(clf.intercept_[0])},
                 {'model':'EfficientNet + regularized logistic','C':c})
    histories=[]
    for decay in [.01,.1]:
        torch.manual_seed(seed)
        net=nn.Sequential(nn.Linear(a.shape[1],32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,1))
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=decay)
        tx=torch.tensor(np.clip(a,-10,10),dtype=torch.float32)
        vx=torch.tensor(np.clip(b,-10,10),dtype=torch.float32)
        ty=torch.tensor(y_train,dtype=torch.float32);tw=torch.tensor(weights,dtype=torch.float32)
        best=None;best_key=(-1.,-float('inf'));stale=0;history=[]
        for epoch in range(1,151):
            net.train();optimizer.zero_grad()
            loss=(nn.functional.binary_cross_entropy_with_logits(net(tx).ravel(),ty,reduction='none')*tw).mean()
            loss.backward();optimizer.step()
            if epoch%5:continue
            net.eval()
            with torch.inference_mode():p=torch.sigmoid(net(vx).ravel()).numpy()
            stats=score(y_val,p);key=(stats['roc_auc'],-stats['brier'])
            history.append({'epoch':epoch,'training_loss':float(loss.detach()),'validation_auc':stats['roc_auc'],'validation_brier':stats['brier']})
            if key>best_key:best_key=key;best=(copy.deepcopy(net.state_dict()),epoch);stale=0
            else:stale+=1
            if stale>=8:break
        net.load_state_dict(best[0]);net.eval()
        head={**base,'classifier_type':'mlp','feature_clip':10.,
              'hidden_weight':net[0].weight.detach().numpy().T.tolist(),'hidden_bias':net[0].bias.detach().numpy().tolist(),
              'coef':net[3].weight.detach().numpy().ravel().tolist(),'intercept':float(net[3].bias.detach()[0])}
        register(head,{'model':'EfficientNet + neural head','hidden_units':32,'weight_decay':decay,'selected_epoch':best[1]})
        histories.append({'weight_decay':decay,'epochs':history})
    chosen=max(range(len(choices)),key=lambda i:(choices[i]['validation']['roc_auc'],-choices[i]['validation']['brier']))
    return heads[chosen],choices,histories,chosen
