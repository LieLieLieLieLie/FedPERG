"""Common-harness ports of publicly specified recent aggregators.

FedAWA: CVPR 2025, ChanglongShi/FedAWA/server_funct.py (one Adam step).
FedDS: Information Sciences 745 (2026) 123425, Gia99999/FedDS.
These implementations retain the aggregation mechanisms, not the authors'
dataset/local-training protocol. See BASELINE_PROVENANCE.md for differences.
"""
import numpy as np
import torch


def dawid_skene_weights(predictions, classes, max_iter=500, tol=1e-5):
    """Released standard DS: majority initialization, uniform latent prior.

    Log-space E steps avoid underflow without changing the uniform-prior
    likelihood. Empty rows and all-zero likelihoods match the released fallback.
    No true public labels or sample counts enter the reliability weights.
    """
    pred=np.asarray(predictions,dtype=np.int64)
    n,k=pred.shape
    q=np.zeros((n,classes),dtype=np.float64)
    for j in range(k):q[np.arange(n),pred[:,j]]+=1./k
    q+=1e-10;q/=q.sum(1,keepdims=True)
    indicators=[]
    for j in range(k):
        labels=np.unique(pred[:,j])
        indicators.append((labels,(pred[:,j,None]==labels).astype(np.float64)))

    def m_step(q):
        cm=np.zeros((k,classes,classes),dtype=np.float64)
        for j in range(k):
            # Sparse prediction support avoids a dense C-by-C item loop.
            labels,indicator=indicators[j]
            cm[j][:,labels]=q.T@indicator
        mass=cm.sum(2,keepdims=True)
        return cm/np.where(mass==0,1.,mass)

    cm=m_step(q)
    for iteration in range(max_iter):
        old=q.copy();logq=np.zeros_like(q)
        with np.errstate(divide='ignore'):
            for j in range(k):logq+=np.log(cm[j][:,pred[:,j]].T)
        maximum=logq.max(1,keepdims=True)
        valid=np.isfinite(maximum[:,0])
        q[:]=1./classes
        values=np.exp(logq[valid]-maximum[valid])
        q[valid]=values/values.sum(1,keepdims=True)
        cm=m_step(q)
        if np.max(np.abs(q-old))<tol:break
    reliability=np.diagonal(cm,axis1=1,axis2=2).mean(1)
    weights=(reliability/reliability.sum() if reliability.sum()>0
             else np.full(k,1./k))
    return weights,{'ds_iterations':iteration+1,'ds_reliability':reliability.tolist()}


class FedAWAState:
    """Persist client logits; use the official last pre-step probability."""
    def __init__(self,clients):self.logits=None;self.clients=clients

    def aggregate(self,deltas,p,global_flat,selected):
        if self.logits is None:
            self.logits=torch.zeros(self.clients,device=p.device)
            self.initialized=torch.zeros(self.clients,device=p.device,dtype=torch.bool)
        ids=torch.as_tensor(selected,device=p.device)
        fresh=~self.initialized[ids]
        self.logits[ids[fresh]]=p[fresh]
        self.initialized[ids]=True
        alpha=self.logits[ids].detach().clone().requires_grad_(True)
        vectors=torch.stack([torch.cat([v.reshape(-1) for v in d]) for d in deltas])
        models=global_flat[None]+vectors
        costs=1-torch.nn.functional.cosine_similarity(models,global_flat[None],dim=1)
        opt=torch.optim.Adam([alpha],lr=.001,betas=(.5,.999))
        probability=alpha.softmax(0)
        center=(probability[:,None]*vectors).sum(0)
        objective=(probability*(vectors-center).norm(dim=1)).sum()+(probability*costs).sum()
        opt.zero_grad();objective.backward();opt.step()
        self.logits[ids]=alpha.detach()
        weights=probability.detach()
        out=[]
        for layer in range(len(deltas[0])):
            stacked=torch.stack([d[layer] for d in deltas])
            out.append(torch.einsum('i,i...->...',weights,stacked))
        return out,weights
