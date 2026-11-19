"""Mathematical and data-access invariants of the revised evidence protocol."""
import numpy as np
import torch
from fedperg.aggregators import PERGAggregator
from fedperg.recent_baselines import dawid_skene_weights


def test_certificate_and_matched_random_control():
    p=torch.tensor([.15,.2,.3,.35]);reward=torch.tensor([.2,-.1,.7,.3])
    gen=torch.Generator().manual_seed(123)
    deltas=[[torch.randn(13,generator=gen),torch.randn(3,generator=gen)] for _ in range(4)]
    gradients=[torch.randn(13,generator=gen),torch.randn(3,generator=gen)]
    outputs=[]
    for control in ['informative','permuted']:
        agg=PERGAggregator([torch.zeros(13),torch.zeros(3)],48,4,.003,1,.3,
                          torch.device('cpu'),'lite_paired_gate_bank_selector')
        agg.utility_gradient=gradients;agg.evidence_control=control
        out=agg.aggregate(deltas,p,reward,1);outputs.append(out)
        assert max(agg.utility_linear_effects)<=1e-7
    a,b=outputs
    torch.testing.assert_close(a.weights.sort(0).values,b.weights.sort(0).values)
    for r in [0,1,3,5]:
        for x,y in zip(a.expert_deltas[r],b.expert_deltas[r]):torch.testing.assert_close(x,y)


def test_mirror_identity():
    p=torch.tensor([.1,.2,.3,.4],dtype=torch.float64)
    u=torch.tensor([.4,-.7,.3,.8],dtype=torch.float64);temperature=.9;eta=1.2
    m=(p.log()-eta*u/temperature).softmax(0)
    symmetric=(m*(m/p).log()).sum()+(p*(p/m).log()).sum()
    torch.testing.assert_close((u*(m-p)).sum(),-temperature/eta*symmetric)


def test_projected_field_certificate_and_router_counterfactual():
    torch.manual_seed(819)
    deltas=[[torch.randn(23)] for _ in range(5)];p=torch.tensor([.1,.15,.2,.25,.3])
    gradient=[torch.randn(23)]
    outs=[]
    for variant in ['lite_paired_gate_bank_selector','lite_paired_gate_bank_router_control']:
        agg=PERGAggregator([torch.zeros(23)],48,4,.003,1,.3,torch.device('cpu'),variant)
        agg.utility_gate_mode='field';agg.utility_strength=.1;agg.utility_gradient=gradient
        out=agg.aggregate(deltas,p,torch.randn(5),1);outs.append(out)
        assert max(agg.utility_linear_effects)<=1e-7
    # Source-prior randomness cannot affect any Router action at this state.
    for index in [0,1,3,5]:torch.testing.assert_close(outs[0].expert_deltas[index][0],outs[1].expert_deltas[index][0])
    for r,e in [(1,2),(3,4),(5,6)]:torch.testing.assert_close(outs[1].expert_deltas[r][0],outs[1].expert_deltas[e][0])


def test_dawid_skene_matches_released_uniform_prior_loops():
    pred=np.array([[0,0,1],[0,0,0],[1,1,0],[1,1,1],[2,2,1],[2,2,2]])
    # Literal equations of the released algorithm, independently implemented.
    n,k=pred.shape;c=3;q=np.stack([(pred==v).mean(1) for v in range(c)],1)+1e-10
    q/=q.sum(1,keepdims=True)
    def cm_from(q):
        cm=np.zeros((k,c,c))
        for j in range(k):
            for i in range(n):
                for label in range(c):cm[j,label,pred[i,j]]+=q[i,label]
        mass=cm.sum(2,keepdims=True);mass[mass==0]=1
        return cm/mass
    cm=cm_from(q)
    for _ in range(500):
        previous=q.copy()
        for i in range(n):
            for label in range(c):q[i,label]=np.prod([cm[j,label,pred[i,j]] for j in range(k)])
            q[i]=q[i]/q[i].sum() if q[i].sum()>0 else np.ones(c)/c
        cm=cm_from(q)
        if np.max(np.abs(q-previous))<1e-5:break
    expected=np.diagonal(cm,axis1=1,axis2=2).mean(1);expected/=expected.sum()
    weights,_=dawid_skene_weights(pred,c)
    np.testing.assert_allclose(weights,expected,rtol=1e-9,atol=1e-9)
