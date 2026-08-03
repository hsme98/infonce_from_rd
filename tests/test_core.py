import numpy as np
from itw_rd.core import fixed_output_rd, free_output_rd, kl_divergence, rd_objective, sinkhorn_log, weighted_row_kl
from itw_rd.generators import make_generic_channel


def test_sinkhorn_marginals():
    rng=np.random.default_rng(0); r=rng.dirichlet(np.ones(7)); c=rng.dirichlet(np.ones(5))
    J=sinkhorn_log(rng.normal(size=(7,5)),r,c,tol=1e-12).joint
    assert np.max(np.abs(J.sum(1)-r))<1e-9 and np.max(np.abs(J.sum(0)-c))<1e-9


def test_exact_kl_gap_identity():
    p=make_generic_channel(n=8,seed=3,matched=False)
    d=-(np.log(p.channel)-np.log(p.p_y)[None,:])
    V=sinkhorn_log(np.random.default_rng(5).normal(size=(8,8)),p.p_x,p.p_y,tol=1e-12).joint
    assert abs((rd_objective(V,d)-rd_objective(p.joint,d))-kl_divergence(V,p.joint))<1e-9


def test_fixed_output_y_gauge_invariance():
    p=make_generic_channel(n=9,seed=8,matched=False)
    d=-(np.log(p.channel)-np.log(p.p_y)[None,:]); g=np.random.default_rng(9).normal(size=9)
    J0=fixed_output_rd(p.p_x,p.p_y,d,tol=1e-12).joint
    J1=fixed_output_rd(p.p_x,p.p_y,d+2*g[None,:],tol=1e-12).joint
    assert np.max(np.abs(J0-J1))<1e-9


def test_free_output_can_move_under_y_gauge():
    p=make_generic_channel(n=7,seed=11,matched=False)
    d=-(np.log(p.channel)-np.log(p.p_y)[None,:]); g=np.linspace(-1,1,7)
    W0=free_output_rd(p.p_x,d,tol=1e-13).channel
    W1=free_output_rd(p.p_x,d+3*g[None,:],tol=1e-13).channel
    assert weighted_row_kl(p.p_x,W0,W1)>1e-5
