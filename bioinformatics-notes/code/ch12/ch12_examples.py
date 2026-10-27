"""Chapter 12 teaching experiments. Python 3 + NumPy; no downloads.

Run: python ch12_examples.py
Writes results.json and results.md beside this file. These are synthetic
experiments, not implementations or benchmarks of production tools.
"""
from pathlib import Path
import json, math, platform
import numpy as np

OUT = Path(__file__).resolve().parent
results = {}

# 1. Molecule deduplication: deliberately excludes sequencing/UMI errors.
records = [('c1','g1','AAAA')]*5 + [('c1','g1','AAAT')]*2 + [('c2','g1','AAAA')]*3
results['01_umi'] = {'reads':len(records), 'molecules':len(set(records)),
                     'c1_g1_molecules':len({u for c,g,u in records if c=='c1' and g=='g1'})}
assert len(set(records)) == 3

# 2. Symmetric matrix scaling on a dense positive matrix, not full ICE.
n=24
i,j=np.indices((n,n))
structure=(1+4*((i//8)==(j//8)))/(1+abs(i-j))
bias=np.linspace(.3,3,n)
C=structure*bias[:,None]*bias[None,:]
w=np.ones(n)
for step in range(10000):
    B=w[:,None]*C*w[None,:]
    row=B.sum(axis=1)
    if np.max(abs(row-1))<1e-10:break
    w/=np.sqrt(row)
assert np.allclose(B,B.T) and np.max(abs(B.sum(1)-1))<1e-9
results['02_balance']={'iterations':step,'raw_row_max_min':float(C.sum(1).max()/C.sum(1).min()),
 'balanced_row_max_min':float(B.sum(1).max()/B.sum(1).min()),
 'caution':'Equal row sums verify scaling only, not recovery of true biological structure.'}

# 3. Rare-cell detection uses the exact binomial probability for >=1.
p=.001;q=.95
N=math.ceil(math.log1p(-q)/math.log1p(-p))
assert 1-(1-p)**N>=q and 1-(1-p)**(N-1)<q
results['03_rare_cell']={'p':p,'target_probability':q,'N_min':N,'probability':1-(1-p)**N}

# 4. Null simulation with known Gaussian donor and cell variances.
# Group mean has variance donor_var/donors + cell_var/(donors*cells).
# Generate group means directly: this is algebraically equivalent to
# generating every cell in the stated Gaussian hierarchical model.
rng=np.random.default_rng(20261023)
trials=20000;donors=6;cells=500;dv=1.;cv=1.
true_var=2*(dv/donors+cv/(donors*cells))
diff=rng.normal(0,np.sqrt(true_var),trials)
naive_z=diff/np.sqrt(2*(dv+cv)/(donors*cells))
correct_z=diff/np.sqrt(true_var)
results['04_pseudoreplication']={'null_trials':trials,'donors_per_group':donors,'cells_per_donor':cells,
 'naive_rejection_fraction':float(np.mean(abs(naive_z)>1.959963984540054)),
 'known_variance_hierarchical_rejection_fraction':float(np.mean(abs(correct_z)>1.959963984540054))}
# Stochastic scientific check with tolerance, not an exact target count.
assert .04 < results['04_pseudoreplication']['known_variance_hierarchical_rejection_fraction'] < .06

# 5. Ridge primal/dual equivalence on centered training features.
rng=np.random.default_rng(12)
M=rng.normal(size=(40,90));M-=M.mean(0)
y=rng.normal(size=40);y-=y.mean();lam=5.
bp=np.linalg.solve(M.T@M+lam*np.eye(90),M.T@y)
bd=M.T@np.linalg.solve(M@M.T+lam*np.eye(40),y)
c=90.;G=M@M.T/c
u_g=G@np.linalg.solve(G+(lam/c)*np.eye(40),y)
results['05_ridge_gblup']={'max_beta_difference':float(np.max(abs(bp-bd))),
 'max_prediction_difference':float(np.max(abs(M@bp-u_g)))}
assert np.allclose(bp,bd) and np.allclose(M@bp,u_g)

# 6. Extreme family-tag example: proves why split target matters.
# One-hot family tags are NOT SNPs. Phenotype is family effect + noise.
rng=np.random.default_rng(13)
families=np.repeat(np.arange(40),8)
X=np.eye(40)[families]
family_effect=rng.normal(size=40)
y=family_effect[families]+rng.normal(scale=.2,size=len(families))
def prediction_metrics(train,test):
    mean=y[train].mean()
    beta=np.linalg.solve(X[train].T@X[train]+.01*np.eye(40),X[train].T@(y[train]-mean))
    pred=mean+X[test]@beta
    mse=float(np.mean((pred-y[test])**2));null=float(np.mean((mean-y[test])**2))
    return {'rmse':mse**.5,'skill_vs_train_mean':1-mse/null}
test=np.zeros(len(y),bool)
for family in range(40):
    ix=np.flatnonzero(families==family);test[rng.choice(ix,2,replace=False)]=True
random_metrics=prediction_metrics(~test,test)
group_test=families>=30
group_metrics=prediction_metrics(~group_test,group_test)
results['06_split_target']={'random_individual_split':random_metrics,'unseen_family_split':group_metrics,
 'caution':'Constructed family-identity example, not an empirical estimate of genomic prediction performance.'}
assert random_metrics['skill_vs_train_mean']>.8 and abs(group_metrics['skill_vs_train_mean'])<1e-12

# 7. Ribo frame convention: local transcript coordinates, no splice handling.
start=100
psites=start+3*np.arange(30)
correct=np.bincount((psites-start)%3,minlength=3)
wrong=np.bincount((psites+1-start)%3,minlength=3)
def delta_te(rA,rB,mA,mB):return math.log2(rB/rA)-math.log2(mB/mA)
results['07_ribo']={'correct_frame_counts':correct.tolist(),'offset_plus_one_frame_counts':wrong.tolist(),
 'both_double_delta_log2_TE':delta_te(200,400,100,200),
 'ribo_quadruples_rna_doubles_delta_log2_TE':delta_te(200,800,100,200)}
assert delta_te(200,400,100,200)==0 and delta_te(200,800,100,200)==1

# 8. Tiny foreground: all-background prediction has high accuracy, zero Dice.
truth=np.zeros((100,100),bool);truth[45:55,45:55]=True
pred=np.zeros_like(truth)
intersection=np.sum(pred&truth);denom=pred.sum()+truth.sum()
results['08_segmentation']={'pixel_accuracy':float(np.mean(pred==truth)),
 'dice':float(2*intersection/denom),'foreground_recall':float(intersection/truth.sum())}
assert results['08_segmentation']['pixel_accuracy']==.99 and results['08_segmentation']['dice']==0

# 9. Valid convolution is a cross-correlation in typical neural-net APIs.
x=np.array([0.,1.,2.,0.,1.]);kernel=np.array([1.,2.,1.])
conv=np.array([np.dot(x[j:j+3],kernel) for j in range(len(x)-2)])
results['09_convolution']={'input':x.tolist(),'kernel':kernel.tolist(),'output':conv.tolist(),'length':len(conv)}
assert np.array_equal(conv,[4.,5.,3.])

# 10. First-order Hill repression: Euler numerical solution vs analytic limit.
alpha=10.;delta=2.;u=3.;K=1.;h=2.
production=alpha/(1+(u/K)**h);steady=production/delta
value=0.;dt=.001
for _ in range(10000):value+=dt*(production-delta*value)
results['10_dynamics']={'analytic_steady_state':steady,'euler_at_t10':value,'absolute_error':abs(value-steady)}
assert abs(value-steady)<1e-7

payload={'environment':{'python':platform.python_version(),'numpy':np.__version__},'results':results}
(OUT/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 第十二章教学实验实际输出','',
       '这些结果来自本目录脚本的本地运行；全部为构造数据或模拟，不是实际生物数据基准。','',
       '```json',json.dumps(payload,ensure_ascii=False,indent=2),'```','']
(OUT/'results.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(payload,ensure_ascii=False,indent=2))
