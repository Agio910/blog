"""Chapter 13: reproducible teaching examples and textbook exercises.
Python + NumPy + SciPy. Run this file; outputs are written beside it.
All experiments except the three textbook numerical exercises are synthetic.
"""
from pathlib import Path
import json, math, platform
import numpy as np
import scipy
from scipy import stats, special

OUT=Path(__file__).resolve().parent
result={}

def hwe_exact(obs):
    """Conditional probability-ordering two-sided exact HWE test."""
    aa,het,bb=map(int,obs);n=aa+het+bb;copies=2*aa+het
    hs=np.arange(copies%2,min(copies,2*n-copies)+1,2)
    nas=(copies-hs)//2;nbs=n-nas-hs
    logp=hs*np.log(2)-special.gammaln(nas+1)-special.gammaln(hs+1)-special.gammaln(nbs+1)
    probs=np.exp(logp-special.logsumexp(logp))
    observed=probs[hs==het][0]
    return float(probs[probs<=observed*(1+1e-10)].sum())

# 1. Textbook exercise 3: allele frequency estimated from the same sample.
obs=np.array([1787,3039,1303]);n=obs.sum();p=(2*obs[0]+obs[1])/(2*n)
freq=np.array([p*p,2*p*(1-p),(1-p)**2]);expected=n*freq
chisq=np.sum((obs-expected)**2/expected)
result['01_exercise3_hwe']={'N':int(n),'observed_frequencies':(obs/n).tolist(),'p_A':p,
 'expected_frequencies':freq.tolist(),'expected_counts':expected.tolist(),'chi2':chisq,
 'df':1,'p_asymptotic':stats.chi2.sf(chisq,1),'p_exact':hwe_exact(obs),
 'F':1-(obs[1]/n)/(2*p*(1-p))}
assert hwe_exact([1,2,1])>.99

# 2. Wahlund effect: both subpopulations are in HWE.
ps=np.array([.2,.8]);hs=np.mean(2*ps*(1-ps));ht=2*ps.mean()*(1-ps.mean())
result['02_wahlund']={'within_H':hs,'pooled_expected_H':ht,'pooled_deficit_F':1-hs/ht,
 'heterozygosity_Fst':(ht-hs)/ht}
assert np.isclose(1-hs/ht,.36)

# 3. SFS and Tajima D: n is the number of haploid sequences, not people.
def sfs_stats(xi):
    n=len(xi)+1;i=np.arange(1,n);S=int(np.sum(xi))
    a1=np.sum(1/i);a2=np.sum(1/i**2)
    b1=(n+1)/(3*(n-1));b2=2*(n*n+n+3)/(9*n*(n-1))
    c1=b1-1/a1;c2=b2-(n+2)/(a1*n)+a2/a1**2
    e1=c1/a1;e2=c2/(a1*a1+a2)
    k=np.sum(2*i*(n-i)*xi/(n*(n-1)));tw=S/a1
    variance=e1*S+e2*S*(S-1)
    return {'n':n,'S':S,'theta_W_region':tw,'pi_region':k,
     'Tajima_D':float((k-tw)/np.sqrt(variance)) if variance>0 else None,
     'theta_H_region':float(np.sum(2*i*i*xi/(n*(n-1))))}
single=np.zeros(9);single[0]=10
mid=np.zeros(9);mid[4]=10
high=np.zeros(9);high[8]=10
result['03_sfs']={key:sfs_stats(x) for key,x in [('singletons',single),('intermediate',mid),('high_derived',high)]}
assert result['03_sfs']['singletons']['Tajima_D']<0<result['03_sfs']['intermediate']['Tajima_D']
assert np.isclose(result['03_sfs']['singletons']['pi_region'],result['03_sfs']['high_derived']['pi_region'])

# 4. LD decay, without drift/mutation/migration/selection.
D=.4-.5*.5
result['04_ld']={'D0':D,'r_squared0':D*D/(.5**4),'D_after_10_c01':D*.9**10}

# 5. One generation of Wright-Fisher sampling, 2Ne gene copies.
rng=np.random.default_rng(1305);p=.3
result['05_drift']={}
for ne in [50,5000]:
    vals=rng.binomial(2*ne,p,size=100000)/(2*ne)
    result['05_drift'][str(ne)]={'empirical_variance':float(vals.var()),'theory_variance':p*(1-p)/(2*ne)}
    assert abs(vals.var()/(p*(1-p)/(2*ne))-1)<.03

# 6. Kingman coalescent, constant diploid Ne, no recombination.
rng=np.random.default_rng(1306);ne=10000;n_hap=10;k=np.arange(2,n_hap+1)
wait=rng.exponential(4*ne/(k*(k-1)),size=(50000,n_hap-1))
tmrca=wait.sum(1);total_branch=(wait*k).sum(1)
result['06_coalescent']={'mean_TMRCA_generations':float(tmrca.mean()),
 'theory_TMRCA':4*ne*(1-1/n_hap),'mean_total_branch':float(total_branch.mean()),
 'theory_total_branch':float(4*ne*np.sum(1/np.arange(1,n_hap)))}
assert abs(tmrca.mean()/result['06_coalescent']['theory_TMRCA']-1)<.02

# 7. Exercise 4: independent equal-variance and Welch two-sided t tests.
a=np.array([67.4,72.8,68.4,66.0,70.8,69.6,67.2,68.9,62.6])
b=np.array([60.6,66.6,64.9,61.8,61.7,67.2,56.7,62.4,61.3])
sp=((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2)
se=np.sqrt(sp*(1/len(a)+1/len(b)));diff=a.mean()-b.mean()
equal=stats.ttest_ind(a,b,equal_var=True);welch=stats.ttest_ind(a,b,equal_var=False)
critical=stats.t.ppf(.975,len(a)+len(b)-2)
result['07_exercise4_ttest']={'mean_MM':a.mean(),'mean_Mm':b.mean(),'variance_MM':a.var(ddof=1),
 'variance_Mm':b.var(ddof=1),'difference':diff,'pooled_t':float(equal.statistic),'df':16,
 'pooled_p':float(equal.pvalue),'welch_t':float(welch.statistic),'welch_df':float(welch.df),
 'welch_p':float(welch.pvalue),'mean_difference_95CI':[diff-critical*se,diff+critical*se]}

# 8. Exercise 5: unphased two-locus genotype independence; NOT haplotype chi2.
table=np.array([[45,10,5],[15,10,5],[5,0,5]])
stat,pv,df,E=stats.chi2_contingency(table,correction=False)
rng=np.random.default_rng(1308);B=100000
draws=stats.random_table.rvs(table.sum(1),table.sum(0),size=B,random_state=rng)
simstat=((draws-E)**2/E).sum(axis=(1,2))
mc=(1+np.sum(simstat>=stat-1e-12))/(B+1)
pa=pb=.75
Ds=np.array([(115+x)/200-pa*pb for x in range(11)])
result['08_exercise5_genotype_association']={'table':table.tolist(),'expected':E.tolist(),
 'chi2':stat,'df':df,'asymptotic_p':pv,'fixed_margin_MC_p':mc,'MC_replicates':B,
 'MC_standard_error_approx':math.sqrt(mc*(1-mc)/B),'p_A':pa,'p_B':pb,
 'D_possible_min_max':[float(Ds.min()),float(Ds.max())],
 'r2_possible_min_max':[float(Ds.min()**2/(pa*(1-pa)*pb*(1-pb))),float(Ds.max()**2/(pa*(1-pa)*pb*(1-pb)))]}
assert np.allclose(E.sum(0),table.sum(0)) and len(Ds)==11

# 9. Population confounding, with known residual variance=1 for clean z test.
rng=np.random.default_rng(1309);n=2000;z=np.repeat([0.,1.],n//2)
g=rng.binomial(2,np.where(z==0,.1,.9));y=3*z+rng.normal(size=n)
def known_variance_fit(X,y):
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    se=np.sqrt(np.diag(np.linalg.inv(X.T@X)))
    zz=beta[-1]/se[-1]
    return {'snp_beta':float(beta[-1]),'z_known_sigma1':float(zz),
     'two_sided_p_float':float(2*stats.norm.sf(abs(zz))),
     'log10_two_sided_p':float((np.log(2)+stats.norm.logsf(abs(zz)))/np.log(10))}
result['09_confounding']={'unadjusted':known_variance_fit(np.column_stack([np.ones(n),g]),y),
 'adjusted_known_group':known_variance_fit(np.column_stack([np.ones(n),z,g]),y),
 'true_direct_snp_effect':0}

# 10. GLS whitening at a fixed, positive-definite covariance matrix.
rng=np.random.default_rng(1310);n=40
R=rng.normal(size=(n,8));V=R@R.T/8+np.eye(n)*.5
X=np.column_stack([np.ones(n),rng.normal(size=n)]);y=rng.normal(size=n)
gls=np.linalg.solve(X.T@np.linalg.solve(V,X),X.T@np.linalg.solve(V,y))
L=np.linalg.cholesky(V);white=np.linalg.lstsq(np.linalg.solve(L,X),np.linalg.solve(L,y),rcond=None)[0]
result['10_gls']={'max_beta_difference':float(np.max(abs(gls-white)))}
assert np.allclose(gls,white)

# 11. ABC on an exactly solvable toy model: prior Uniform(0,1), Binomial(10,p), x=7.
# Exact matching is possible because the summary is discrete and sufficient.
rng=np.random.default_rng(1311);prior=rng.uniform(size=200000);obs=rng.binomial(10,prior)
accepted=prior[obs==7]
result['11_abc']={'simulations':len(prior),'accepted':len(accepted),'ABC_mean':float(accepted.mean()),
 'exact_Beta_8_4_mean':8/12,'ABC_95_interval':np.quantile(accepted,[.025,.975]).tolist(),
 'exact_95_interval':stats.beta.ppf([.025,.975],8,4).tolist()}
assert abs(accepted.mean()-8/12)<.01

# 12. Two-stage Pool-seq sampling. Equal DNA; independent diploid copies; no linkage.
rng=np.random.default_rng(1312);N=50;p=.5
result['12_poolseq']={}
for depth in [100,1000,10000]:
    pool=rng.binomial(2*N,p,size=100000)/(2*N)
    af=rng.binomial(depth,pool)/depth
    theoretical=p*(1-p)*(1/(2*N)+(1-1/(2*N))/depth)
    result['12_poolseq'][str(depth)]={'empirical_variance':float(af.var()),'theory_variance':theoretical,
     'incorrect_read_only_variance':p*(1-p)/depth,'infinite_depth_floor':p*(1-p)/(2*N)}
    assert abs(af.var()/theoretical-1)<.03

# 13. Genome-scan permutation threshold: iid simulated individuals, no covariates.
rng=np.random.default_rng(1313);n=80;m=100
G=rng.binomial(2,.3,size=(n,m)).astype(float);G-=G.mean(0);G/=np.sqrt((G**2).sum(0))
y=rng.normal(size=n);y-=y.mean();Y=np.column_stack([rng.permutation(y) for _ in range(2000)])
r2=(G.T@Y)**2/(Y**2).sum(0)
lod=-n/2*np.log10(1-r2)
result['13_scan_permutation']={'n':n,'markers':m,'permutations':2000,
 'pointwise_lod95_first_marker':float(np.quantile(lod[0],.95)),
 'genome_max_lod95':float(np.quantile(lod.max(0),.95))}
assert result['13_scan_permutation']['genome_max_lod95']>result['13_scan_permutation']['pointwise_lod95_first_marker']

payload={'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},'results':result}
(OUT/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'results.md').write_text('# 第十三章教学实验实际输出\n\n除教材习题3—5外，均为构造数据或模拟。\n\n```json\n'+json.dumps(payload,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
print(json.dumps(payload,ensure_ascii=False,indent=2))
