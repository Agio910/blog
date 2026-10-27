"""Chapter 14: reproducible teaching examples, not production bioinformatics tools.
Requires Python 3.10+, numpy and scipy. Run: python ch14_examples.py
All biological-looking inputs except the textbook sequence/knapsack are constructed.
Outputs are written beside this script. Each experiment has its own fixed seed.
"""
from pathlib import Path
from itertools import product
import json, platform
import numpy as np
import scipy
from scipy.special import logsumexp, expit, betaln, gammaln
from scipy.stats import beta

OUT=Path(__file__).resolve().parent

def safe_log(x):
    with np.errstate(divide='ignore'): return np.log(x)

def hmm(x, pi, A, B, end=None):
    """A[i,j]=P(next=j|current=i). Optional end makes A substochastic.
    Returns log likelihood, smoothed state/transition probabilities, Viterbi path.
    Fixed-length models use end=ones; terminating models require A.sum(1)+end=1.
    """
    T=len(x); K=len(pi); end=np.ones(K) if end is None else end
    lp,la,lb,le=map(safe_log,(pi,A,B,end))
    alpha=np.full((T,K),-np.inf);alpha[0]=lp+lb[:,x[0]]
    delta=alpha.copy();ptr=np.zeros((T,K),int)
    for t in range(1,T):
        alpha[t]=lb[:,x[t]]+logsumexp(alpha[t-1,:,None]+la,axis=0)
        scores=delta[t-1,:,None]+la;ptr[t]=scores.argmax(axis=0)
        delta[t]=lb[:,x[t]]+scores.max(axis=0)
    ll=logsumexp(alpha[-1]+le)
    backward=np.zeros((T,K));backward[-1]=le
    for t in range(T-2,-1,-1):
        backward[t]=logsumexp(la+lb[:,x[t+1]][None,:]+backward[t+1][None,:],axis=1)
    gamma=np.exp(alpha+backward-ll)
    xi=np.empty((T-1,K,K))
    for t in range(T-1):
        xi[t]=np.exp(alpha[t,:,None]+la+lb[:,x[t+1]][None,:]+backward[t+1][None,:]-ll)
    path=np.empty(T,int);path[-1]=(delta[-1]+le).argmax()
    for t in range(T-1,0,-1):path[t-1]=ptr[t,path[t]]
    assert np.allclose(gamma.sum(1),1)
    assert np.allclose(xi.sum((1,2)),1)
    assert np.allclose(xi.sum(2),gamma[:-1])
    return dict(log_likelihood=float(ll),gamma=gamma,xi=xi,path=path,
                viterbi_log_joint=float(np.max(delta[-1]+le)))

def gambling():
    a,b=6,4
    bob=float(np.exp(betaln(a,b+3)-betaln(a,b)))
    assert np.isclose(bob,1/11)
    return dict(posterior_beta=[a,b],mean_A=a/(a+b),CI95=beta.ppf([.025,.975],a,b),
                known_fair_Bob=.5**3,known_Bob_one_eighth=(1/8)**3,
                plug_in_Bob=(3/8)**3,posterior_predictive_Bob=bob,
                cube_of_posterior_mean_Bob=(b/(a+b))**3)

def markov_counts():
    s='CTTCATGTGAAAGCAGACGTAAGTCA';alphabet='ACGT'
    x=np.array([alphabet.index(c) for c in s]);counts=np.zeros((4,4),int)
    np.add.at(counts,(x[:-1],x[1:]),1)
    assert counts.sum()==len(s)-1
    return dict(alphabet=alphabet,counts=counts,row_totals=counts.sum(1),
                mle=counts/counts.sum(1,keepdims=True),laplace=(counts+1)/(counts.sum(1,keepdims=True)+4))

def splice_hmm():
    s='CTTCATGTGAAAGCAGACGTAAGTCA';x=np.array(['ACGT'.index(c) for c in s])
    pi=np.array([1.,0,0]);A=np.array([[.9,.1,0],[0,0,1],[0,0,.9]])
    B=np.array([[.25]*4,[.05,0,.95,0],[.4,.1,.1,.4]]);end=np.array([0,0,.1])
    assert np.allclose(A.sum(1)+end,1)
    r=hmm(x,pi,A,B,end);rows=[]
    for t in range(1,len(x)-1):
        z=np.array([0]*t+[1]+[2]*(len(x)-t-1))
        logp=safe_log(pi[z[0]])+sum(safe_log(B[z[i],x[i]]) for i in range(len(x)))
        logp+=sum(safe_log(A[z[i-1],z[i]]) for i in range(1,len(x)))+safe_log(end[z[-1]])
        if np.isfinite(logp): rows.append(dict(position=t+1,base=s[t],log_joint=float(logp),joint=float(np.exp(logp))))
    exact=logsumexp([v['log_joint'] for v in rows]);assert np.isclose(exact,r['log_likelihood'])
    for v in rows:
        v['posterior']=float(np.exp(v['log_joint']-exact))
        assert np.isclose(v['posterior'],r['gamma'][v['position']-1,1])
    rows.sort(key=lambda v:-v['posterior'])
    assert np.where(r['path']==1)[0].tolist()==[18]
    return dict(log_likelihood=r['log_likelihood'],likelihood=float(np.exp(exact)),
                viterbi_splice=19,viterbi_log_joint=r['viterbi_log_joint'],all_paths=rows,
                posterior_mass_top2=sum(v['posterior'] for v in rows[:2]),
                posterior_mass_top4=sum(v['posterior'] for v in rows[:4]))

def baum_welch():
    rng=np.random.default_rng(1404);T=350
    trueA=np.array([[.96,.04],[.08,.92]]);trueB=np.array([[.7,.2,.1],[.1,.2,.7]])
    z=np.zeros(T,int);x=np.zeros(T,int)
    for t in range(T):
        if t:z[t]=rng.choice(2,p=trueA[z[t-1]])
        x[t]=rng.choice(3,p=trueB[z[t]])
    pi=np.array([.5,.5]);A=np.array([[.8,.2],[.2,.8]]);B=np.array([[.5,.3,.2],[.2,.3,.5]])
    history=[]
    for step in range(60):
        r=hmm(x,pi,A,B);history.append(r['log_likelihood'])
        pi=r['gamma'][0];cnt=r['xi'].sum(0);A=cnt/cnt.sum(1,keepdims=True)
        cnt=np.stack([r['gamma'][x==b].sum(0) for b in range(3)],axis=1)
        B=cnt/cnt.sum(1,keepdims=True)
    r=hmm(x,pi,A,B);history.append(r['log_likelihood'])
    assert np.min(np.diff(history))>-1e-7
    return dict(T=T,iterations=60,initial_logL=history[0],final_logL=history[-1],
                minimum_increment=float(np.min(np.diff(history))),A=A,B=B,
                label_sensitive_state_accuracy=float(np.mean(r['path']==z)),history=history)

def bayesian_network():
    # 1=U, 0=D; P(b=U|a=U)=.8, P(c=U|a=U)=.6.
    joint={}
    for a,b,c,d in product([0,1],repeat=4):
        pa=.7;pb=.8 if a else .5;pc=.6 if a else .9
        pd={(1,1):1.,(1,0):.7,(0,1):.6,(0,0):.5}[b,c]
        joint[a,b,c,d]=np.prod([pa if a else 1-pa,pb if b else 1-pb,pc if c else 1-pc,pd if d else 1-pd])
    def prob(**kw):return sum(p for states,p in joint.items() if all(states['abcd'.index(k)]==v for k,v in kw.items()))
    assert np.isclose(sum(joint.values()),1)
    return dict(joint_UUDU=prob(a=1,b=1,c=0,d=1),P_d_U=prob(d=1),
                P_a_U_given_d_U=prob(a=1,d=1)/prob(d=1),
                P_b_U_given_a_U=prob(a=1,b=1)/prob(a=1),
                P_b_U_given_a_U_c_U_d_U=prob(a=1,b=1,c=1,d=1)/prob(a=1,c=1,d=1),
                P_b_U_given_a_U_c_D_d_U=prob(a=1,b=1,c=0,d=1)/prob(a=1,c=0,d=1))

def naive_bayes_dependence():
    # Y balanced; X is 80% accurate; X2 is an exact copy, not independent evidence.
    correct=.8;wrong=.2;posterior_naive=correct**2/(correct**2+wrong**2)
    brier=lambda p: correct*(1-p)**2+wrong*p**2
    nll=lambda p: -correct*np.log(p)-wrong*np.log(1-p)
    return dict(correct_posterior=.8,duplicated_feature_NB=posterior_naive,
                expected_Brier_correct=brier(.8),expected_Brier_duplicate=brier(posterior_naive),
                expected_NLL_correct=nll(.8),expected_NLL_duplicate=nll(posterior_naive))

def neural_network():
    rng=np.random.default_rng(1407);X=np.array([[0,0],[0,1],[1,0],[1,1]],float);y=np.array([[0],[1],[1],[0]],float)
    pars=[rng.normal(0,.6,(2,4)),np.zeros((1,4)),rng.normal(0,.6,(4,1)),np.zeros((1,1))]
    def calc(p):
        W,b,V,c=p;h=np.tanh(X@W+b);logits=h@V+c
        loss=np.mean(np.logaddexp(0,logits)-y*logits);dz=(expit(logits)-y)/len(X)
        dh=(dz@V.T)*(1-h*h)
        return float(loss),[X.T@dh,dh.sum(0,keepdims=True),h.T@dz,dz.sum(0,keepdims=True)],expit(logits)
    initial,grad,_=calc(pars);errors=[];eps=1e-5
    for p,g in zip(pars,grad):
        for idx in np.ndindex(p.shape):
            old=p[idx];p[idx]=old+eps;lp=calc(pars)[0];p[idx]=old-eps;lm=calc(pars)[0];p[idx]=old
            errors.append(abs((lp-lm)/(2*eps)-g[idx]))
    assert max(errors)<1e-7
    for _ in range(6000):
        loss,gr,_=calc(pars)
        for p,g in zip(pars,gr):p-=.3*g
    final,_,pred=calc(pars);assert np.all((pred>.5)==y)
    return dict(initial_BCE=initial,final_BCE=final,gradient_max_absolute_error=max(errors),XOR_probabilities=pred.ravel())

def convolution():
    x=np.array([1,2,-3,4,5]);w=np.array([2,1]);conv=np.correlate(x,w,mode='valid');relu=np.maximum(conv,0);pool=relu.reshape(2,2).max(1)
    assert conv.tolist()==[4,1,-2,13] and pool.tolist()==[4,13]
    pwm=np.array([[.7,.1,.1,.1],[.1,.1,.7,.1],[.1,.1,.1,.7]])
    seq='ACGTAGT';ix=np.array(['ACGT'.index(c) for c in seq]);onehot=np.eye(4)[ix];W=np.log(pwm/.25)
    scores=np.array([(onehot[i:i+3]*W).sum() for i in range(len(seq)-2)])
    direct=np.array([sum(W[j,ix[i+j]] for j in range(3)) for i in range(len(seq)-2)])
    assert np.allclose(scores,direct)
    rc=lambda s: s.translate(str.maketrans('ACGT','TGCA'))[::-1]
    def strand_score(s):
        a=np.array(['ACGT'.index(c) for c in s]);return max(sum(W[j,a[i+j]] for j in range(3)) for i in range(len(s)-2))
    invariant=lambda s:max(strand_score(s),strand_score(rc(s)))
    assert np.isclose(invariant(seq),invariant(rc(seq)))
    return dict(cross_correlation=conv,ReLU=relu,max_pool=pool,PWM_log_odds_scores=scores,
                reverse_complement_invariant_max=invariant(seq))

def gmm_em():
    rng=np.random.default_rng(1409);x=np.r_[rng.normal(-2,1,160),rng.normal(2,1,240)]
    mu=np.array([-.5,.5]);weights=np.array([.5,.5]);history=[]
    # Known unit component variance avoids variance-collapse degeneracy.
    for it in range(100):
        logs=np.log(weights)[None,:]-.5*(x[:,None]-mu)**2-.5*np.log(2*np.pi)
        lse=logsumexp(logs,axis=1);history.append(float(lse.sum()));r=np.exp(logs-lse[:,None])
        Nk=r.sum(0);weights=Nk/len(x);mu=(r*x[:,None]).sum(0)/Nk
    assert min(np.diff(history))>-1e-8
    return dict(known_variance=1,means=mu,weights=weights,initial_logL=history[0],final_logL=history[-1],
                minimum_increment=float(min(np.diff(history))))

def basic_rhat(draws):
    # Traditional split-Rhat ONLY; not the rank-normalized folded modern diagnostic.
    m,n=draws.shape;halves=np.concatenate([draws[:,:n//2],draws[:,-n//2:]],axis=0);n=halves.shape[1]
    W=halves.var(axis=1,ddof=1).mean();between=n*halves.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n-1)/n*W+between/n)/W))

def mcmc():
    rng=np.random.default_rng(1410);n=14000;burn=2000;z=np.array([-3.,-1.,1.,3.]);samples=[];accepted=np.zeros(4,int)
    # beta(6,4) in p, sampled in z=logit(p): includes dp/dz=p(1-p).
    def target(v):return -6*np.logaddexp(0,-v)-4*np.logaddexp(0,v)
    for t in range(n):
        proposal=z+rng.normal(0,1.1,4);ok=np.log(rng.random(4))<target(proposal)-target(z)
        z=np.where(ok,proposal,z);accepted+=ok
        if t>=burn:samples.append(expit(z.copy()))
    draws=np.array(samples).T
    def approx_ess(v):
        a=v-v.mean();n=len(a);ft=np.fft.rfft(a,n=2*n);cov=np.fft.irfft(ft*np.conjugate(ft))[:n]/np.arange(n,0,-1)
        rho=cov/cov[0];total=0.
        for k in range(1,n-1,2):
            pair=rho[k]+rho[k+1]
            if pair<=0:break
            total+=pair
        return n/(1+2*total)
    ess=sum(approx_ess(v) for v in draws);mean=float(draws.mean())
    assert abs(mean-.6)<.02
    return dict(chains=4,retained_per_chain=n-burn,acceptance=accepted/n,
                posterior_mean=mean,analytic_mean=.6,CI95=np.quantile(draws,[.025,.975]),
                analytic_CI95=beta.ppf([.025,.975],6,4),basic_split_Rhat=basic_rhat(draws),
                approximate_sum_within_chain_ESS=ess,approximate_MCSE=float(draws.std()/np.sqrt(ess)))

def gibbs_motif():
    seqs=['TACGTT','GACGAA','CACGGC'];X=np.array([['ACGT'.index(c) for c in s] for s in seqs]);K,L=X.shape;w=3;M=L-w+1
    configs=list(product(range(M),repeat=K));logs=[]
    def counts(pos,omit=-1):
        c=np.zeros((w,4))
        for k,a in enumerate(pos):
            if k!=omit:c[np.arange(w),X[k,a:a+w]]+=1
        return c
    for pos in configs:
        c=counts(pos);logs.append(float(np.sum(gammaln(4)-gammaln(4+K)+np.sum(gammaln(c+1),axis=1))))
    exact=np.exp(logs-logsumexp(logs));marg=np.zeros((K,M))
    for pos,p in zip(configs,exact):
        for k,a in enumerate(pos):marg[k,a]+=p
    rng=np.random.default_rng(1411);pos=rng.integers(0,M,K);hist=np.zeros((K,M));burn=1000;sweeps=13000
    for it in range(sweeps):
        for k in range(K):
            c=counts(pos,omit=k)+1
            lw=np.array([np.log(c[np.arange(w),X[k,a:a+w]]/(K-1+4)).sum() for a in range(M)])
            pos[k]=rng.choice(M,p=np.exp(lw-logsumexp(lw)))
        if it>=burn:hist[np.arange(K),pos]+=1
    empirical=hist/(sweeps-burn)
    return dict(sequences=seqs,width=w,configurations=len(configs),exact_position_marginals=marg,
                sampled_position_marginals=empirical,max_absolute_error=float(np.max(abs(marg-empirical))),
                exact_MAP_positions_1based=np.array(configs[np.argmax(exact)])+1,
                exact_MAP_probability=float(exact.max()))

def alignment():
    a,b='ACGT','AGT';gap=-2;F=np.zeros((len(a)+1,len(b)+1),int)
    F[:,0]=np.arange(len(a)+1)*gap;F[0]=np.arange(len(b)+1)*gap
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):F[i,j]=max(F[i-1,j-1]+(2 if a[i-1]==b[j-1] else -1),F[i-1,j]+gap,F[i,j-1]+gap)
    i,j=len(a),len(b);aa=[];bb=[]
    while i or j:
        if i and j and F[i,j]==F[i-1,j-1]+(2 if a[i-1]==b[j-1] else -1):aa.append(a[i-1]);bb.append(b[j-1]);i-=1;j-=1
        elif i and F[i,j]==F[i-1,j]+gap:aa.append(a[i-1]);bb.append('-');i-=1
        else:aa.append('-');bb.append(b[j-1]);j-=1
    assert F[-1,-1]==4
    return dict(score=int(F[-1,-1]),aligned_a=''.join(aa[::-1]),aligned_b=''.join(bb[::-1]),DP=F)

def knapsack():
    w=np.array([15,3,2,5,9,20]);v=np.array([15,7,10,5,8,17]);capacity=30
    choices=np.array(list(product([0,1],repeat=6)));weights=choices@w;values=choices@v
    fitness=np.where(weights<=capacity,values,0);best=fitness.max();solutions=choices[fitness==best]
    dp=np.zeros(capacity+1,int)
    for wi,vi in zip(w,v):
        for cap in range(capacity,wi-1,-1):dp[cap]=max(dp[cap],dp[cap-wi]+vi)
    assert dp[-1]==best
    rng=np.random.default_rng(1413);pop=rng.integers(0,2,(24,6));history=[]
    for gen in range(40):
        fit=np.where(pop@w<=capacity,pop@v,0);elite=pop[fit.argmax()].copy();history.append(int(fit.max()))
        nxt=[elite]
        while len(nxt)<len(pop):
            parents=[]
            for _ in range(2):
                ids=rng.integers(0,len(pop),3);parents.append(pop[ids[np.argmax(fit[ids])]])
            cut=rng.integers(1,6);child=np.r_[parents[0][:cut],parents[1][cut:]].copy()
            child^=(rng.random(6)<1/6).astype(int);nxt.append(child)
        pop=np.array(nxt)
    assert np.all(np.diff(history)>=0)
    return dict(exact_optimum=int(best),optimal_bitstrings=[''.join(map(str,s)) for s in solutions],
                optimal_weights=solutions@w,DP_optimum=int(dp[-1]),GA_best=history[-1],GA_history=history,
                roulette_probabilities=np.array([28,23,12,34])/97)

def leakage():
    # A null example: selecting one of many random features using all labels leaks test labels.
    rng=np.random.default_rng(1414);n=160;d=12000;X=rng.normal(size=(n,d));y=rng.choice([-1.,1.],n)
    train=np.arange(80);test=np.arange(80,160)
    def select(rows):
        corr=X[rows].T@y[rows];j=int(np.argmax(abs(corr)));return j,np.sign(corr[j])
    j_all,s_all=select(np.arange(n));j_train,s_train=select(train)
    accuracy=lambda j,s:float(np.mean(np.sign(X[test,j])*s==y[test]))
    return dict(n=n,features=d,null_truth='X independent of Y',
                leaked_test_accuracy=accuracy(j_all,s_all),proper_test_accuracy=accuracy(j_train,s_train))

def calibration():
    # Four groups have known event probabilities; overconfident forecasts preserve ranking.
    true=np.array([.1,.3,.7,.9]);forecast=expit(3*np.log(true/(1-true)))
    brier=lambda p:float(np.mean(true*(1-p)**2+(1-true)*p**2))
    return dict(true_probabilities=true,overconfident_probabilities=forecast,
                Bayes_Brier=brier(true),overconfident_Brier=brier(forecast),
                ranking_unchanged=bool(np.array_equal(np.argsort(true),np.argsort(forecast))))

def protein_hmm():
    # Fully specified toy answer to exercise 3. H/P are composition states, not ground-truth structures.
    alphabet='ACDEFGHIKLMNPQRSTVWY';hydrophobic=set('AVILMFWYC')
    B=np.array([[.08 if a in hydrophobic else .28/11 for a in alphabet],
                [.02 if a in hydrophobic else .82/11 for a in alphabet]])
    A=np.array([[.95,.05],[.1,.9]]);pi=np.array([.3,.7])
    seq='DEKRSTAVILMFWYCVILMDEKRST';x=np.array([alphabet.index(a) for a in seq])
    assert np.allclose(B.sum(1),1)
    r=hmm(x,pi,A,B)
    return dict(alphabet=alphabet,sequence=seq,A=A,B=B,initial=pi,
                log_likelihood=r['log_likelihood'],state_path=''.join('HP'[i] for i in r['path']),
                posterior_H=r['gamma'][:,0])

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x))

if __name__=='__main__':
    funcs=[gambling,markov_counts,splice_hmm,baum_welch,bayesian_network,naive_bayes_dependence,
           neural_network,convolution,gmm_em,mcmc,gibbs_motif,alignment,knapsack,leakage,calibration,protein_hmm]
    results={}
    for i,f in enumerate(funcs,1):
        results[f'{i:02}_{f.__name__}']=f();print(f'PASS {i:02} {f.__name__}',flush=True)
    payload=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),results=results)
    s=json.dumps(payload,ensure_ascii=False,indent=2,default=serial,allow_nan=False)
    (OUT/'results.json').write_text(s,encoding='utf-8')
    (OUT/'results.md').write_text('# 第十四章教学实验实际输出\n\n教材序列、概率表及背包参数之外的数据均为构造或模拟。\n\n```json\n'+s+'\n```\n',encoding='utf-8')
    print('Saved results.json and results.md')
