"""Publication figure style and shared ten-method comparisons."""
from pathlib import Path
import itertools
import shutil
from statistics import mean

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.optimization import METHODS
from fedperg.paths import ROOT, RESULTS
from reporting.summary_inputs import mean_metric

PAPER = ROOT.parents[1] / 'paper' / 'INS' / 'R1'
SOURCES = ['cifar10', 'cifar100', 'officehome']
REGIMES = ['label_skew', 'quantity_skew', 'compound']
REGIME_NAMES = ['Label skew', 'Quantity skew', 'Compound']
NAMES = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'officehome': 'Office-Home'}
COLORS = dict(zip(METHODS, ['#FFAA53','#50CC55','#3399FF','#6666FF','#9933FF',
                           '#00DDDD','#4D4D4D','#B8860B','#D65DB1','#FF6666']))

def method_name(name):
    return {'FedAvgM-Gate': 'AvgM+Gate', 'Router': 'Router-only', 'SpatialOnly': 'Spatial-only',
            'TemporalOnly': 'Temporal-only', 'RandomInput': 'Random input',
            'Permuted': 'Permuted field', 'NoResidual': 'No residual'}.get(name, name)

def value(record, metric='convergence_auc'):
    return 100 * record['final'][metric]

def figures_setup():
    mpl.rcParams.update({'font.family':'Times New Roman','font.size':19,'axes.labelsize':19,
                         'axes.titlesize':19,'xtick.labelsize':19,'ytick.labelsize':19,
                         'legend.fontsize':19,'pdf.fonttype':42,'axes.linewidth':.9})

def decorate(ax,j,title):
    ax.set_title(title,pad=7)
    ax.text(-.07,1.04,f'({chr(97+j)})',transform=ax.transAxes,fontweight='bold')
    ax.grid(axis='y',alpha=.24,linewidth=.7)
    ax.spines[['top','right']].set_visible(False)

def save(fig,stem):
    path=RESULTS/'figures'/f'{stem}.pdf'
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,bbox_inches='tight',pad_inches=.06)
    if (PAPER/'paper.tex').is_file():
        (PAPER/'figures').mkdir(parents=True,exist_ok=True)
        (PAPER/'response_assets').mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,PAPER/'figures'/path.name)
        fig.savefig(PAPER/'response_assets'/f'{stem}.png',dpi=200,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)

def convergence(ix, seeds=range(220,223)):
    fig,axes=plt.subplots(3,3,figsize=(15.3,9.8),sharex=True)
    for i,ds in enumerate(SOURCES[:3]):
        for j,regime in enumerate(REGIMES):
            ax=axes[i,j]
            for k,method in enumerate(METHODS):
                runs=[ix[ds,regime,s,method] for s in seeds]
                rounds=np.array([h['round'] for h in runs[0]['history']])
                acc=np.array([[100*h['accuracy'] for h in r['history']] for r in runs])
                mu=acc.mean(0);sd=acc.std(0,ddof=1)
                ax.plot(rounds,mu,color=COLORS[method],lw=2.5 if method=='FedPERG' else 1.55,
                        linestyle=['-','--','-.',':'][k%4] if method!='FedPERG' else '-')
                ax.fill_between(rounds,mu-sd,mu+sd,color=COLORS[method],alpha=.11,linewidth=0)
                ax.plot(rounds[-1], mean_metric(runs, 'accuracy'), 'o',
                        color=COLORS[method], ms=3, zorder=5)
            decorate(ax,3*i+j,f'{NAMES[ds]} · {REGIME_NAMES[j]}')
            if j==0:ax.set_ylabel('Accuracy (%) ↑')
            if i==2:ax.set_xlabel('Communication round')
    handles=[Line2D([],[],color=COLORS[m],lw=2,label=method_name(m),
                    linestyle=['-','--','-.',':'][k%4] if m!='FedPERG' else '-') for k,m in enumerate(METHODS)]
    fig.legend(handles=handles,ncol=5,loc='lower center',bbox_to_anchor=(.5,.0),frameon=False)
    fig.subplots_adjust(left=.08,right=.995,top=.94,bottom=.17,wspace=.28,hspace=.32)
    save(fig,'r1_optimized_convergence')

def performance_figure(ix, seeds=range(220,223)):
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
    fig,axes=plt.subplots(2,3,figsize=(15.7,9.2))
    settings=list(itertools.product(SOURCES[:3],REGIMES))
    run_groups={m:[[ix[d,g,s,m] for s in seeds] for d,g in settings] for m in METHODS}
    ax=axes.flat[0]
    for j,m in enumerate(METHODS):
        vals=[value(r,'worst20_accuracy') for group in run_groups[m] for r in group]
        violin=ax.violinplot(vals,positions=[j],widths=.8,showextrema=False,showmedians=True)
        violin['bodies'][0].set(facecolor=COLORS[m],alpha=.55)
        violin['cmedians'].set(color=COLORS[m])
    ax.set_xticks([])
    ax.set_ylabel('Worst 20% accuracy (%) ↑');decorate(ax,0,'Tail-client distribution')

    ax=axes.flat[1]
    eces=np.array([[mean(value(r,'ece') for r in group) for group in run_groups[m]] for m in METHODS])
    im=ax.imshow(eces,aspect='auto',vmin=0,cmap=LinearSegmentedColormap.from_list('white_red',['#FFFFFF','#FF4F4F']))
    ax.set_xticks([1,4,7],['C10','C100','OH-65'])
    ax.set_yticks(range(10),[method_name(m) for m in METHODS])
    for j in [2.5,5.5]:ax.axvline(j,color='.6',lw=.7)
    fig.colorbar(im,ax=ax,fraction=.05,pad=.03,label='ECE (%) ↓');decorate(ax,1,'Calibration by setting')

    ax=axes.flat[2]
    metrics=['convergence_auc','accuracy','worst20_accuracy','ece']
    rank=np.zeros((len(METHODS),4))
    from scipy.stats import rankdata
    for j,metric in enumerate(metrics):
        matrix=np.array([[mean_metric(group,metric) for group in run_groups[m]] for m in METHODS])
        rank[:,j]=rankdata(matrix if metric=='ece' else -matrix,axis=0).mean(1)
    for j,m in enumerate(METHODS):ax.plot(range(4),rank[j],'.-',color=COLORS[m],lw=1.7)
    ax.invert_yaxis();ax.set_xticks(range(4),['AUC','Accuracy','Tail','ECE'],rotation=27,ha='right')
    ax.set_ylabel('Mean rank (1 = best)');decorate(ax,2,'Metric-dependent rankings')

    ax=axes.flat[3]
    for method in METHODS[:-1]:
        gains=np.sort([value(ix[d,g,s,'FedPERG'])-value(ix[d,g,s,method]) for d,g in settings for s in seeds])
        ax.step(gains,np.arange(1,len(gains)+1)/len(gains),where='post',color=COLORS[method],lw=1.7)
    ax.axvline(0,ls='--',color='.4');ax.set(xlabel='FedPERG − baseline AUC (pp) ↑',ylabel='Empirical fraction')
    decorate(ax,3,'Paired AUC differences')

    ax=axes.flat[4]
    for method in METHODS:
        for d in SOURCES[:3]:
            runs=[ix[d,g,s,method] for g in REGIMES for s in seeds]
            ax.scatter(mean_metric(runs,'ece'),mean_metric(runs,'accuracy'),
                       color=COLORS[method],s=43 if method=='FedPERG' else 28,alpha=.85)
    ax.set(xlabel='ECE (%) ↓',ylabel='Final accuracy (%) ↑');decorate(ax,4,'Accuracy and calibration')

    ax=axes.flat[5]
    for j,method in enumerate(METHODS[:-1]):
        dif=np.array([value(ix[d,g,s,'FedPERG'])-value(ix[d,g,s,method]) for d,g in settings for s in seeds])
        negative=(dif<-.01).sum();ties=(abs(dif)<=.01).sum();positive=(dif>.01).sum()
        ax.barh(j,-negative,color='#007FFF',alpha=.65)
        ax.barh(j,ties,color='#BBBBBB')
        ax.barh(j,positive,left=ties,color='#FF6666',alpha=.8)
    ax.axvline(0,color='.4',lw=.8);ax.invert_yaxis()
    ax.set_yticks(range(9),[method_name(m) for m in METHODS[:-1]])
    ax.set_xlabel('Paired runs: losses ← 0 → wins / ties')
    ax.xaxis.label.set_wrap(True)
    decorate(ax,5,'Win/tie/loss by baseline')
    handles=[Line2D([],[],color=COLORS[m],lw=2,label=method_name(m)) for m in METHODS]
    fig.legend(handles=handles,ncol=5,loc='lower center',bbox_to_anchor=(.5,.098),frameon=False)
    violins=[Patch(facecolor=COLORS[m],alpha=.55,label=method_name(m)) for m in METHODS]
    for handle, method in zip(violins, METHODS):handle.set_label('(a) '+method_name(method))
    fig.legend(handles=violins,ncol=5,loc='lower center',bbox_to_anchor=(.5,.003),frameon=False)
    fig.subplots_adjust(left=.085,right=.987,top=.94,bottom=.30,wspace=.46,hspace=.6)
    save(fig,'r1_optimized_performance')
