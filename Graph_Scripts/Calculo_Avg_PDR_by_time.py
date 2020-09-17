#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:12:25 2019

@author: emanuel
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from sklearn import metrics
import scipy.stats
import statistics
import csv
import re
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import glob
from collections import Counter

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


def jfi(vetor): #Fairness Index
    sum0=0
    sum1=0
    jfi = [None] * 25
    for i in range(25):
        sum0 += vetor[i]
        sum1 += pow(vetor[i],2)
        jfi[i] = pow(sum0,2)/((i+1)*sum1)
    return jfi

def timeMean(NRuns, time, metric):
    unique_times = sorted(Counter(time).keys())
    metric_by_time = {x: 0 for x in unique_times}
    for i in range(len(time)):
        metric_by_time[time[i]] = metric_by_time[time[i]] + metric[i]
    for x in unique_times:
        metric_by_time[x] = metric_by_time[x] / NRuns
    return metric_by_time

def getScenarioParameters(scenario):
    tokens = scenario.split('/')
    if tokens[0] == 'percept':
        algo = 'LTE+PERCEPT'
        nUABS = tokens[3]
        nUE = tokens[4]
    else:
        if tokens[2] == 'scen=4':
            algo = 'LTE+UOS'
        else:
            algo = 'LTE'
        nUABS = tokens[4]
        nUE = tokens[5]
    nUABS = int(nUABS.replace('nUABS=', ''))
    nUE = int(nUE.replace('nUE=',''))
    #print('nUABS:{} nUE:{} algo:{}'.format(nUABS, nUE, algo))
    return algo, nUABS, nUE

def removeHeader(pltFile):
    lines = pltFile.readlines()
    return lines[6:-1]

prefix = ''
#prefix = '/run/user/1000/gvfs/sftp:host=gercom.ddns.net,port=8372'
root_path = '/home/hao123/Downloads/ieee_article_simulations/'

scenarios_path = [
                #small cells 100 UEs
                'uos+lte/enableSCs=true/scen=3/nENB=4/nUABS=0/nUE=100/*/', #LTE
                'uos+lte/enableSCs=true/scen=4/nENB=4/nUABS=8/nUE=100/*/', #UOS 8UABS
                'percept/enableSCs=true/nENB=4/nUABS=8/nUE=100/*/', #PERCEPT 8UABS

                #small cells 200 UEs
                'uos+lte/enableSCs=true/scen=3/nENB=4/nUABS=0/nUE=200/*/', #LTE
                'uos+lte/enableSCs=true/scen=4/nENB=4/nUABS=8/nUE=200/*/', #UOS 8UABS
                'percept/enableSCs=true/nENB=4/nUABS=8/nUE=200/*/', #PERCEPT 8UABS

                #small cells 300 UEs
                'uos+lte/enableSCs=true/scen=3/nENB=4/nUABS=0/nUE=300/*/', #LTE
                'uos+lte/enableSCs=true/scen=4/nENB=4/nUABS=8/nUE=300/*/', #UOS 8UABS
                'percept/enableSCs=true/nENB=4/nUABS=8/nUE=300/*/', #PERCEPT 8UABS
                ]

No_scenarios = len(scenarios_path)
data_dict = dict()
for scenario_number in range(0,No_scenarios):
    algo, nUABS, nUE = getScenarioParameters(scenarios_path[scenario_number])
    scenario = prefix + root_path + scenarios_path[scenario_number]
    plts = glob.glob(scenario + 'PDR.plt')
    No_runs = len(plts)
    data = []
    #print('y{}: {} {}'.format(scenario_number+1, scenarios_path[scenario_number], No_runs))
    for run_number in range(0,No_runs):
        pltFile = open(plts[run_number], 'r')
        csv_data = removeHeader(pltFile)
        data.extend(list((int(run_number), int(time), float(Throughput)) for time, Throughput in csv.reader(csv_data, delimiter= ' ')))
    data = np.array(data)
    run, time,Throughput = data.T
    mean_by_time = timeMean(No_runs, time, Throughput)
    data_dict['y{0}'.format(scenario_number+1)] = list(mean_by_time.values())
data_dict['time'] = list(mean_by_time)

df = DataFrame(data_dict)
print(df)

# Create an axes instance
fig, (ax1, ax2, ax3)=plt.subplots(3)

#Color palette used in master thesis
uniquefuckingcolors  = ["#eb3b5a", "#7c7f9e", "#3c4563","#a55eea","#8854d0","#26de81","#20bf6b"] #with LTE red

legends = []
time_limit = data_dict['time'][-1]
# multiple line plot

#---------------------100 Users SMALL CELLS------------------------------
ax1.set(ylim=(0,100), xlim=(0,time_limit))
ax1.set(xlabel='Simulation time (s)', ylabel='PDR (%)')
ax1.xaxis.get_label().set_fontsize(14)
ax1.yaxis.get_label().set_fontsize(14)
ax1.plot( 'time', 'y1', '', data=df, marker='', color= uniquefuckingcolors[0], linewidth=2,linestyle='dashed', label='LTE')
ax1.plot( 'time', 'y2', '', data=df, marker='', color= uniquefuckingcolors[1], linewidth=2,label='LTE + UOS')
ax1.plot( 'time', 'y3', '', data=df, marker='', color= uniquefuckingcolors[2], linewidth=2,label='LTE + Percept')
# legends.append(ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2))
legends.append(ax1.legend(loc='upper center',bbox_to_anchor=(0.5, 0.27), ncol=2))

#---------------------200 Users SMALL CELLS------------------------------
ax2.set(ylim=(0,100), xlim=(0,time_limit))
ax2.set(xlabel='Simulation time (s)', ylabel='PDR (%)')
ax2.xaxis.get_label().set_fontsize(14)
ax2.yaxis.get_label().set_fontsize(14)
ax2.plot( 'time', 'y4', '', data=df, marker='', color= uniquefuckingcolors[0], linewidth=2,linestyle='dashed', label='LTE')
ax2.plot( 'time', 'y5', '', data=df, marker='', color= uniquefuckingcolors[1], linewidth=2,label='LTE + UOS')
ax2.plot( 'time', 'y6', '', data=df, marker='', color= uniquefuckingcolors[2], linewidth=2,label='LTE + Percept')
# legends.append(ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2))
legends.append(ax2.legend(loc='upper center',bbox_to_anchor=(0.5, 0.27), ncol=2))

#---------------------300 Users SMALL CELLS------------------------------
ax3.set(ylim=(0,100), xlim=(0,time_limit))
ax3.set(xlabel='Simulation time (s)', ylabel='PDR (%)')
ax3.xaxis.get_label().set_fontsize(14)
ax3.yaxis.get_label().set_fontsize(14)
ax3.plot( 'time', 'y7', '', data=df, marker='', color=uniquefuckingcolors[0], linewidth=2,linestyle='dashed', label='LTE')
ax3.plot( 'time', 'y8', '', data=df, marker='', color= uniquefuckingcolors[1], linewidth=2,label='LTE + UOS')
ax3.plot( 'time', 'y9', '', data=df, marker='', color= uniquefuckingcolors[2], linewidth=2,label='LTE + Percept')
# legends.append(ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2))
legends.append(ax3.legend(loc='upper center',bbox_to_anchor=(0.5, 0.27), ncol=2))


#plt.ylabel('Number of UEs')
#plt.xlabel('Simulation time (s)')
#plt.suptitle('Average Low SINR Users')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
fig.subplots_adjust(right=1.95, top=4.0, hspace=0.3)
# plt.subplots_adjust(right=1.5, top=2.25, hspace=0.3)
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
fig.savefig('lineplot_PDR.pdf', dpi=1000, bbox_inches='tight', bbox_extra_artists=legends)

extent = ax1.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
fig.savefig('lineplot_PDR_100U.pdf', bbox_inches=extent.expanded(1.02, 1.05))

extent = ax2.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
fig.savefig('lineplot_PDR_200U.pdf', bbox_inches=extent.expanded(1.02, 1.05))

extent = ax3.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
fig.savefig('lineplot_PDR_300U.pdf', bbox_inches=extent.expanded(1.02, 1.05))
