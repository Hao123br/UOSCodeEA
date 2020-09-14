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
                #no small cells 100 UEs
                'uos+lte/enableSCs=false/scen=3/nENB=4/nUABS=0/nUE=100/*/', #LTE
                'uos+lte/enableSCs=false/scen=4/nENB=4/nUABS=4/nUE=100/*/', #UOS 4UABS
                'percept/enableSCs=false/nENB=4/nUABS=4/nUE=100/*/', #PERCEPT 4UABS
                
                'uos+lte/enableSCs=false/scen=4/nENB=4/nUABS=8/nUE=100/*/', #UOS 8UABS
                'percept/enableSCs=false/nENB=4/nUABS=8/nUE=100/*/', #PERCEPT 8UABS
                
                'uos+lte/enableSCs=false/scen=4/nENB=4/nUABS=15/nUE=100/*/', #UOS 15UABS
                'percept/enableSCs=false/nENB=4/nUABS=15/nUE=100/*/', #PERCEPT 15UABS

                #no small cells 200 UEs
                'uos+lte/enableSCs=false/scen=3/nENB=4/nUABS=0/nUE=200/*/', #LTE
                'uos+lte/enableSCs=false/scen=4/nENB=4/nUABS=4/nUE=200/*/', #UOS 4UABS
                'percept/enableSCs=false/nENB=4/nUABS=4/nUE=200/*/', #PERCEPT 4UABS

                'uos+lte/enableSCs=false/scen=4/nENB=4/nUABS=8/nUE=200/*/', #UOS 8UABS
                'percept/enableSCs=false/nENB=4/nUABS=8/nUE=200/*/', #PERCEPT 8UABS

                'uos+lte/enableSCs=false/scen=4/nENB=4/nUABS=15/nUE=200/*/', #UOS 15UABS
                'percept/enableSCs=false/nENB=4/nUABS=15/nUE=200/*/', #PERCEPT 15UABS
                
                #small cells 100 UEs
                'uos+lte/enableSCs=true/scen=3/nENB=4/nUABS=0/nUE=100/*/', #LTE
                'uos+lte/enableSCs=true/scen=4/nENB=4/nUABS=4/nUE=100/*/', #UOS 4UABS
                'percept/enableSCs=true/nENB=4/nUABS=4/nUE=100/*/', #PERCEPT 4UABS

                'uos+lte/enableSCs=true/scen=4/nENB=4/nUABS=8/nUE=100/*/', #UOS 8UABS
                'percept/enableSCs=true/nENB=4/nUABS=8/nUE=100/*/', #PERCEPT 8UABS

                'uos+lte/enableSCs=true/scen=4/nENB=4/nUABS=15/nUE=100/*/', #UOS 15UABS
                'percept/enableSCs=true/nENB=4/nUABS=15/nUE=100/*/', #PERCEPT 15UABS

                #small cells 200 UEs
                'uos+lte/enableSCs=true/scen=3/nENB=4/nUABS=0/nUE=200/*/', #LTE
                'uos+lte/enableSCs=true/scen=4/nENB=4/nUABS=4/nUE=200/*/', #UOS 4UABS
                'percept/enableSCs=true/nENB=4/nUABS=4/nUE=200/*/', #PERCEPT 4UABS

                'uos+lte/enableSCs=true/scen=4/nENB=4/nUABS=8/nUE=200/*/', #UOS 8UABS
                'percept/enableSCs=true/nENB=4/nUABS=8/nUE=200/*/', #PERCEPT 8UABS

                'uos+lte/enableSCs=true/scen=4/nENB=4/nUABS=15/nUE=200/*/', #UOS 15UABS
                'percept/enableSCs=true/nENB=4/nUABS=15/nUE=200/*/' #PERCEPT 15UABS
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
ax1=plt.subplot(4,2,1)
ax2=plt.subplot(4,2,2)
ax3=plt.subplot(4,2,3)
ax4=plt.subplot(4,2,4)

#Color palette used in master thesis
#uniquefuckingcolors  = ['#cdd5e4', '#7c7f9e']
uniquefuckingcolors  = ["#cdd5e4", "#7c7f9e", "#3c4563","#a55eea","#8854d0","#26de81","#20bf6b"]
uniquefuckingcolors  = ["#eb3b5a", "#7c7f9e", "#3c4563","#a55eea","#8854d0","#26de81","#20bf6b"] #with LTE red

legends = []
# multiple line plot
time_limit = data_dict['time'][-1]
ax1.set(ylim=(0,100), xlim=(0,time_limit))

#---------------------100 Users NO SMALL CELLS------------------------------
ax1.plot( 'time', 'y1', '', data=df, marker='', color= uniquefuckingcolors[0], linewidth=2,linestyle='dashed', label='LTE')
ax1.plot( 'time', 'y2', '', data=df, marker='', color= uniquefuckingcolors[1], linewidth=2,label='LTE + UOS 4 UAV-BS')
ax1.plot( 'time', 'y3', '', data=df, marker='', color= uniquefuckingcolors[2], linewidth=2,label='LTE + Percept 4 UAV-BS')
ax1.plot( 'time', 'y4', '', data=df, marker='', color= uniquefuckingcolors[3], linewidth=2,label='LTE + UOS 8 UAV-BS')
ax1.plot( 'time', 'y5', '', data=df, marker='', color= uniquefuckingcolors[4], linewidth=2,label='LTE + Percept 8 UAV-BS')
ax1.plot( 'time', 'y6', '', data=df, marker='', color= uniquefuckingcolors[5], linewidth=2,label='LTE + UOS 15 UAV-BS')
ax1.plot( 'time', 'y7', '', data=df, marker='', color= uniquefuckingcolors[6], linewidth=2,label='LTE + Percept 15 UAV-BS')
ax1.set(xlabel='Simulation time (s)', ylabel='PDR (%)')
ax1.xaxis.get_label().set_fontsize(14)
ax1.yaxis.get_label().set_fontsize(14)
# legends.append(ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2))
legends.append(ax1.legend(loc='upper center',bbox_to_anchor=(0.5, 0.27), ncol=2))
ax1.set_title('100 Users | No Small Cells',fontsize=14)

#---------------------200 Users NO SMALL CELLS------------------------------
ax2.set(ylim=(0,100), xlim=(0,time_limit))
ax2.plot( 'time', 'y8', '', data=df, marker='', color=uniquefuckingcolors[0], linewidth=2,linestyle='dashed', label='LTE')
ax2.plot( 'time', 'y9', '', data=df, marker='', color= uniquefuckingcolors[1], linewidth=2,label='LTE + UOS 4 UAV-BS')
ax2.plot( 'time', 'y10', '', data=df, marker='', color= uniquefuckingcolors[2], linewidth=2,label='LTE + Percept 4 UAB-BS')
ax2.plot( 'time', 'y11', '', data=df, marker='', color= uniquefuckingcolors[3], linewidth=2,label='LTE + UOS 8 UAV-BS')
ax2.plot( 'time', 'y12', '', data=df, marker='', color= uniquefuckingcolors[4], linewidth=2,label='LTE + Percept 8 UAB-BS')
ax2.plot( 'time', 'y13', '', data=df, marker='', color= uniquefuckingcolors[5], linewidth=2,label='LTE + UOS 15 UAV-BS')
ax2.plot( 'time', 'y14', '', data=df, marker='', color= uniquefuckingcolors[6], linewidth=2,label='LTE + Percept 15 UAB-BS')
ax2.set(xlabel='Simulation time (s)', ylabel='PDR (%)')
ax2.xaxis.get_label().set_fontsize(14)
ax2.yaxis.get_label().set_fontsize(14)
# legends.append(ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2))
legends.append(ax2.legend(loc='upper center',bbox_to_anchor=(0.5, 0.27), ncol=2))
ax2.set_title('200 Users | No Small Cells',fontsize=14)


#---------------------100 Users SMALL CELLS------------------------------
ax3.set(ylim=(0,100), xlim=(0,time_limit))
ax3.plot( 'time', 'y15', '', data=df, marker='', color=uniquefuckingcolors[0], linewidth=2,linestyle='dashed', label='LTE')
ax3.plot( 'time', 'y16', '', data=df, marker='', color= uniquefuckingcolors[1], linewidth=2,label='LTE + UOS 4 UAV-BS')
ax3.plot( 'time', 'y17', '', data=df, marker='', color= uniquefuckingcolors[2], linewidth=2,label='LTE + Percept 4 UAV-BS')
ax3.plot( 'time', 'y18', '', data=df, marker='', color= uniquefuckingcolors[3], linewidth=2,label='LTE + UOS 8 UAV-BS')
ax3.plot( 'time', 'y19', '', data=df, marker='', color= uniquefuckingcolors[4], linewidth=2,label='LTE + Percept 8 UAV-BS')
ax3.plot( 'time', 'y20', '', data=df, marker='', color= uniquefuckingcolors[5], linewidth=2,label='LTE + UOS 15 UAV-BS')
ax3.plot( 'time', 'y21', '', data=df, marker='', color= uniquefuckingcolors[6], linewidth=2,label='LTE + Percept 15 UAV-BS')
ax3.set(xlabel='Simulation time (s)', ylabel='PDR (%)')
ax3.xaxis.get_label().set_fontsize(14)
ax3.yaxis.get_label().set_fontsize(14)
# legends.append(ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2))
legends.append(ax3.legend(loc='upper center',bbox_to_anchor=(0.5, 0.27), ncol=2))
ax3.set_title('100 Users | Small Cells',fontsize=14)

#---------------------200 Users SMALL CELLS------------------------------
ax4.set(ylim=(0,100), xlim=(0,time_limit))
ax4.plot( 'time', 'y22', '', data=df, marker='', color=uniquefuckingcolors[0], linewidth=2,linestyle='dashed', label='LTE') #200U-4eNBs-noUABS
ax4.plot( 'time', 'y23', '', data=df, marker='', color= uniquefuckingcolors[1], linewidth=2,label='LTE + UOS 4 UAV-BS') #200U-4eNBs-6UABS
ax4.plot( 'time', 'y24', '', data=df, marker='', color= uniquefuckingcolors[2], linewidth=2,label='LTE + Percept 4 UAV-BS') #PERCEPT SCENARIO
ax4.plot( 'time', 'y25', '', data=df, marker='', color= uniquefuckingcolors[3], linewidth=2,label='LTE + UOS 8 UAV-BS') #200U-4eNBs-6UABS
ax4.plot( 'time', 'y26', '', data=df, marker='', color= uniquefuckingcolors[4], linewidth=2,label='LTE + Percept 8 UAV-BS') #PERCEPT SCENARIO
ax4.plot( 'time', 'y27', '', data=df, marker='', color= uniquefuckingcolors[5], linewidth=2,label='LTE + UOS 15 UAV-BS') #200U-4eNBs-6UABS
ax4.plot( 'time', 'y28', '', data=df, marker='', color= uniquefuckingcolors[6], linewidth=2,label='LTE + Percept 15 UAV-BS') #PERCEPT SCENARIO
ax4.set(xlabel='Simulation time (s)', ylabel='PDR (%)')
ax4.xaxis.get_label().set_fontsize(14)
ax4.yaxis.get_label().set_fontsize(14)
# legends.append(ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2))
legends.append(ax4.legend(loc='upper center',bbox_to_anchor=(0.5, 0.27), ncol=2))
ax4.set_title('200 Users | Small Cells',fontsize=14)


#plt.ylabel('Number of UEs')
#plt.xlabel('Simulation time (s)')
#plt.suptitle('Average Low SINR Users')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
plt.subplots_adjust(right=1.95, top=4.0, hspace=0.3)
# plt.subplots_adjust(right=1.5, top=2.25, hspace=0.3)
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
# plt.tight_layout(pad=3.0, w_pad=2, h_pad=1.0)
plt.savefig('lineplot_PDR.png', dpi=1000, bbox_inches='tight', bbox_extra_artists=legends)
