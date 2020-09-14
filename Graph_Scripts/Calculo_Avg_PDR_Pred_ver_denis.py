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

def runMean(NUsers,NRuns,Run,time, metric):
        # Mean general scenario
    countarr = [None] * NRuns #debe ser 33 o 30 o 20 o el # de runs
    metricarr = [None] * NRuns  #debe ser 33 o 30 o 20 o el # de runs
    metricMean = [None] * NRuns #debe ser 33 o 30 o 20 o el # de runs
    for i in range(NRuns):
        count=0
        sumplr=0
        for j in range(len(Run)):
            if (Run[j]==i):
                count+=1
                countarr[i]=count
                sumplr+= (metric[j] / NUsers)  #dividir entre cantidad de usuarios
                # metricarr[i] = sumplr
                if (time[j] == 100):
                    metricarr[i] = metric[j]

        metricMean[i] = (metricarr[i])
    #fairness_index_LTE = jfi(metricMean)
    
    return metricMean

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

                'uos+lte/enableSCs=false/scen=3/nENB=4/nUABS=0/nUE=100/*/', #LTE
                'uos+lte/enableSCs=false/scen=4/nENB=4/nUABS=8/nUE=100/*/', #UOS 8UABS
                'percept/enableSCs=false/nENB=4/nUABS=8/nUE=100/*/', #PERCEPT 8UABS

                'uos+lte/enableSCs=false/scen=3/nENB=4/nUABS=0/nUE=100/*/', #LTE
                'uos+lte/enableSCs=false/scen=4/nENB=4/nUABS=15/nUE=100/*/', #UOS 15UABS
                'percept/enableSCs=false/nENB=4/nUABS=15/nUE=100/*/', #PERCEPT 15UABS

                #no small cells 200 UEs
                'uos+lte/enableSCs=false/scen=3/nENB=4/nUABS=0/nUE=200/*/', #LTE
                'uos+lte/enableSCs=false/scen=4/nENB=4/nUABS=4/nUE=200/*/', #UOS 4UABS
                'percept/enableSCs=false/nENB=4/nUABS=4/nUE=200/*/', #PERCEPT 4UABS

                'uos+lte/enableSCs=false/scen=3/nENB=4/nUABS=0/nUE=200/*/', #LTE
                'uos+lte/enableSCs=false/scen=4/nENB=4/nUABS=8/nUE=200/*/', #UOS 8UABS
                'percept/enableSCs=false/nENB=4/nUABS=8/nUE=200/*/', #PERCEPT 8UABS

                'uos+lte/enableSCs=false/scen=3/nENB=4/nUABS=0/nUE=200/*/', #LTE
                'uos+lte/enableSCs=false/scen=4/nENB=4/nUABS=15/nUE=200/*/', #UOS 15UABS
                'percept/enableSCs=false/nENB=4/nUABS=15/nUE=200/*/', #PERCEPT 15UABS
                
                #small cells 100 UEs
                'uos+lte/enableSCs=true/scen=3/nENB=4/nUABS=0/nUE=100/*/', #LTE
                'uos+lte/enableSCs=true/scen=4/nENB=4/nUABS=4/nUE=100/*/', #UOS 4UABS
                'percept/enableSCs=true/nENB=4/nUABS=4/nUE=100/*/', #PERCEPT 4UABS

                'uos+lte/enableSCs=true/scen=3/nENB=4/nUABS=0/nUE=100/*/', #LTE
                'uos+lte/enableSCs=true/scen=4/nENB=4/nUABS=8/nUE=100/*/', #UOS 8UABS
                'percept/enableSCs=true/nENB=4/nUABS=8/nUE=100/*/', #PERCEPT 8UABS

                'uos+lte/enableSCs=true/scen=3/nENB=4/nUABS=0/nUE=100/*/', #LTE
                'uos+lte/enableSCs=true/scen=4/nENB=4/nUABS=15/nUE=100/*/', #UOS 15UABS
                'percept/enableSCs=true/nENB=4/nUABS=15/nUE=100/*/', #PERCEPT 15UABS

                #small cells 200 UEs
                'uos+lte/enableSCs=true/scen=3/nENB=4/nUABS=0/nUE=200/*/', #LTE
                'uos+lte/enableSCs=true/scen=4/nENB=4/nUABS=4/nUE=200/*/', #UOS 4UABS
                'percept/enableSCs=true/nENB=4/nUABS=4/nUE=200/*/', #PERCEPT 4UABS

                'uos+lte/enableSCs=true/scen=3/nENB=4/nUABS=0/nUE=200/*/', #LTE
                'uos+lte/enableSCs=true/scen=4/nENB=4/nUABS=8/nUE=200/*/', #UOS 8UABS
                'percept/enableSCs=true/nENB=4/nUABS=8/nUE=200/*/', #PERCEPT 8UABS

                'uos+lte/enableSCs=true/scen=3/nENB=4/nUABS=0/nUE=200/*/', #LTE
                'uos+lte/enableSCs=true/scen=4/nENB=4/nUABS=15/nUE=200/*/', #UOS 15UABS
                'percept/enableSCs=true/nENB=4/nUABS=15/nUE=200/*/' #PERCEPT 15UABS
                ]

No_scenarios = len(scenarios_path)
mean_by_run_all_scenarios = []
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
    mean_by_run = runMean(nUE, No_runs, run, time, Throughput)
    mean_by_run_all_scenarios.append(mean_by_run)


scenarios1 = ['100U_4enB_NoUABS','100U_4enB_UABS','100U_4enB_UPred','200U_4enB_NoUABS','200U_4enB_UABS','200U_4enB_UPred','100U_2enB_NoUABS','100U_2enB_UABS','100U_2enB_UPred','200U_2enB_NoUABS','200U_2enB_UABS','200U_2enB_UPred']
scenarios = ['','','100','','','','','200','','']

    
#colors  = ["#b3ffd6","#809c8c","#74b9ff","#0984e3", "#a29bfe", "#6c5ce7", "#dfe6e9", "#b2bec3"]
colors  = ["#edb1bd","#be6b82","#a88565","#705045"]#, "#cdd5e4", "#7c7f9e", "#20355a", "#3c4563"]
colors1  = ["#cdd5e4", "#7c7f9e", "#20355a", "#3c4563"]
uniquefuckingcolors  = ["#cdd5e4", "#7c7f9e", "#3c4563","#a55eea","#8854d0"]

LTE_Label = mpatches.Patch(color=uniquefuckingcolors[0], label='LTE')
UOS_Label = mpatches.Patch(color=uniquefuckingcolors[1], label='LTE+UOS')
PERCEPT_Label = mpatches.Patch(color=uniquefuckingcolors[2], label='LTE+PERCEPT')
# UOS_Label_15 = mpatches.Patch(color=uniquefuckingcolors[3], label='LTE+UOS 15 UAV-BS')
# PERCEPT_Label_15 = mpatches.Patch(color=uniquefuckingcolors[4], label='LTE+PERCEPT 15 UAV-BS')

# Array with means of scenarios
mean_by_scenario = [np.mean(mean_by_run) for mean_by_run in mean_by_run_all_scenarios]

# Array with y error of scenarios
yerr_by_scenario =  [mean_confidence_interval(mean_by_run) for mean_by_run in mean_by_run_all_scenarios]

# Create a figure instance  
fig2 = plt.figure(2, figsize=(11,14))

# Create an axes instance
ax2 = fig2.add_subplot(421)
ax3 = fig2.add_subplot(422)
ax4 = fig2.add_subplot(423)
ax5 = fig2.add_subplot(424)


# use a known color palette (see..)
pal = sns.color_palette("Set2")


#---------------------100 Users NO SMALL CELLS------------------------------
#4 UAV-BS
ax2.bar(np.arange(1), mean_by_scenario[0], yerr=yerr_by_scenario[0],width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2),label="LTE")
ax2.bar(np.arange(1)+0.6, mean_by_scenario[1], yerr=yerr_by_scenario[1],width =0.6, color= uniquefuckingcolors[1], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+UOS")
ax2.bar(np.arange(1)+1.2, mean_by_scenario[2], yerr=yerr_by_scenario[2],width =0.6, color= uniquefuckingcolors[2], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+Percept")
#8 UAV-BS
ax2.bar(np.arange(1)+3, mean_by_scenario[3], yerr=yerr_by_scenario[3],width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax2.bar(np.arange(1)+3.6, mean_by_scenario[4], yerr=yerr_by_scenario[4],width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax2.bar(np.arange(1)+4.2, mean_by_scenario[5], yerr=yerr_by_scenario[5],width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
#15 UAV-BS
ax2.bar(np.arange(1)+6, mean_by_scenario[6], yerr=yerr_by_scenario[6],width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax2.bar(np.arange(1)+6.6, mean_by_scenario[7], yerr=yerr_by_scenario[7],width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax2.bar(np.arange(1)+7.2, mean_by_scenario[8], yerr=yerr_by_scenario[8],width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))

ax2.set_xticks([0.5,3.5,6.5])
ax2.set_xticklabels([4,8,15],fontsize=14)
ax2.set(xlabel='Number of UAV-BS', ylabel='PDR (%)')
ax2.xaxis.get_label().set_fontsize(14)
ax2.yaxis.get_label().set_fontsize(14)
# ax2.set_title("4 TBS + No Small Cells \n 100 Users", fontsize=16)
ax2.grid(color='green',ls = 'dotted')
ax2.set_ylim([0,100])
ax2.legend(handles=[LTE_Label, UOS_Label, PERCEPT_Label])

#---------------------200 Users NO SMALL CELLS------------------------------
#4 UAV-BS
ax3.bar(np.arange(1), mean_by_scenario[9], yerr=yerr_by_scenario[9],width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2),label="LTE")
ax3.bar(np.arange(1)+0.6, mean_by_scenario[10], yerr=yerr_by_scenario[10],width =0.6, color= uniquefuckingcolors[1], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+UOS")
ax3.bar(np.arange(1)+1.2, mean_by_scenario[11], yerr=yerr_by_scenario[11],width =0.6, color= uniquefuckingcolors[2], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+Percept")
#8 UAV-BS
ax3.bar(np.arange(1)+3, mean_by_scenario[12], yerr=yerr_by_scenario[12],width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax3.bar(np.arange(1)+3.6, mean_by_scenario[13], yerr=yerr_by_scenario[13],width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax3.bar(np.arange(1)+4.2, mean_by_scenario[14], yerr=yerr_by_scenario[14],width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
#15 UAV-BS
ax3.bar(np.arange(1)+6, mean_by_scenario[15], yerr=yerr_by_scenario[15],width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax3.bar(np.arange(1)+6.6, mean_by_scenario[16], yerr=yerr_by_scenario[16],width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax3.bar(np.arange(1)+7.2, mean_by_scenario[17], yerr=yerr_by_scenario[17],width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))

ax3.set_xticks([0.5,3.5,6.5])
ax3.set_xticklabels([4,8,15],fontsize=14)
ax3.set(xlabel='Number of UAV-BS', ylabel='PDR (%)')
ax3.xaxis.get_label().set_fontsize(14)
ax3.yaxis.get_label().set_fontsize(14)
# ax3.set_title("4 TBS + No Small Cells \n 200 Users", fontsize=16)
ax3.grid(color='green',ls = 'dotted')
ax3.set_ylim([0,100])
ax3.legend(handles=[LTE_Label, UOS_Label, PERCEPT_Label])


#---------------------100 Users SMALL CELLS------------------------------
#4 UAV-BS
ax4.bar(np.arange(1), mean_by_scenario[18], yerr=yerr_by_scenario[18],width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2),label="LTE")
ax4.bar(np.arange(1)+0.6, mean_by_scenario[19], yerr=yerr_by_scenario[19],width =0.6, color= uniquefuckingcolors[1], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+UOS")
ax4.bar(np.arange(1)+1.2, mean_by_scenario[20], yerr=yerr_by_scenario[20],width =0.6, color= uniquefuckingcolors[2], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+Percept")
#8 UAV-BS
ax4.bar(np.arange(1)+3, mean_by_scenario[21], yerr=yerr_by_scenario[21],width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax4.bar(np.arange(1)+3.6, mean_by_scenario[22], yerr=yerr_by_scenario[22],width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax4.bar(np.arange(1)+4.2, mean_by_scenario[23], yerr=yerr_by_scenario[23],width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
#15 UAV-BS
ax4.bar(np.arange(1)+6, mean_by_scenario[24], yerr=yerr_by_scenario[24],width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax4.bar(np.arange(1)+6.6, mean_by_scenario[25], yerr=yerr_by_scenario[25],width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax4.bar(np.arange(1)+7.2, mean_by_scenario[26], yerr=yerr_by_scenario[26],width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))

ax4.set_xticks([0.5,3.5,6.5])
ax4.set_xticklabels([4,8,15],fontsize=14)
ax4.set(xlabel='Number of UAV-BS', ylabel='PDR (%)')
ax4.xaxis.get_label().set_fontsize(14)
ax4.yaxis.get_label().set_fontsize(14)
# ax4.set_title("4 TBS + Small Cells \n 100 Users", fontsize=16)
ax4.grid(color='green',ls = 'dotted')
ax4.set_ylim([0,100])
ax4.legend(handles=[LTE_Label, UOS_Label, PERCEPT_Label])

#---------------------200 Users SMALL CELLS------------------------------
#4 UAV-BS
ax5.bar(np.arange(1), mean_by_scenario[27], yerr=yerr_by_scenario[27],width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2),label="LTE")
ax5.bar(np.arange(1)+0.6, mean_by_scenario[28], yerr=yerr_by_scenario[28],width =0.6, color= uniquefuckingcolors[1], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+UOS")
ax5.bar(np.arange(1)+1.2, mean_by_scenario[29], yerr=yerr_by_scenario[29],width =0.6, color= uniquefuckingcolors[2], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+Percept")
#8 UAV-BS
ax5.bar(np.arange(1)+3, mean_by_scenario[30], yerr=yerr_by_scenario[30],width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax5.bar(np.arange(1)+3.6, mean_by_scenario[31], yerr=yerr_by_scenario[31],width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax5.bar(np.arange(1)+4.2, mean_by_scenario[32], yerr=yerr_by_scenario[32],width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
#15 UAV-BS
ax5.bar(np.arange(1)+6, mean_by_scenario[33], yerr=yerr_by_scenario[33],width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax5.bar(np.arange(1)+6.6, mean_by_scenario[34], yerr=yerr_by_scenario[34],width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax5.bar(np.arange(1)+7.2, mean_by_scenario[35], yerr=yerr_by_scenario[35],width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))

ax5.set_xticks([0.5,3.5,6.5])
ax5.set_xticklabels([4,8,15],fontsize=14)
ax5.set(xlabel='Number of UAV-BS', ylabel='PDR (%)')
ax5.xaxis.get_label().set_fontsize(14)
ax5.yaxis.get_label().set_fontsize(14)
# ax5.set_title("4 TBS + Small Cells \n 200 Users", fontsize=16)
ax5.grid(color='green',ls = 'dotted')
ax5.set_ylim([0,100])
ax5.legend(handles=[LTE_Label, UOS_Label, PERCEPT_Label])







plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.tight_layout(pad=3.0, w_pad=2, h_pad=1.0)



plt.savefig("Graph_Avg_PDR_Percept.pdf", format='pdf', dpi=1000, bbox_inches = "tight")
#plt.show()

# Save just the portion _inside_ the second axis's boundaries

extent = ax2.get_tightbbox(fig2.canvas.renderer).transformed(fig2.dpi_scale_trans.inverted())
# Pad the saved area by 20% in the x-direction and 20% in the y-direction
fig2.savefig('PDR_100U_NS.pdf', bbox_inches=extent)#.expanded(1.15, 1.15))

extent = ax3.get_tightbbox(fig2.canvas.renderer).transformed(fig2.dpi_scale_trans.inverted())
# extent = ax3.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig('PDR_200U_NS.pdf',  bbox_inches=extent)#.expanded(1.15, 1.15))

extent = ax4.get_tightbbox(fig2.canvas.renderer).transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig('PDR_100U_S.pdf',  bbox_inches=extent)#.expanded(1.15, 1.15))

extent = ax5.get_tightbbox(fig2.canvas.renderer).transformed(fig2.dpi_scale_trans.inverted())
fig2.savefig('PDR_200U_S.pdf',  bbox_inches=extent)#.expanded(1.15, 1.15))




exit()
#TODO fix statistics
#---------------------------------------------------------------------------------------------------
#--------------------------Statistics---------------------------------------------------------------

scen_comp = ["LTE_vs_UOS_4UAV_100",
             "LTE_vs_Percept_4UAV_100", 
             "LTE_vs_UOS_8UAV_100",
             "LTE_vs_Percept_8UAV_100",
             "LTE_vs_UOS_15UAV_100",
             "LTE_vs_Percept_15UAV_100",
             
             "LTE_vs_UOS_4UAV_100_SC",
             "LTE_vs_Percept_4UAV_100_SC", 
             "LTE_vs_UOS_8UAV_100_SC",
             "LTE_vs_Percept_8UAV_100_SC",
             "LTE_vs_UOS_15UAV_100_SC",
             "LTE_vs_Percept_15UAV_100_SC",
             
             "LTE_vs_UOS_4UAV_200",
             "LTE_vs_Percept_4UAV_200", 
             "LTE_vs_UOS_8UAV_200",
             "LTE_vs_Percept_8UAV_200",
             "LTE_vs_UOS_15UAV_200",
             "LTE_vs_Percept_15UAV_200",
             
             "LTE_vs_UOS_4UAV_200_SC",
             "LTE_vs_Percept_4UAV_200_SC", 
             "LTE_vs_UOS_8UAV_200_SC",
             "LTE_vs_Percept_8UAV_200_SC",
             "LTE_vs_UOS_15UAV_200_SC",
             "LTE_vs_Percept_15UAV_200_SC"
             ]
             
scen_comp_UOS_vs_LTE = ["UOS_vs_Percept_4UAV_100", 
                        "UOS_vs_Percept_8UAV_100", 
                        "UOS_vs_Percept_15UAV_100",
                        
                        "UOS_vs_Percept_4UAV_100_SC", 
                        "UOS_vs_Percept_8UAV_100_SC", 
                        "UOS_vs_Percept_15UAV_100_SC",
                        
                        "UOS_vs_Percept_4UAV_200", 
                        "UOS_vs_Percept_8UAV_200", 
                        "UOS_vs_Percept_15UAV_200",
                        
                         "UOS_vs_Percept_4UAV_200_SC", 
                        "UOS_vs_Percept_8UAV_200_SC", 
                        "UOS_vs_Percept_15UAV_200_SC"]    

means_scenarios = np.array(list((means_UOS_100_4,
                     means_Percept_100_4,
                     means_UOS_100_8,
                     means_Percept_100_8,
                     means_UOS_100_15,
                     means_Percept_100_15,
                     
                     means_UOS_100_4_SC,
                     means_Percept_100_4_SC,
                     means_UOS_100_8_SC,
                     means_Percept_100_8_SC,
                     means_UOS_100_15_SC,
                     means_Percept_100_15_SC,
                     
                     means_UOS_200_4,
                     means_Percept_200_4,
                     means_UOS_200_8,
                     means_Percept_200_8,
                     means_UOS_200_15,
                     means_Percept_200_15,
                     
                     means_UOS_200_4_SC,
                     means_Percept_200_4_SC,
                     means_UOS_200_8_SC,
                     means_Percept_200_8_SC,
                     means_UOS_200_15_SC,
                     means_Percept_200_15_SC)))

means_scenarios_UOS = np.array(list((means_UOS_100_4,
                                     means_UOS_100_8,
                                     means_UOS_100_15,
                     
                                     means_UOS_100_4_SC,
                                     means_UOS_100_8_SC,
                                     means_UOS_100_15_SC,
                                     
                                     means_UOS_200_4,
                                     means_UOS_200_8,
                                     means_UOS_200_15,
                                     
                                     means_UOS_200_4_SC,
                                     means_UOS_200_8_SC,
                                     means_UOS_200_15_SC)))

means_scenarios_Percept = np.array(list((means_Percept_100_4,
                                     means_Percept_100_8,
                                     means_Percept_100_15,
                     
                                     means_Percept_100_4_SC,
                                     means_Percept_100_8_SC,
                                     means_Percept_100_15_SC,
                                     
                                     means_Percept_200_4,
                                     means_Percept_200_8,
                                     means_Percept_200_15,
                                     
                                     means_Percept_200_4_SC,
                                     means_Percept_200_8_SC,
                                     means_Percept_200_15_SC)))

print("LTE vs UOS/Percept Statistics:")
print("\n")

for i in range(len(means_scenarios)):
    if (i<13):
        if (i<7):
            ImprovRatio_LTE_vs_UAV_100 = means_LTE_100/means_scenarios[i]
    
            print("Improvement ratio " + str(scen_comp[i]) +" : "  + str(ImprovRatio_LTE_vs_UAV_100))  # improvement ratio = value after change / value before change
        
            print("Improvement %: " + str( 100 * (ImprovRatio_LTE_vs_UAV_100 - 1)) +"%") 
        else:
           ImprovRatio_LTE_vs_UAV_100_SC = means_LTE_100_SC/means_scenarios[i] 
           
           print("Improvement ratio " + str(scen_comp[i]) +" : "  + str(ImprovRatio_LTE_vs_UAV_100_SC))  # improvement ratio = value after change / value before change
        
           print("Improvement %: " + str( 100 * (ImprovRatio_LTE_vs_UAV_100_SC - 1)) +"%") 
    else:
        if (i<19):
            ImprovRatio_LTE_vs_UAV_200 = means_LTE_200/means_scenarios[i]
    
            print("Improvement ratio " + str(scen_comp[i]) +" : "  + str(ImprovRatio_LTE_vs_UAV_200))  # improvement ratio = value after change / value before change
        
            print("Improvement %: " + str( 100 * (ImprovRatio_LTE_vs_UAV_200 - 1)) +"%") 
        else:
           ImprovRatio_LTE_vs_UAV_200_SC = means_LTE_200_SC/means_scenarios[i]
           
           print("Improvement ratio " + str(scen_comp[i]) +" : "  + str(ImprovRatio_LTE_vs_UAV_200_SC))  # improvement ratio = value after change / value before change
        
           print("Improvement %: " + str( 100 * (ImprovRatio_LTE_vs_UAV_200_SC - 1)) +"%") 
        

print("\n")
print("UOS vs Percept Statistics:")
print("\n")

sum_improv_ratio = 0
for i in range(len(means_scenarios_UOS)):
    ImprovRatio_UOS_vs_Percept = means_scenarios_UOS[i]/means_scenarios_Percept[i]
            
    print("Improvement ratio " + str(scen_comp_UOS_vs_LTE[i]) +" : "  + str(ImprovRatio_UOS_vs_Percept))  # improvement ratio = value after change / value before change
        
    print("Improvement %: " + str( 100 * (ImprovRatio_UOS_vs_Percept - 1)) +"%") 
    
    sum_improv_ratio += 100 * (ImprovRatio_UOS_vs_Percept - 1)

mean_improv_ratio = abs(sum_improv_ratio/len(means_scenarios_Percept))    
print("Improv Ratio Mean Percept vs UOS: " + str(mean_improv_ratio) + "%")
