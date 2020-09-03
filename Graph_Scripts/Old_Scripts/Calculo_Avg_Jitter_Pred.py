#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:12:25 2019

@author: emanuel
"""

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
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
import math


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def jfi(vetor): #Fairness Index
    sum0=0
    sum1=0
    jfi = [None] * 26
    for i in range(26):
        sum0 += vetor[i]
        sum1 += pow(vetor[i],2)
        jfi[i] = pow(sum0,2)/((i+1)*sum1)
    return jfi

def meanMetric(NUsers,NRuns,Run,time,Throughput):
        # Mean general scenario
    countarr = [None] * NRuns #debe ser 33 o 30 o 20 o el # de runs
    Throughputarr = [None] * NRuns  #debe ser 33 o 30 o 20 o el # de runs
    ThroughputMean = [None] * NRuns #debe ser 33 o 30 o 20 o el # de runs
    for i in range(NRuns):
        count=0
        sumplr=0
        for j in range(len(Run)):
            if (Run[j]==i):
                count+=1
                countarr[i]=count
                sumplr+= (Throughput[j] / NUsers)  #dividir entre cantidad de usuarios
                # Throughputarr[i] = sumplr
                if (time[j] == 100):
                    # if (np.isnan(Throughput[j])):
                        # print("Nan " + str(math.isnan(Throughput[j])))
                    Throughputarr[i] = Throughput[j]

        ThroughputMean[i] = (Throughputarr[i])
    #fairness_index_LTE = jfi(ThroughputMean)
    
    return ThroughputMean

# path= '/run/user/1000/gvfs/sftp:host=gercom.ddns.net,port=8372'
path= ''

#smallcells
#LTE
with open(path+'/home/emanuel/Desktop/IEEE_Article/Small_Cells/LTE_and_UOS_Results/LTE/4enB_200U/plot_Jitter_200_4_0') as MeanThroughput:

    data1 = np.array(list((int(Run),int(time), float(Throughput)) for Run, time, Throughput in csv.reader(MeanThroughput, delimiter= ' ')))
#UOS 8UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Small_Cells/LTE_and_UOS_Results/UOS/4enB_8UABS_200U/plot_Jitter_200_4_8') as MeanThroughputUABS:

    data2 = np.array(list((int(Run1),int(time1), float(Throughput1)) for Run1, time1, Throughput1 in csv.reader(MeanThroughputUABS, delimiter= ' ')))    
#PERCEPT 8UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Small_Cells/Percept_Results/4enB_8UABS_200U/plot_Jitter_200_4_8') as MeanThroughputUABS7:

    data12 = np.array(list((int(Run11),int(time11), float(Throughput11)) for Run11, time11, Throughput11 in csv.reader(MeanThroughputUABS7, delimiter= ' '))) 
#UOS 15UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Small_Cells/LTE_and_UOS_Results/UOS/4enB_15UABS_200U/plot_Jitter_200_4_15') as MeanThroughputUABS8:

    data13 = np.array(list((int(Run12),int(time12), float(Throughput12)) for Run12, time12, Throughput12 in csv.reader(MeanThroughputUABS8, delimiter= ' ')))    
#PERCEPT 15UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Small_Cells/Percept_Results/4enB_15UABS_200U/plot_Jitter_200_4_15') as MeanThroughputUABS9:

    data14 = np.array(list((int(Run13),int(time13), float(Throughput13)) for Run13, time13, Throughput13 in csv.reader(MeanThroughputUABS9, delimiter= ' '))) 


#no_smallcell
#LTE
with open(path+'/home/emanuel/Desktop/IEEE_Article/No_Small_Cells/LTE_and_UOS_Results/LTE/4enB_200U/plot_Jitter_200_4_0') as MeanThroughput1:

    data3 = np.array(list((int(Run2),int(time2), float(Throughput2)) for Run2, time2, Throughput2 in csv.reader(MeanThroughput1, delimiter= ' ')))
#UOS 8UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/No_Small_Cells/LTE_and_UOS_Results/UOS/4enB_8UABS_200U/plot_Jitter_200_4_8') as MeanThroughputUABS1:

    data4 = np.array(list((int(Run3),int(time3), float(Throughput3)) for Run3, time3, Throughput3 in csv.reader(MeanThroughputUABS1, delimiter= ' ')))   
#PERCEPT 8UABS 
with open(path+'/home/emanuel/Desktop/IEEE_Article/No_Small_Cells/Percept_Results/4enB_8UABS_200U/plot_Jitter_200_4_8') as MeanThroughputUABS6:

    data11 = np.array(list((int(Run10),int(time10), float(Throughput10)) for Run10, time10, Throughput10 in csv.reader(MeanThroughputUABS6, delimiter= ' ')))  
#UOS 15UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/No_Small_Cells/LTE_and_UOS_Results/UOS/4enB_15UABS_200U/plot_Jitter_200_4_15') as MeanThroughputUABS10:

    data15 = np.array(list((int(Run14),int(time14), float(Throughput14)) for Run14, time14, Throughput14 in csv.reader(MeanThroughputUABS10, delimiter= ' ')))   
#PERCEPT 15UABS 
with open(path+'/home/emanuel/Desktop/IEEE_Article/No_Small_Cells/Percept_Results/4enB_15UABS_200U/plot_Jitter_200_4_15') as MeanThroughputUABS11:

    data16 = np.array(list((int(Run15),int(time15), float(Throughput15)) for Run15, time15, Throughput15 in csv.reader(MeanThroughputUABS11, delimiter= ' ')))  


#smallcell
#LTE
with open(path+'/home/emanuel/Desktop/IEEE_Article/Small_Cells/LTE_and_UOS_Results/LTE/4enB_100U/plot_Jitter_100_4_0') as MeanThroughput2:

    data5 = np.array(list((int(Run4),int(time4), float(Throughput4)) for Run4, time4, Throughput4 in csv.reader(MeanThroughput2, delimiter= ' ')))
#UOS 8UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Small_Cells/LTE_and_UOS_Results/UOS/4enB_8UABS_100U/plot_Jitter_100_4_8') as MeanThroughputUABS2:

    data6 = np.array(list((int(Run5),int(time5), float(Throughput5)) for Run5, time5, Throughput5 in csv.reader(MeanThroughputUABS2, delimiter= ' ')))  
#PERCEPT 8UABS    
with open(path+'/home/emanuel/Desktop/IEEE_Article/Small_Cells/Percept_Results/4enB_8UABS_100U/plot_Jitter_100_4_8') as MeanThroughputUABS5:

    data10 = np.array(list((int(Run9),int(time9), float(Throughput9)) for Run9, time9, Throughput9 in csv.reader(MeanThroughputUABS5, delimiter= ' ')))     
#UOS 15UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Small_Cells/LTE_and_UOS_Results/UOS/4enB_15UABS_100U/plot_Jitter_100_4_15') as MeanThroughputUABS12:

    data17 = np.array(list((int(Run16),int(time16), float(Throughput16)) for Run16, time16, Throughput16 in csv.reader(MeanThroughputUABS12, delimiter= ' ')))  
#PERCEPT 15UABS    
with open(path+'/home/emanuel/Desktop/IEEE_Article/Small_Cells/Percept_Results/4enB_15UABS_100U/plot_Jitter_100_4_15') as MeanThroughputUABS13:

    data18 = np.array(list((int(Run17),int(time17), float(Throughput17)) for Run17, time17, Throughput17 in csv.reader(MeanThroughputUABS13, delimiter= ' ')))        

#no_smallcell
#LTE 
with open(path+'/home/emanuel/Desktop/IEEE_Article/No_Small_Cells/LTE_and_UOS_Results/LTE/4enB_200U/plot_Jitter_200_4_0') as MeanThroughput3:

    data7 = np.array(list((int(Run6),int(time6), float(Throughput6)) for Run6, time6, Throughput6 in csv.reader(MeanThroughput3, delimiter= ' ')))
#UOS 8UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/No_Small_Cells/LTE_and_UOS_Results/UOS/4enB_8UABS_100U/plot_Jitter_100_4_8') as MeanThroughputUABS3:

    data8 = np.array(list((int(Run7),int(time7), float(Throughput7)) for Run7, time7, Throughput7 in csv.reader(MeanThroughputUABS3, delimiter= ' ')))
#PERCEPT 8UABS      
with open(path+'/home/emanuel/Desktop/IEEE_Article/No_Small_Cells/Percept_Results/4enB_8UABS_100U/plot_Jitter_100_4_8') as MeanThroughputUABS4:

    data9 = np.array(list((int(Run8),int(time8), float(Throughput8)) for Run8, time8, Throughput8 in csv.reader(MeanThroughputUABS4, delimiter= ' '))) 
#UOS 15UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/No_Small_Cells/LTE_and_UOS_Results/UOS/4enB_15UABS_100U/plot_Jitter_100_4_15') as MeanThroughputUABS14:

    data19 = np.array(list((int(Run18),int(time18), float(Throughput18)) for Run18, time18, Throughput18 in csv.reader(MeanThroughputUABS14, delimiter= ' ')))
#PERCEPT 15UABS      
with open(path+'/home/emanuel/Desktop/IEEE_Article/No_Small_Cells/Percept_Results/4enB_15UABS_100U/plot_Jitter_100_4_15') as MeanThroughputUABS15:

    data20 = np.array(list((int(Run19),int(time19), float(Throughput19)) for Run19, time19, Throughput19 in csv.reader(MeanThroughputUABS15, delimiter= ' ')))  

Run,time,Throughput = data1.T
Run1,time1,Throughput1 = data2.T
Run2,time2,Throughput2 = data3.T
Run3,time3,Throughput3 = data4.T
Run4, time4, Throughput4 = data5.T
Run5, time5, Throughput5 = data6.T
Run6, time6, Throughput6 = data7.T
Run7, time7, Throughput7 = data8.T
Run8, time8, Throughput8 = data9.T
Run9, time9, Throughput9 = data10.T
Run10, time10, Throughput10 = data11.T
Run11, time11, Throughput11 = data12.T
Run12, time12, Throughput12 = data13.T
Run13, time13, Throughput13 = data14.T
Run14, time14, Throughput14 = data15.T
Run15, time15, Throughput15 = data16.T
Run16, time16, Throughput16 = data17.T
Run17, time17, Throughput17 = data18.T
Run18, time18, Throughput18 = data19.T
Run19, time19, Throughput19 = data20.T

NUsers=100
NUsers2=200
NRuns_scen1 = int(max(Run))
NRuns_scen2 = int(max(Run1))
NRuns_scen3 = int(max(Run2))
NRuns_scen4 = int(max(Run3))
NRuns_scen5 = int(max(Run4))
NRuns_scen6 = int(max(Run5))
NRuns_scen7 = int(max(Run6))
NRuns_scen8 = int(max(Run7))
NRuns_scen9 = int(max(Run8))
NRuns_scen10 = int(max(Run9))
NRuns_scen11 = int(max(Run10))
NRuns_scen12 = int(max(Run11))
NRuns_scen13 = int(max(Run12))
NRuns_scen14 = int(max(Run13))
NRuns_scen15 = int(max(Run14))
NRuns_scen16 = int(max(Run15))
NRuns_scen17 = int(max(Run16))
NRuns_scen18 = int(max(Run17))
NRuns_scen19 = int(max(Run18))
NRuns_scen20 = int(max(Run19))


ThroughputMean_200_6_NoUABS= meanMetric(NUsers2,NRuns_scen1,Run,time,Throughput) #4enb 
#
ThroughputMean_200_6_UABS= meanMetric(NUsers2,NRuns_scen2,Run1,time1,Throughput1)  #4enb 

ThroughputMean_200_2_NoUABS= meanMetric(NUsers2,NRuns_scen3,Run2,time2,Throughput2)  #2enb 

ThroughputMean_200_2_UABS= meanMetric(NUsers2,NRuns_scen4,Run3,time3,Throughput3)  #2enb 

ThroughputMean_200_2_UABS_pred= meanMetric(NUsers2,NRuns_scen11,Run10, time10, Throughput10)  #2enb 

ThroughputMean_200_4_UABS_pred= meanMetric(NUsers2,NRuns_scen12,Run11, time11, Throughput11)  #4enb

Metric_Mean_200_SMALL_15UABS_UOS= meanMetric(NUsers2,NRuns_scen13,Run12, time12, Throughput12)  # small cell

Metric_Mean_200_SMALL_15UABS_Percept= meanMetric(NUsers2,NRuns_scen14,Run13, time13, Throughput13)  #small cell

Metric_Mean_200_NSMALL_15UABS_UOS= meanMetric(NUsers2,NRuns_scen15,Run14, time14, Throughput14)  # no small cell

Metric_Mean_200_NSMALL_15UABS_Percept= meanMetric(NUsers2,NRuns_scen16,Run15, time15, Throughput15)  #no small cell



ThroughputMean_100_6_NoUABS= meanMetric(NUsers,NRuns_scen5,Run4, time4, Throughput4) #4enb 
#
ThroughputMean_100_6_UABS= meanMetric(NUsers,NRuns_scen6,Run5, time5, Throughput5)  #4enb 

ThroughputMean_100_2_NoUABS= meanMetric(NUsers,NRuns_scen7,Run6, time6, Throughput6)  #2enb 

ThroughputMean_100_2_UABS= meanMetric(NUsers,NRuns_scen8,Run7, time7, Throughput7)  #2enb 

ThroughputMean_100_2_UABS_pred= meanMetric(NUsers,NRuns_scen9,Run8, time8, Throughput8)  #2enb 

ThroughputMean_100_4_UABS_pred= meanMetric(NUsers,NRuns_scen10,Run9, time9, Throughput9)  #4enb

Metric_Mean_100_SMALL_15UABS_UOS= meanMetric(NUsers2,NRuns_scen17,Run16, time16, Throughput16)  #Small cell

Metric_Mean_100_SMALL_15UABS_Percept= meanMetric(NUsers2,NRuns_scen18,Run17, time17, Throughput17)  #Small cell

Metric_Mean_100_NSMALL_15UABS_UOS= meanMetric(NUsers2,NRuns_scen19,Run18, time18, Throughput18)  # No small cell

Metric_Mean_100_NSMALL_15UABS_Percept= meanMetric(NUsers2,NRuns_scen20,Run19, time19, Throughput19)  # No small cell

 
scenarios1 = ['100U_4enB_NoUABS','100U_4enB_UABS','100U_4enB_UPred','200U_4enB_NoUABS','200U_4enB_UABS','200U_4enB_UPred','100U_2enB_NoUABS','100U_2enB_UABS','100U_2enB_UPred','200U_2enB_NoUABS','200U_2enB_UABS','200U_2enB_UPred']
scenarios = ['','','100','','','','','200','','']

    
#colors  = ["#b3ffd6","#809c8c","#74b9ff","#0984e3", "#a29bfe", "#6c5ce7", "#dfe6e9", "#b2bec3"]
colors  = ["#edb1bd","#be6b82","#a88565","#705045"]#, "#cdd5e4", "#7c7f9e", "#20355a", "#3c4563"]
colors1  = ["#cdd5e4", "#7c7f9e", "#20355a", "#3c4563"]
uniquefuckingcolors  = ["#cdd5e4", "#7c7f9e", "#3c4563","#a55eea","#8854d0"]

LTE_Label = mpatches.Patch(color=uniquefuckingcolors[0], label='LTE')
UOS_Label = mpatches.Patch(color=uniquefuckingcolors[1], label='LTE+UOS 8 UAV-BS')
PERCEPT_Label = mpatches.Patch(color=uniquefuckingcolors[2], label='LTE+PERCEPT 8 UAV-BS')
UOS_Label_15 = mpatches.Patch(color=uniquefuckingcolors[3], label='LTE+UOS 15 UAV-BS')
PERCEPT_Label_15 = mpatches.Patch(color=uniquefuckingcolors[4], label='LTE+PERCEPT 15 UAV-BS')

# Array with means of scenarios
#no small cells
means_2enb = [np.mean(ThroughputMean_100_2_NoUABS),np.mean(ThroughputMean_100_2_UABS),np.mean(ThroughputMean_100_2_UABS_pred),
              np.mean(Metric_Mean_100_NSMALL_15UABS_UOS),np.mean(Metric_Mean_100_NSMALL_15UABS_Percept),
              np.mean(ThroughputMean_200_2_NoUABS),np.mean(ThroughputMean_200_2_UABS),np.mean(ThroughputMean_200_2_UABS_pred),
              np.mean(Metric_Mean_200_NSMALL_15UABS_UOS),np.mean(Metric_Mean_200_NSMALL_15UABS_Percept)]

#small cells
means_4enb = [np.mean(ThroughputMean_100_6_NoUABS),np.mean(ThroughputMean_100_6_UABS),np.mean(ThroughputMean_100_4_UABS_pred),
              np.mean(Metric_Mean_100_SMALL_15UABS_UOS),np.mean(Metric_Mean_100_SMALL_15UABS_Percept),
              np.mean(ThroughputMean_200_6_NoUABS),np.mean(ThroughputMean_200_6_UABS),np.mean(ThroughputMean_200_4_UABS_pred),
              np.mean(Metric_Mean_200_SMALL_15UABS_UOS),np.mean(Metric_Mean_200_SMALL_15UABS_Percept)]

# Array with y error of scenarios
yerr_2enb =  [mean_confidence_interval(ThroughputMean_100_2_NoUABS),mean_confidence_interval(ThroughputMean_100_2_UABS),mean_confidence_interval(ThroughputMean_100_2_UABS_pred),
              mean_confidence_interval(Metric_Mean_100_SMALL_15UABS_UOS),mean_confidence_interval(Metric_Mean_100_SMALL_15UABS_Percept),
              mean_confidence_interval(ThroughputMean_200_2_NoUABS),mean_confidence_interval(ThroughputMean_200_2_UABS),mean_confidence_interval(ThroughputMean_200_2_UABS_pred),
              mean_confidence_interval(Metric_Mean_200_NSMALL_15UABS_UOS),mean_confidence_interval(Metric_Mean_200_NSMALL_15UABS_Percept)]

yerr_4enb =  [mean_confidence_interval(ThroughputMean_100_6_NoUABS),mean_confidence_interval(ThroughputMean_100_6_UABS),mean_confidence_interval(ThroughputMean_100_4_UABS_pred),
              mean_confidence_interval(Metric_Mean_100_SMALL_15UABS_UOS),mean_confidence_interval(Metric_Mean_100_SMALL_15UABS_Percept),
              mean_confidence_interval(ThroughputMean_200_6_NoUABS),mean_confidence_interval(ThroughputMean_200_6_UABS),mean_confidence_interval(ThroughputMean_200_4_UABS_pred),
              mean_confidence_interval(Metric_Mean_200_SMALL_15UABS_UOS),mean_confidence_interval(Metric_Mean_200_SMALL_15UABS_Percept)]




# Create a figure instance  
fig2 = plt.figure(2, figsize=(11,9))

# Create an axes instance
ax2 = fig2.add_subplot(221)
ax3 = fig2.add_subplot(222)


# use a known color palette (see..)
pal = sns.color_palette("Set2")

bars = ax2.bar(np.arange(10), means_2enb, yerr= yerr_2enb, error_kw=dict(lw=2, capsize=5, capthick=2),
               width =0.8, color= uniquefuckingcolors, align="center", tick_label=scenarios)
ax2.set(xlabel='Number of Users', ylabel='Jitter (ms)')
ax2.xaxis.get_label().set_fontsize(14)
ax2.yaxis.get_label().set_fontsize(14)
ax2.legend(handles=[LTE_Label, UOS_Label, PERCEPT_Label,UOS_Label_15,PERCEPT_Label_15])
ax2.set_title("4 TBS + No Small Cells", fontsize=16)
ax2.grid(color='green', ls = 'dotted')


bars2 = ax3.bar(np.arange(10),means_4enb, yerr= yerr_4enb, error_kw=dict(lw=2, capsize=5, capthick=2), 
                width =0.8, color= uniquefuckingcolors, align="center", tick_label=scenarios)
ax3.set(xlabel='Number of Users', ylabel='Jitter (ms)')
ax3.xaxis.get_label().set_fontsize(14)
ax3.yaxis.get_label().set_fontsize(14)
ax3.set_title("4 TBS + Small Cells", fontsize=16)
ax3.legend(handles=[LTE_Label, UOS_Label, PERCEPT_Label,UOS_Label_15,PERCEPT_Label_15])
ax3.grid(color='green', ls = 'dotted')



plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.tight_layout(pad=3.0, w_pad=2, h_pad=1.0)



plt.savefig("Graph_Avg_Jitter_PERCEPT.pdf", format='pdf', dpi=1000, bbox_inches = "tight")
#plt.show()


#---------------------------------------------------------------------------------------------------
#--------------------------Statistics---------------------------------------------------------------

test= statistics.mean(Throughput)
tmean= np.mean(ThroughputMean_100_6_NoUABS)

print(scenarios1[0] + ": "+ str(np.mean(ThroughputMean_100_6_NoUABS)))

#test1= statistics.mean(Throughput1)
tmean1= np.mean(ThroughputMean_100_6_UABS)

print(scenarios1[1] + ": "+ str(np.mean(ThroughputMean_100_6_UABS)))

#Comparing with LTE with UOS

ImprovRatio =tmean1/tmean

print("Improvement ratio LTE/UOS: " + str(ImprovRatio))  # improvement ratio = value after change / value before change

print("Improvement %: " + str( 100 * (ImprovRatio - 1)) +"%") 

#Comparing with UOS with Percept

tmean_pred= np.mean(ThroughputMean_100_4_UABS_pred)

print(scenarios1[2] + ": "+ str(np.mean(ThroughputMean_100_4_UABS_pred)))

ImprovRatio_Percept =tmean_pred/tmean1

print("Improvement ratio UOS/Percept: " + str(ImprovRatio_Percept))  # improvement ratio = value after change / value before change

print("Improvement %: " + str( 100 * (ImprovRatio_Percept - 1)) +"%"  + "\n") 

#-------------------------------------------------------------------------------------

#test= statistics.mean(Throughput)
tmean2= np.mean(ThroughputMean_200_6_NoUABS)

print(scenarios1[3] + ": "+ str(np.mean(ThroughputMean_200_6_NoUABS)))

#test1= statistics.mean(Throughput1)
tmean3= np.mean(ThroughputMean_200_6_UABS)

print(scenarios1[4] + ": "+ str(np.mean(ThroughputMean_200_6_UABS)))

#Comparing with LTE with UOS

ImprovRatio1 =tmean3/tmean2

print("Improvement ratio LTE/UOS: " + str(ImprovRatio1))  # improvement ratio = value after change / value before change

print("Improvement %: " + str( 100 * (ImprovRatio1 - 1)) +"%") 

#Comparing with UOS with Percept

tmean_pred1= np.mean(ThroughputMean_200_4_UABS_pred)

print(scenarios1[5] + ": "+ str(np.mean(ThroughputMean_200_4_UABS_pred)))

ImprovRatio_Percept1 =tmean_pred1/tmean3

print("Improvement ratio UOS/Percept: " + str(ImprovRatio_Percept1))  # improvement ratio = value after change / value before change

print("Improvement %: " + str( 100 * (ImprovRatio_Percept1 - 1)) +"%"  + "\n") 


#-------------------------------------------------------------------------------------

#test= statistics.mean(Throughput)
tmean4= np.mean(ThroughputMean_100_2_NoUABS)

print(scenarios1[6] + ": "+ str(np.mean(ThroughputMean_100_2_NoUABS)))

#test1= statistics.mean(Throughput1)
tmean5= np.mean(ThroughputMean_100_2_UABS)

print(scenarios1[7] + ": "+ str(np.mean(ThroughputMean_100_2_UABS)))

#Comparing with LTE with UOS

ImprovRatio2 =tmean5/tmean4

print("Improvement ratio LTE/UOS: " + str(ImprovRatio2))  # improvement ratio = value after change / value before change

print("Improvement %: " + str( 100 * (ImprovRatio2 - 1)) +"%") 

#Comparing with UOS with Percept

tmean_pred2= np.mean(ThroughputMean_100_2_UABS_pred)

print(scenarios1[8] + ": "+ str(np.mean(ThroughputMean_100_2_UABS_pred)))

ImprovRatio_Percept2 =tmean_pred2/tmean5

print("Improvement ratio UOS/Percept: " + str(ImprovRatio_Percept2))  # improvement ratio = value after change / value before change

print("Improvement %: " + str( 100 * (ImprovRatio_Percept2 - 1)) +"%"  + "\n") 


#-------------------------------------------------------------------------------------

#test= statistics.mean(Throughput)
tmean6= np.mean(ThroughputMean_200_2_NoUABS)

print(scenarios1[9] + ": "+ str(np.mean(ThroughputMean_200_2_NoUABS)))

#test1= statistics.mean(Throughput1)
tmean7= np.mean(ThroughputMean_200_2_UABS)

print(scenarios1[10] + ": "+ str(np.mean(ThroughputMean_200_2_UABS)))

#Comparing with LTE with UOS

ImprovRatio3 =tmean7/tmean6

print("Improvement ratio LTE/UOS: "  + str(ImprovRatio3))  # improvement ratio = value after change / value before change

print("Improvement %: " + str( 100 * (ImprovRatio3 - 1)) +"%") 

#Comparing with UOS with Percept

tmean_pred3= np.mean(ThroughputMean_200_2_UABS_pred)

print(scenarios1[11] + ": "+ str(np.mean(ThroughputMean_200_2_UABS_pred)))

ImprovRatio_Percept3 =tmean_pred3/tmean7

print("Improvement ratio UOS/Percept: " + str(ImprovRatio_Percept3))  # improvement ratio = value after change / value before change

print("Improvement %: " + str( 100 * (ImprovRatio_Percept3 - 1)) +"%"  + "\n") 


#-------------------------------------------------------------------------------------

ImprovTotal = (ImprovRatio+ImprovRatio1+ImprovRatio2+ImprovRatio3)/4
print("Improvement LTE vs UOS")
print("Improvement ratio Todos Scenarios Total: " + str(ImprovTotal)) 
print("Improvement % Todos Scenarios Total:: " + str( 100 * (ImprovTotal - 1)) +"%" + "\n") 


ImprovTotal2 = (ImprovRatio_Percept+ImprovRatio_Percept1+ImprovRatio_Percept2+ImprovRatio_Percept3)/4
print("Improvement UOS vs PERCEPT")
print("Improvement ratio Todos Scenarios Total: " + str(ImprovTotal2)) 
print("Improvement % Todos Scenarios Total:: " + str( 100 * (ImprovTotal2 - 1)) +"%") 