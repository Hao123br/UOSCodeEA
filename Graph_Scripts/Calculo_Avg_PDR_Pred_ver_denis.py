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
import matplotlib.ticker as mtick

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
                    Throughputarr[i] = Throughput[j]

        ThroughputMean[i] = (Throughputarr[i])
    #fairness_index_LTE = jfi(ThroughputMean)
    
    return ThroughputMean

#path= '/run/user/1000/gvfs/sftp:host=gercom.ddns.net,port=8372'
path= ''


#smallcells
#LTE
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/LTE/4enB_200U/plot_PDR_200_4_0') as MeanThroughput:

    data1 = np.array(list((int(Run),int(time), float(Throughput)) for Run, time, Throughput in csv.reader(MeanThroughput, delimiter= ' ')))
#UOS 8UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_200U/plot_PDR_200_4_8') as MeanThroughputUABS:

    data2 = np.array(list((int(Run1),int(time1), float(Throughput1)) for Run1, time1, Throughput1 in csv.reader(MeanThroughputUABS, delimiter= ' ')))    
#PERCEPT 8UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_8UABS_200U/plot_PDR_200_4_8') as MeanThroughputUABS7:

    data12 = np.array(list((int(Run11),int(time11), float(Throughput11)) for Run11, time11, Throughput11 in csv.reader(MeanThroughputUABS7, delimiter= ' '))) 
#UOS 15UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_15UABS_200U/plot_PDR_200_4_15') as MeanThroughputUABS8:

    data13 = np.array(list((int(Run12),int(time12), float(Throughput12)) for Run12, time12, Throughput12 in csv.reader(MeanThroughputUABS8, delimiter= ' ')))    
#PERCEPT 15UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_15UABS_200U/plot_PDR_200_4_15') as MeanThroughputUABS9:

    data14 = np.array(list((int(Run13),int(time13), float(Throughput13)) for Run13, time13, Throughput13 in csv.reader(MeanThroughputUABS9, delimiter= ' '))) 
#UOS 4UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_4UABS_200U/plot_PDR_200_4_4') as MeanThroughputUABS16:

    data21 = np.array(list((int(Run20),int(time20), float(Throughput20)) for Run20, time20, Throughput20 in csv.reader(MeanThroughputUABS16, delimiter= ' ')))    
#PERCEPT 4UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_4UABS_200U/plot_PDR_200_4_4') as MeanThroughputUABS17:

    data22 = np.array(list((int(Run21),int(time21), float(Throughput21)) for Run21, time21, Throughput21 in csv.reader(MeanThroughputUABS17, delimiter= ' ')))

#no_smallcell
#LTE
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/LTE/4enB_200U/plot_PDR_200_4_0') as MeanThroughput1:

    data3 = np.array(list((int(Run2),int(time2), float(Throughput2)) for Run2, time2, Throughput2 in csv.reader(MeanThroughput1, delimiter= ' ')))
#UOS 8UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_200U/plot_PDR_200_4_8') as MeanThroughputUABS1:

    data4 = np.array(list((int(Run3),int(time3), float(Throughput3)) for Run3, time3, Throughput3 in csv.reader(MeanThroughputUABS1, delimiter= ' ')))   
#PERCEPT 8UABS 
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_8UABS_200U/plot_PDR_200_4_8') as MeanThroughputUABS6:

    data11 = np.array(list((int(Run10),int(time10), float(Throughput10)) for Run10, time10, Throughput10 in csv.reader(MeanThroughputUABS6, delimiter= ' ')))  
#UOS 15UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_15UABS_200U/plot_PDR_200_4_15') as MeanThroughputUABS10:

    data15 = np.array(list((int(Run14),int(time14), float(Throughput14)) for Run14, time14, Throughput14 in csv.reader(MeanThroughputUABS10, delimiter= ' ')))   
#PERCEPT 15UABS 
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_15UABS_200U/plot_PDR_200_4_15') as MeanThroughputUABS11:

    data16 = np.array(list((int(Run15),int(time15), float(Throughput15)) for Run15, time15, Throughput15 in csv.reader(MeanThroughputUABS11, delimiter= ' ')))  
#UOS 4UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_4UABS_200U/plot_PDR_200_4_4') as MeanThroughputUABS18:

    data23 = np.array(list((int(Run22),int(time22), float(Throughput22)) for Run22, time22, Throughput22 in csv.reader(MeanThroughputUABS18, delimiter= ' ')))    
#PERCEPT 4UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_4UABS_200U/plot_PDR_200_4_4') as MeanThroughputUABS19:

    data24 = np.array(list((int(Run23),int(time23), float(Throughput23)) for Run23, time23, Throughput23 in csv.reader(MeanThroughputUABS19, delimiter= ' ')))


#smallcell
#LTE
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/LTE/4enB_100U/plot_PDR_100_4_0') as MeanThroughput2:

    data5 = np.array(list((int(Run4),int(time4), float(Throughput4)) for Run4, time4, Throughput4 in csv.reader(MeanThroughput2, delimiter= ' ')))
#UOS 8UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_100U/plot_PDR_100_4_8') as MeanThroughputUABS2:

    data6 = np.array(list((int(Run5),int(time5), float(Throughput5)) for Run5, time5, Throughput5 in csv.reader(MeanThroughputUABS2, delimiter= ' ')))  
#PERCEPT 8UABS    
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_8UABS_100U/plot_PDR_100_4_8') as MeanThroughputUABS5:

    data10 = np.array(list((int(Run9),int(time9), float(Throughput9)) for Run9, time9, Throughput9 in csv.reader(MeanThroughputUABS5, delimiter= ' ')))     
#UOS 15UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_15UABS_100U/plot_PDR_100_4_15') as MeanThroughputUABS12:

    data17 = np.array(list((int(Run16),int(time16), float(Throughput16)) for Run16, time16, Throughput16 in csv.reader(MeanThroughputUABS12, delimiter= ' ')))  
#PERCEPT 15UABS    
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_15UABS_100U/plot_PDR_100_4_15') as MeanThroughputUABS13:

    data18 = np.array(list((int(Run17),int(time17), float(Throughput17)) for Run17, time17, Throughput17 in csv.reader(MeanThroughputUABS13, delimiter= ' ')))        
#UOS 4UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_4UABS_100U/plot_PDR_100_4_4') as MeanThroughputUABS20:

    data25 = np.array(list((int(Run24),int(time24), float(Throughput24)) for Run24, time24, Throughput24 in csv.reader(MeanThroughputUABS20, delimiter= ' ')))    
#PERCEPT 4UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_4UABS_100U/plot_PDR_100_4_4') as MeanThroughputUABS21:

    data26 = np.array(list((int(Run25),int(time25), float(Throughput25)) for Run25, time25, Throughput25 in csv.reader(MeanThroughputUABS21, delimiter= ' ')))



#no_smallcell
#LTE 
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/LTE/4enB_100U/plot_PDR_100_4_0') as MeanThroughput3:

    data7 = np.array(list((int(Run6),int(time6), float(Throughput6)) for Run6, time6, Throughput6 in csv.reader(MeanThroughput3, delimiter= ' ')))
#UOS 8UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_100U/plot_PDR_100_4_8') as MeanThroughputUABS3:

    data8 = np.array(list((int(Run7),int(time7), float(Throughput7)) for Run7, time7, Throughput7 in csv.reader(MeanThroughputUABS3, delimiter= ' ')))
#PERCEPT 8UABS      
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_8UABS_100U/plot_PDR_100_4_8') as MeanThroughputUABS4:

    data9 = np.array(list((int(Run8),int(time8), float(Throughput8)) for Run8, time8, Throughput8 in csv.reader(MeanThroughputUABS4, delimiter= ' '))) 
#UOS 15UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_15UABS_100U/plot_PDR_100_4_15') as MeanThroughputUABS14:

    data19 = np.array(list((int(Run18),int(time18), float(Throughput18)) for Run18, time18, Throughput18 in csv.reader(MeanThroughputUABS14, delimiter= ' ')))
#PERCEPT 15UABS      
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_15UABS_100U/plot_PDR_100_4_15') as MeanThroughputUABS15:

    data20 = np.array(list((int(Run19),int(time19), float(Throughput19)) for Run19, time19, Throughput19 in csv.reader(MeanThroughputUABS15, delimiter= ' ')))  

#UOS 4UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_4UABS_100U/plot_PDR_100_4_4') as MeanThroughputUABS22:

    data27 = np.array(list((int(Run26),int(time26), float(Throughput26)) for Run26, time26, Throughput26 in csv.reader(MeanThroughputUABS22, delimiter= ' ')))    
#PERCEPT 4UABS
with open(path+'/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_4UABS_100U/plot_PDR_100_4_4') as MeanThroughputUABS23:

    data28 = np.array(list((int(Run27),int(time27), float(Throughput27)) for Run27, time27, Throughput27 in csv.reader(MeanThroughputUABS23, delimiter= ' ')))



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

Run20, time20, Throughput20 = data21.T
Run21, time21, Throughput21 = data22.T
Run22, time22, Throughput22 = data23.T
Run23, time23, Throughput23 = data24.T
Run24, time24, Throughput24 = data25.T
Run25, time25, Throughput25 = data26.T
Run26, time26, Throughput26 = data27.T
Run27, time27, Throughput27 = data28.T

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

NRuns_scen21 = int(max(Run20))
NRuns_scen22 = int(max(Run21))
NRuns_scen23 = int(max(Run22))
NRuns_scen24 = int(max(Run23))
NRuns_scen25 = int(max(Run24))
NRuns_scen26 = int(max(Run25))
NRuns_scen27 = int(max(Run26))
NRuns_scen28 = int(max(Run27))


ThroughputMean_200_6_NoUABS= meanMetric(NUsers2,NRuns_scen1,Run,time,Throughput) #4enb 
#
ThroughputMean_200_6_UABS= meanMetric(NUsers2,NRuns_scen2,Run1,time1,Throughput1)  #4enb 

ThroughputMean_200_2_NoUABS= meanMetric(NUsers2,NRuns_scen3,Run2,time2,Throughput2)  #2enb 

ThroughputMean_200_2_UABS= meanMetric(NUsers2,NRuns_scen4,Run3,time3,Throughput3)  #2enb 

ThroughputMean_200_2_UABS_pred= meanMetric(NUsers2,NRuns_scen11,Run10, time10, Throughput10)  #2enb 

ThroughputMean_200_4_UABS_pred= meanMetric(NUsers2,NRuns_scen12,Run11, time11, Throughput11)  #4enb
#### 15UABS
Metric_Mean_200_SMALL_15UABS_UOS= meanMetric(NUsers2,NRuns_scen13,Run12, time12, Throughput12)  # small cell

Metric_Mean_200_SMALL_15UABS_Percept= meanMetric(NUsers2,NRuns_scen14,Run13, time13, Throughput13)  #small cell

Metric_Mean_200_NSMALL_15UABS_UOS= meanMetric(NUsers2,NRuns_scen15,Run14, time14, Throughput14)  # no small cell

Metric_Mean_200_NSMALL_15UABS_Percept= meanMetric(NUsers2,NRuns_scen16,Run15, time15, Throughput15)  #no small cell
#### 4UABS
Metric_Mean_200_SMALL_4UABS_UOS= meanMetric(NUsers2,NRuns_scen21,Run20, time20, Throughput20)  # small cell

Metric_Mean_200_SMALL_4UABS_Percept= meanMetric(NUsers2,NRuns_scen22,Run21, time21, Throughput21)  #small cell

Metric_Mean_200_NSMALL_4UABS_UOS= meanMetric(NUsers2,NRuns_scen23,Run22, time22, Throughput22)  # no small cell

Metric_Mean_200_NSMALL_4UABS_Percept= meanMetric(NUsers2,NRuns_scen24,Run23, time23, Throughput23)  #no small cell



ThroughputMean_100_6_NoUABS= meanMetric(NUsers,NRuns_scen5,Run4, time4, Throughput4) #4enb 
#
ThroughputMean_100_6_UABS= meanMetric(NUsers,NRuns_scen6,Run5, time5, Throughput5)  #4enb 

ThroughputMean_100_2_NoUABS= meanMetric(NUsers,NRuns_scen7,Run6, time6, Throughput6)  #2enb 

ThroughputMean_100_2_UABS= meanMetric(NUsers,NRuns_scen8,Run7, time7, Throughput7)  #2enb 

ThroughputMean_100_2_UABS_pred= meanMetric(NUsers,NRuns_scen9,Run8, time8, Throughput8)  #2enb 

ThroughputMean_100_4_UABS_pred= meanMetric(NUsers,NRuns_scen10,Run9, time9, Throughput9)  #4enb
## 15 UABS
Metric_Mean_100_SMALL_15UABS_UOS= meanMetric(NUsers,NRuns_scen17,Run16, time16, Throughput16)  #Small cell

Metric_Mean_100_SMALL_15UABS_Percept= meanMetric(NUsers,NRuns_scen18,Run17, time17, Throughput17)  #Small cell

Metric_Mean_100_NSMALL_15UABS_UOS= meanMetric(NUsers,NRuns_scen19,Run18, time18, Throughput18)  # No small cell

Metric_Mean_100_NSMALL_15UABS_Percept= meanMetric(NUsers,NRuns_scen20,Run19, time19, Throughput19)  # No small cell
# 4 UABS
Metric_Mean_100_SMALL_4UABS_UOS= meanMetric(NUsers,NRuns_scen25,Run24, time24, Throughput24)  #Small cell

Metric_Mean_100_SMALL_4UABS_Percept= meanMetric(NUsers,NRuns_scen26,Run25, time25, Throughput25)  #Small cell

Metric_Mean_100_NSMALL_4UABS_UOS= meanMetric(NUsers,NRuns_scen27,Run26, time26, Throughput26)  # No small cell

Metric_Mean_100_NSMALL_4UABS_Percept= meanMetric(NUsers,NRuns_scen28,Run27, time27, Throughput27)  # No small cell



 
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
#no small cells
means_LTE_100 = [np.mean(ThroughputMean_100_2_NoUABS)]

means_UOS_100_8 = [np.mean(ThroughputMean_100_2_UABS)]

means_Percept_100_8 = [np.mean(ThroughputMean_100_2_UABS_pred)]

means_UOS_100_15 = [np.mean(Metric_Mean_100_NSMALL_15UABS_UOS)]

means_Percept_100_15 = [np.mean(Metric_Mean_100_NSMALL_15UABS_Percept)]

means_UOS_100_4 = [np.mean(Metric_Mean_100_NSMALL_4UABS_UOS)]

means_Percept_100_4 = [np.mean(Metric_Mean_100_NSMALL_4UABS_Percept)]

means_LTE_200 = [np.mean(ThroughputMean_200_2_NoUABS)]

means_UOS_200_8 = [np.mean(ThroughputMean_200_2_UABS)]

means_Percept_200_8 = [np.mean(ThroughputMean_200_2_UABS_pred)]

means_UOS_200_15 = [np.mean(Metric_Mean_200_NSMALL_15UABS_UOS)]

means_Percept_200_15 = [np.mean(Metric_Mean_200_NSMALL_15UABS_Percept)]

means_UOS_200_4 = [np.mean(Metric_Mean_200_NSMALL_4UABS_UOS)]

means_Percept_200_4 = [np.mean(Metric_Mean_200_NSMALL_4UABS_Percept)]

#small cells
means_LTE_100_SC = [np.mean(ThroughputMean_100_6_NoUABS)]

means_UOS_100_8_SC = [np.mean(ThroughputMean_100_6_UABS)]

means_Percept_100_8_SC = [np.mean(ThroughputMean_100_4_UABS_pred)]

means_UOS_100_15_SC = [np.mean(Metric_Mean_100_SMALL_15UABS_UOS)]

means_Percept_100_15_SC = [np.mean(Metric_Mean_100_SMALL_15UABS_Percept)]

means_UOS_100_4_SC = [np.mean(Metric_Mean_100_SMALL_4UABS_UOS)]

means_Percept_100_4_SC = [np.mean(Metric_Mean_100_SMALL_4UABS_Percept)]

means_LTE_200_SC = [np.mean(ThroughputMean_200_6_NoUABS)]

means_UOS_200_8_SC = [np.mean(ThroughputMean_200_6_UABS)]

means_Percept_200_8_SC = [np.mean(ThroughputMean_200_4_UABS_pred)]

means_UOS_200_15_SC = [np.mean(Metric_Mean_200_SMALL_15UABS_UOS)]

means_Percept_200_15_SC = [np.mean(Metric_Mean_200_SMALL_15UABS_Percept)]

means_UOS_200_4_SC = [np.mean(Metric_Mean_200_SMALL_4UABS_UOS)]

means_Percept_200_4_SC = [np.mean(Metric_Mean_200_SMALL_4UABS_Percept)]

# Array with y error of scenarios
#no small cells
yerr_LTE_100 =  [mean_confidence_interval(ThroughputMean_100_2_NoUABS)]

yerr_UOS_100_8 = [mean_confidence_interval(ThroughputMean_100_2_UABS)]
 
yerr_Percept_100_8 = [mean_confidence_interval(ThroughputMean_100_2_UABS_pred)]

yerr_UOS_100_15 = [mean_confidence_interval(Metric_Mean_100_NSMALL_15UABS_UOS)]
 
yerr_Percept_100_15 = [mean_confidence_interval(Metric_Mean_100_NSMALL_15UABS_Percept)]

yerr_UOS_100_4 = [mean_confidence_interval(Metric_Mean_100_NSMALL_4UABS_UOS)]
 
yerr_Percept_100_4 = [mean_confidence_interval(Metric_Mean_100_NSMALL_4UABS_Percept)]

yerr_LTE_200 = [mean_confidence_interval(ThroughputMean_200_2_NoUABS)]

yerr_UOS_200_8 = [mean_confidence_interval(ThroughputMean_200_2_UABS)]
 
yerr_Percept_200_8 = [mean_confidence_interval(ThroughputMean_200_2_UABS_pred)]

yerr_UOS_200_15 = [mean_confidence_interval(Metric_Mean_200_NSMALL_15UABS_UOS)]
 
yerr_Percept_200_15 = [mean_confidence_interval(Metric_Mean_200_NSMALL_15UABS_Percept)]

yerr_UOS_200_4 = [mean_confidence_interval(Metric_Mean_200_NSMALL_4UABS_UOS)]
 
yerr_Percept_200_4 = [mean_confidence_interval(Metric_Mean_200_NSMALL_4UABS_Percept)]

#small cells

yerr_LTE_100_SC =  [mean_confidence_interval(ThroughputMean_100_6_NoUABS)]

yerr_UOS_100_8_SC = [mean_confidence_interval(ThroughputMean_100_6_UABS)]
 
yerr_Percept_100_8_SC = [mean_confidence_interval(ThroughputMean_100_4_UABS_pred)]

yerr_UOS_100_15_SC = [mean_confidence_interval(Metric_Mean_100_SMALL_15UABS_UOS)]
 
yerr_Percept_100_15_SC = [mean_confidence_interval(Metric_Mean_100_SMALL_15UABS_Percept)]

yerr_UOS_100_4_SC = [mean_confidence_interval(Metric_Mean_100_SMALL_4UABS_UOS)]
 
yerr_Percept_100_4_SC = [mean_confidence_interval(Metric_Mean_100_SMALL_4UABS_Percept)]

yerr_LTE_200_SC = [mean_confidence_interval(ThroughputMean_200_6_NoUABS)]

yerr_UOS_200_8_SC = [mean_confidence_interval(ThroughputMean_200_6_UABS)]
 
yerr_Percept_200_8_SC = [mean_confidence_interval(ThroughputMean_200_4_UABS_pred)]

yerr_UOS_200_15_SC = [mean_confidence_interval(Metric_Mean_200_SMALL_15UABS_UOS)]
 
yerr_Percept_200_15_SC = [mean_confidence_interval(Metric_Mean_200_SMALL_15UABS_Percept)]

yerr_UOS_200_4_SC = [mean_confidence_interval(Metric_Mean_200_SMALL_4UABS_UOS)]
 
yerr_Percept_200_4_SC = [mean_confidence_interval(Metric_Mean_200_SMALL_4UABS_Percept)]


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
ax2.bar(np.arange(1),means_LTE_100, yerr=yerr_LTE_100 ,width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2),label="LTE")
ax2.bar(np.arange(1)+0.6,means_UOS_100_4, yerr=yerr_UOS_100_4 ,width =0.6, color= uniquefuckingcolors[1], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+UOS")
ax2.bar(np.arange(1)+1.2,means_Percept_100_4, yerr=yerr_Percept_100_4 ,width =0.6, color= uniquefuckingcolors[2], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+Percept")
#8 UAV-BS
ax2.bar(np.arange(1)+3,means_LTE_100, yerr=yerr_LTE_100 ,width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax2.bar(np.arange(1)+3.6,means_UOS_100_8, yerr=yerr_UOS_100_8 ,width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax2.bar(np.arange(1)+4.2,means_Percept_100_8, yerr=yerr_Percept_100_8 ,width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
#15 UAV-BS
ax2.bar(np.arange(1)+6,means_LTE_100, yerr=yerr_LTE_100 ,width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax2.bar(np.arange(1)+6.6,means_UOS_100_15, yerr=yerr_UOS_100_15 ,width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax2.bar(np.arange(1)+7.2,means_Percept_100_15, yerr=yerr_Percept_100_15 ,width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))

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
ax3.bar(np.arange(1),means_LTE_200, yerr=yerr_LTE_200 ,width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2),label="LTE")
ax3.bar(np.arange(1)+0.6,means_UOS_200_4, yerr=yerr_UOS_200_4 ,width =0.6, color= uniquefuckingcolors[1], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+UOS")
ax3.bar(np.arange(1)+1.2,means_Percept_100_4, yerr=yerr_Percept_100_4 ,width =0.6, color= uniquefuckingcolors[2], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+Percept")
#8 UAV-BS
ax3.bar(np.arange(1)+3,means_LTE_200, yerr=yerr_LTE_200 ,width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax3.bar(np.arange(1)+3.6,means_UOS_200_8, yerr=yerr_UOS_200_8 ,width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax3.bar(np.arange(1)+4.2,means_Percept_200_8, yerr=yerr_Percept_200_8 ,width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
#15 UAV-BS
ax3.bar(np.arange(1)+6,means_LTE_200, yerr=yerr_LTE_200 ,width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax3.bar(np.arange(1)+6.6,means_UOS_200_15, yerr=yerr_UOS_200_15 ,width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax3.bar(np.arange(1)+7.2,means_Percept_200_15, yerr=yerr_Percept_200_15 ,width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))

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
ax4.bar(np.arange(1),means_LTE_100_SC, yerr=yerr_LTE_100_SC ,width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2),label="LTE")
ax4.bar(np.arange(1)+0.6,means_UOS_100_4_SC, yerr=yerr_UOS_100_4_SC ,width =0.6, color= uniquefuckingcolors[1], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+UOS")
ax4.bar(np.arange(1)+1.2,means_Percept_100_4_SC, yerr=yerr_Percept_100_4_SC ,width =0.6, color= uniquefuckingcolors[2], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+Percept")
#8 UAV-BS
ax4.bar(np.arange(1)+3,means_LTE_100_SC, yerr=yerr_LTE_100_SC ,width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax4.bar(np.arange(1)+3.6,means_UOS_100_8_SC, yerr=yerr_UOS_100_8_SC ,width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax4.bar(np.arange(1)+4.2,means_Percept_100_8_SC, yerr=yerr_Percept_100_8_SC ,width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
#15 UAV-BS
ax4.bar(np.arange(1)+6,means_LTE_100_SC, yerr=yerr_LTE_100_SC ,width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax4.bar(np.arange(1)+6.6,means_UOS_100_15_SC, yerr=yerr_UOS_100_15_SC ,width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax4.bar(np.arange(1)+7.2,means_Percept_100_15_SC, yerr=yerr_Percept_100_15_SC ,width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))

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
ax5.bar(np.arange(1),means_LTE_200_SC, yerr=yerr_LTE_200_SC ,width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2),label="LTE")
ax5.bar(np.arange(1)+0.6,means_UOS_200_4_SC, yerr=yerr_UOS_200_4_SC ,width =0.6, color= uniquefuckingcolors[1], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+UOS")
ax5.bar(np.arange(1)+1.2,means_Percept_100_4_SC, yerr=yerr_Percept_100_4_SC ,width =0.6, color= uniquefuckingcolors[2], align="center", error_kw=dict(lw=2, capsize=5, capthick=2), label="LTE+Percept")
#8 UAV-BS
ax5.bar(np.arange(1)+3,means_LTE_200_SC, yerr=yerr_LTE_200_SC ,width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax5.bar(np.arange(1)+3.6,means_UOS_200_8_SC, yerr=yerr_UOS_200_8_SC ,width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax5.bar(np.arange(1)+4.2,means_Percept_200_8_SC, yerr=yerr_Percept_200_8_SC ,width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
#15 UAV-BS
ax5.bar(np.arange(1)+6,means_LTE_200_SC, yerr=yerr_LTE_200_SC ,width =0.6, color= uniquefuckingcolors[0], align="center", error_kw=dict(lw=2, capsize=5, capthick=2))
ax5.bar(np.arange(1)+6.6,means_UOS_200_15_SC, yerr=yerr_UOS_200_15_SC ,width =0.6, color= uniquefuckingcolors[1], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))
ax5.bar(np.arange(1)+7.2,means_Percept_200_15_SC, yerr=yerr_Percept_200_15_SC ,width =0.6, color= uniquefuckingcolors[2], align="center",error_kw=dict(lw=2, capsize=5, capthick=2))

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
    
            print("Improvement ratio " + str(scen_comp[i]) +" : "  + str(ImprovRatio_LTE_vs_UAV_100))  # improvement ratio = value after change / value before change
        
            print("Improvement %: " + str( 100 * (ImprovRatio_LTE_vs_UAV_100 - 1)) +"%") 
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