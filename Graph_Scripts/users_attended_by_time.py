#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:18:09 2020

@author: emanuel
"""


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def get_number_of_ues(path):
    folders = path.strip('/').split('/')
    sim_info = folders[-1].split('_')
    nUEs = int(sim_info[-1].strip('U'))
    return nUEs

def update_ues_status(status_matrix, time_data, ue_id_data, cell_id_data):
    data_size = len(time_data)
    first_id = ue_id_data.min()
    for i in range(data_size):
        time = int(time_data[i])
        ue_index = ue_id_data[i] - first_id
        cell_id = cell_id_data[i]
        if cell_id > 12:
            status_matrix[time][ue_index] = True

#info_users = pd.read_csv('LTEUEs_Log',names=["Time", "X", "Y", "Z","UE_ID","Cell_ID","sinr_Linear"])

#UABS 1 CellID 5
#UABS 2 CellID 6
#UABS 3 CellID 7
#UABS 4 CellID 8
#UABS 5 CellID 9
#UABS 6 CellID 10
#UABS 7 CellID 11
#UABS 8 CellID 12

# path= '/run/user/1000/gvfs/sftp:host=gercom.ddns.net,port=8372'
path= ''
root_path = '/home/emanuel/Desktop/IEEE_Article'

scenarios_path = [                  
                  # '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_4UABS_200U/',
                  # '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_200U/',
                  # '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_15UABS_200U/',
                  # '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_4UABS_200U/',
                  # '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_8UABS_200U/',
                  # '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_15UABS_200U/',
                  
                  '/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_100U/',
                  '/Sera_el_final_o_no/Small_Cells/Percept/4enB_8UABS_100U/',
                  '/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_200U/',
                  '/Sera_el_final_o_no/Small_Cells/Percept/4enB_8UABS_200U/',
                  '/results_300_and_400/Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_300U/',
                  '/results_300_and_400/Small_Cells/Percept/4enB_8UABS_300U/']

                 

No_scenarios= len(scenarios_path)
data_to_plot = []
data_to_plot2 = []

# data_dict['x'] = [x*5 for x in range(1,20)]

for j in range(0,No_scenarios):
    nUEs = get_number_of_ues(scenarios_path[j])
    percent_by_time_sum = np.zeros(shape=100)
    complete_iterations = 0
    for i in range(0,29):

        info_users_UABS = pd.read_csv(path + root_path + scenarios_path[j] +'UE_info_UABS.~{0}~'.format(i),names=["Time","UE_ID","Cell_ID","sinr_db"])
        sim_time = info_users_UABS["Time"].max()
        if sim_time < 99:
            continue #skip incomplete data

        #UE status matrix. 0 means connected to TBS and 1 means connected to UABS
        ue_connection_status = np.zeros(shape=(100, nUEs))
        update_ues_status(ue_connection_status, info_users_UABS["Time"],info_users_UABS["UE_ID"], info_users_UABS["Cell_ID"])

        #percent users served by UABS each second of current scenario iteration
        percent_by_time = ue_connection_status.sum(1) * 100 / nUEs
        percent_by_time_sum += percent_by_time
        complete_iterations += 1
    percent_by_time_all_iterations = percent_by_time_sum / complete_iterations
    data_to_plot.append(percent_by_time_all_iterations)
    data_to_plot2.append(100 - percent_by_time_all_iterations)

#//-----------------Statistics Mean Percent Offloaded UOS and Percept---------------\\
        
mean_percent_Users_UOS_scens = np.mean(list([data_to_plot[0],data_to_plot[2],data_to_plot[4]]))
print("Percent users attended by UOS Scenarios: " + str(mean_percent_Users_UOS_scens))

mean_percent_Users_Percept_scens = np.mean(list([data_to_plot[1],data_to_plot[3],data_to_plot[5]]))
print("Percent users attended by Percept Scenarios: " + str(mean_percent_Users_Percept_scens))
      
#//--------------------------------------Stack Plot ---------------------------------------------------------\\

scen_names_SP = [
                 'UOS 100 Users',
         
                 'UOS 200 Users',
                 
                 'UOS 300 Users',
                 
                 'Percept 100 Users',
                 
                 'Percept 200 Users',
            
                 'Percept 300 Users']  


fig2 = plt.figure(4, figsize=(15, 18)) 
    
# Create an axes instance
#200 users UOS and Percept
ax2 = fig2.add_subplot(6,3,1)
ax3 = fig2.add_subplot(6,3,2)
ax4 = fig2.add_subplot(6,3,3)
ax5 = fig2.add_subplot(6,3,4)
ax6 = fig2.add_subplot(6,3,5)
ax7 = fig2.add_subplot(6,3,6)


# use a known color palette (see..)
pal = sns.color_palette("Set2")
Palette_Colors  = ["#cdd5e4", "#7c7f9e"]
Palette_Colors_Percept  = ["#45aaf2", "#3867d6"]

# Make the plot

#UOS - 4 UAV-BS - Small cells
#200 users
ax2.stackplot(range(100),data_to_plot2[0], data_to_plot[0], labels=['TBS','UAV-BS'],colors=Palette_Colors)
ax2.set(xlabel='Time (s)', ylabel='User Attended (%)')
ax2.legend(loc='lower right')
ax2.set_title(scen_names_SP[0], fontsize=10)
#UOS - 8 UAV-BS - Small cells
#200 users
ax3.stackplot(range(100),data_to_plot2[2], data_to_plot[2], labels=['TBS','UAV-BS'],colors=Palette_Colors)
ax3.set(xlabel='Time (s)', ylabel='User Attended (%)')
ax3.legend(loc='lower right')
ax3.set_title(scen_names_SP[1], fontsize=10)
#UOS - 15 UAV-BS - Small cells
#200 users
ax4.stackplot(range(100),data_to_plot2[4], data_to_plot[4], labels=['TBS','UAV-BS'],colors=Palette_Colors)
ax4.set(xlabel='Time (s)', ylabel='User Attended (%)')
ax4.legend(loc='lower right')
ax4.set_title(scen_names_SP[2], fontsize=10)


#Percept - 4 UAV-BS - Small cells
#200 users
ax5.stackplot(range(100),data_to_plot2[1], data_to_plot[1], labels=['TBS','UAV-BS'],colors=Palette_Colors_Percept)
ax5.set(xlabel='Time (s)', ylabel='User Attended (%)')
ax5.legend(loc='lower right')
ax5.set_title(scen_names_SP[3], fontsize=10)
#Percept - 8 UAV-BS - Small cells
#200 users
ax6.stackplot(range(100),data_to_plot2[3], data_to_plot[3], labels=['TBS','UAV-BS'],colors=Palette_Colors_Percept)
ax6.set(xlabel='Time (s)', ylabel='User Attended (%)')
ax6.legend(loc='lower right')
ax6.set_title(scen_names_SP[4], fontsize=10)
#Percept - 15 UAV-BS - Small cells
#200 users
ax7.stackplot(range(100),data_to_plot2[5], data_to_plot[5], labels=['TBS','UAV-BS'],colors=Palette_Colors_Percept)
ax7.set(xlabel='Time (s)', ylabel='User Attended (%)')
ax7.legend(loc='lower right')
ax7.set_title(scen_names_SP[5], fontsize=10)




plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.tight_layout(pad=3.0, w_pad=2, h_pad=1.0)
plt.savefig("Stackplot_Users_perc_UABS_PERCEPT_UOS.pdf", format='pdf', dpi=1000)
#plt.show()


