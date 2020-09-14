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


scenarios_path = [                  
                  # '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_4UABS_200U/',
                  # '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_200U/',
                  # '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_15UABS_200U/',
                  # '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_4UABS_200U/',
                  # '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_8UABS_200U/',
                  # '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_15UABS_200U/',
                  
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_4UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_4UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_8UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_15UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_15UABS_200U/']

                 

No_scenarios= len(scenarios_path)
uabs = np.zeros((3,33))
data_to_plot = [None] * No_scenarios
data_to_plot2 = [None] * No_scenarios

for j in range(0,No_scenarios):
    for i in range(0,33):
    #        print(scenarios_dir[j])
        info_users_UABS = pd.read_csv(path + scenarios_path[j] +'UE_info_UABS.~{0}~'.format(i),names=["Time","UE_ID","Cell_ID","sinr_db"])

  
    # 4 enBs and 8 Small Cells
    #if scenarios contains small cells 4+8 = 12, after 12 ids corresponds to UABSs.
        # if (scenarios_path[j] == scenarios_path[2] or scenarios_path[j] == scenarios_path[3] or scenarios_path[j] == scenarios_path[6] or scenarios_path[j] == scenarios_path[7] or scenarios_path[j] == scenarios_path[10] or scenarios_path[j] == scenarios_path[11] or scenarios_path[j] == scenarios_path[14] or scenarios_path[j] == scenarios_path[15] or scenarios_path[j] == scenarios_path[18] or scenarios_path[j] == scenarios_path[19] or scenarios_path[j] == scenarios_path[22] or scenarios_path[j] == scenarios_path[23]):
        UABS_CellID_Filter = info_users_UABS["Cell_ID"] > 12
        eNB_CellID_Filter = info_users_UABS["Cell_ID"] < 13
        # else:
            # UABS_CellID_Filter = info_users_UABS["Cell_ID"] > 4
            # eNB_CellID_Filter = info_users_UABS["Cell_ID"] < 5 
            
        Users_Connected_UABS = info_users_UABS[UABS_CellID_Filter].nunique()
        Users_Connected_enB = info_users_UABS[eNB_CellID_Filter].nunique()
        percent_user =  Users_Connected_UABS['UE_ID']/info_users_UABS['UE_ID'].nunique()
        percent_user_enB =  Users_Connected_enB['UE_ID']/info_users_UABS['UE_ID'].nunique()
        uabs_served = Users_Connected_UABS['Cell_ID']
    #    print(percent_user)
        if len(Users_Connected_UABS) == 0:
                percent_user = 0
                uabs_served = 0
        uabs[0,i] = percent_user
        uabs[1,i] = uabs_served
        uabs[2,i] = percent_user_enB
        # Users_Connected_UABS = info_users_UABS.groupby('Cell_ID')['UE_ID'].nunique()
    #    Users_Connected_eNB = info_users_UABS[eNB_CellID_Filter].nunique()
        #Users_Connected_eNB = info_users_UABS[eNB_CellID_Filter].count()
        

#        print(uabs.mean(axis=1))
        data_to_plot[j] = uabs[0]*100
        data_to_plot2[j] = 100-(uabs[0]*100)



#//-----------------Statistics Mean Percent Offloaded UOS and Percept---------------\\
        
mean_percent_Users_UOS_scens = np.mean(list([data_to_plot[0],data_to_plot[2],data_to_plot[4]]))
print("Percent users attended by UOS Scenarios: " + str(mean_percent_Users_UOS_scens))

mean_percent_Users_Percept_scens = np.mean(list([data_to_plot[1],data_to_plot[3],data_to_plot[5]]))
print("Percent users attended by Percept Scenarios: " + str(mean_percent_Users_Percept_scens))
      
    
#//------------------BoxPlot ------------------\\
   
   
# scen_names_BP = ['UOS\n200 Users\n4 UAV-BS',
#                  'Percept\n200 Users\n4 UAV-BS',
#                  'UOS\n200 Users\n8 UAV-BS',
#                  'Percept\n200 Users\n8 UAV-BS',
#                  'UOS\n200 Users\n15 UAV-BS',
#                  'Percept\n200 Users\n15 UAV-BS']    


scen_names_BP = ['4','8','15']  


green_diamond = dict(markerfacecolor='g', marker='D')

# Create a figure instance
fig = plt.figure(1, figsize=(12, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot 
bp = ax.boxplot(list([data_to_plot[0],data_to_plot[2],data_to_plot[4]]), positions = [0.5,2.5,4.5]  ,patch_artist = True, flierprops=green_diamond, notch=True)
bp_percept = ax.boxplot(list([data_to_plot[1],data_to_plot[3],data_to_plot[5]]), positions = [1,3,5], patch_artist = True, flierprops=green_diamond, notch=True)
for box in bp['boxes']:
    # change outline color
    box.set( color='#7c7f9e', linewidth=4)
    # change fill color
#    box.set( facecolor = '#1b9e77' )
    box.set( facecolor = '#cdd5e4' )
    
for box in bp_percept['boxes']:
    # change outline color
    box.set( color='#3867d6', linewidth=4)
    # change fill color
#    box.set( facecolor = '#1b9e77' )
    box.set( facecolor = '#45aaf2' )
            
#plt.title('Percentage users attended by UABS', fontsize=16)
plt.ylabel('Users Attended (%)', fontsize=18)
plt.xlabel('Number of UAV-BS', fontsize=18)
plt.yticks(fontsize=16) 

# plt.xticks(np.arange(0,6),scen_names_BP,fontsize=16, ha= 'center') 
plt.xticks([0.8,2.8,4.8],scen_names_BP,fontsize=16, ha= 'center')
plt.legend([bp["boxes"][0],bp_percept["boxes"][0]],['LTE+UOS','LTE+PERCEPT'],loc='upper left',fontsize=18) 
plt.grid(color='green',ls = 'dotted')


plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.tight_layout(pad=3.0, w_pad=2, h_pad=1.0) 
#plt.grid(True)
plt.savefig("Boxplot_Users_perc_UABS_UOS_PERCEPT.pdf", format='pdf', dpi=1000)



#//------------------Stack Plot ------------------\\

scen_names_SP = [
                 'UOS 200 Users | 4 UAV-BS',
         
                 'UOS 200 Users | 8 UAV-BS',
                 
                 'UOS 200 Users | 15 UAV-BS',
                 
                 'Percept 200 Users | 4 UAV-BS',
                 
                 'Percept 200 Users | 8 UAV-BS',
            
                 'Percept 200 Users | 15 UAV-BS']  


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
ax2.stackplot(range(0,33),data_to_plot2[0], data_to_plot[0], labels=['TBS','UAV-BS'],colors=Palette_Colors)
ax2.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax2.legend(loc='lower right')
ax2.set_title(scen_names_SP[0], fontsize=10)
#UOS - 8 UAV-BS - Small cells
#200 users
ax3.stackplot(range(0,33),data_to_plot2[2], data_to_plot[2], labels=['TBS','UAV-BS'],colors=Palette_Colors)
ax3.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax3.legend(loc='lower right')
ax3.set_title(scen_names_SP[1], fontsize=10)
#UOS - 15 UAV-BS - Small cells
#200 users
ax4.stackplot(range(0,33),data_to_plot2[4], data_to_plot[4], labels=['TBS','UAV-BS'],colors=Palette_Colors)
ax4.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax4.legend(loc='lower right')
ax4.set_title(scen_names_SP[2], fontsize=10)


#Percept - 4 UAV-BS - Small cells
#200 users
ax5.stackplot(range(0,33),data_to_plot2[1], data_to_plot[1], labels=['TBS','UAV-BS'],colors=Palette_Colors_Percept)
ax5.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax5.legend(loc='lower right')
ax5.set_title(scen_names_SP[3], fontsize=10)
#Percept - 8 UAV-BS - Small cells
#200 users
ax6.stackplot(range(0,33),data_to_plot2[3], data_to_plot[3], labels=['TBS','UAV-BS'],colors=Palette_Colors_Percept)
ax6.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax6.legend(loc='lower right')
ax6.set_title(scen_names_SP[4], fontsize=10)
#Percept - 15 UAV-BS - Small cells
#200 users
ax7.stackplot(range(0,33),data_to_plot2[5], data_to_plot[5], labels=['TBS','UAV-BS'],colors=Palette_Colors_Percept)
ax7.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax7.legend(loc='lower right')
ax7.set_title(scen_names_SP[5], fontsize=10)




plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.tight_layout(pad=3.0, w_pad=2, h_pad=1.0)
plt.savefig("Stackplot_Users_perc_UABS_PERCEPT_UOS.pdf", format='pdf', dpi=1000)
#plt.show()


