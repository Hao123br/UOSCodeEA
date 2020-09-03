
130707
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
# scenarios_path = [#"/home/emanuel/v2_master_thesis_UOS/100U_LTE_2_UABS/UOSCode/results_thesis",
# #                  "/home/emanuel/v2_master_thesis_UOS/200U_LTE_2_UABS/UOSCode/results_thesis",
#                  "/home/emanuel/v2_master_thesis_UOS/100U_LTE_6_UABS/UOSCode/results_thesis/final_graphs", #check 33
#                  "/home/emanuel/v2_master_thesis_UOS/200U_LTE_6_UABS/UOSCode/results_thesis/final_graphs", #check 33
#                   "/home/emanuel/v2_master_thesis_UOS/100U_2_enB_6_UABS/UOSCode/results_thesis/final_graphs", #check 33
#                   "/home/emanuel/v2_master_thesis_UOS/200U_2_enB_6_UABS/UOSCode/results_thesis/final_graphs", #check 33
#                   "/home/emanuel/IEEE_Article/final_graphs/4enB_6UABS_100U/ueinfo",
#                  "/home/emanuel/IEEE_Article/final_graphs/4enB_6UABS_200U/ueinfo",
#                  "/home/emanuel/IEEE_Article/final_graphs/2enB_6UABS_100U/ueinfo",
#                  "/home/emanuel/IEEE_Article/final_graphs/2enB_6UABS_200U/ueinfo"]
# #                  "/home/emanuel/v2_master_thesis_UOS/100U_2_enB_6_UABS/UOSCode/results_thesis",
# #                  "/home/emanuel/v2_master_thesis_UOS/200U_2_enB_6_UABS/UOSCode/results_thesis",

scenarios_path = [                  
                  
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_4UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_4UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_4UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_4UABS_200U/',
                  
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_200U/',
                  
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_15UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_15UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_15UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_15UABS_200U/',
                  
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_4UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_4UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_4UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_4UABS_200U/',
                  
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_8UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_8UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_8UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_8UABS_200U/',
                  
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_15UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_15UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_15UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_15UABS_200U/']

                 

No_scenarios= len(scenarios_path)
uabs = np.zeros((3,32))
data_to_plot = [None] * No_scenarios
data_to_plot2 = [None] * No_scenarios

for j in range(0,No_scenarios):
    for i in range(0,32):
    #        print(scenarios_dir[j])
        info_users_UABS = pd.read_csv(path + scenarios_path[j] +'UE_info_UABS.~{0}~'.format(i),names=["Time","UE_ID","Cell_ID","sinr_db"])
#        info_users_UABS = pd.read_csv(scenarios_path[j] +'/UE_info_UABS_RUN#{0}'.format(i),names=["Time(s)","UE_ID","Cell_ID","sinr_db"])
#        info_users_UABS = pd.read_csv(r''+scenarios_path[j] +'/UE_info_UABS_RUN#{0}'.format(i),names=["Time(s)","UE_ID","Cell_ID","sinr_db"])
#        info_users_UABS = pd.read_csv(r'/home/emanuel/master_thesis_UOS/100U_LTE_2_UABS/UOSCode/results_thesis/UE_info_UABS_RUN#{0}'.format(i),names=["Time(s)","UE_ID","Cell_ID","sinr_db"])
#        info_users_UABS = pd.read_csv('/run/user/1001/gvfs/sftp:host=192.168.1.2,port=8372/home/emanuel/master_thesis_UOS/100U_LTE_2_UABS/UOSCode/results_thesis/UE_info_UABS_RUN#9',names=["Time(s)","UE_ID","Cell_ID","sinr_db"])
    
    # 4 enBs and 8 Small Cells
    #if scenarios contains small cells 4+8 = 12, after 12 ids corresponds to UABSs.
        if (scenarios_path[j] == scenarios_path[2] or scenarios_path[j] == scenarios_path[3] or scenarios_path[j] == scenarios_path[6] or scenarios_path[j] == scenarios_path[7] or scenarios_path[j] == scenarios_path[10] or scenarios_path[j] == scenarios_path[11] or scenarios_path[j] == scenarios_path[14] or scenarios_path[j] == scenarios_path[15] or scenarios_path[j] == scenarios_path[18] or scenarios_path[j] == scenarios_path[19] or scenarios_path[j] == scenarios_path[22] or scenarios_path[j] == scenarios_path[23]):
            UABS_CellID_Filter = info_users_UABS["Cell_ID"] > 12
            eNB_CellID_Filter = info_users_UABS["Cell_ID"] < 13
        else:
            UABS_CellID_Filter = info_users_UABS["Cell_ID"] > 4
            eNB_CellID_Filter = info_users_UABS["Cell_ID"] < 5 
            
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
        
    
#//------------------BoxPlot ------------------\\
   
   
scen_names_BP = ['','UOS\n100 Users\n4 UAV-BS\nNSC','UOS\n200 Users\n4 UAV-BS\nNSC', 
                 'UOS\n100 Users\n4 UAV-BS\nSC','UOS\n200 Users\n4 UAV-BS\nSC',
                 'UOS\n100 Users\n8 UAV-BS\nNSC','UOS\n200 Users\n8 UAV-BS\nNSC', 
                 'UOS\n100 Users\n8 UAV-BS\nSC','UOS\n200 Users\n8 UAV-BS\nSC',
                 'UOS\n100 Users\n15 UAV-BS\nNSC','UOS\n200 Users\n15 UAV-BS\nNSC', 
                 'UOS\n100 Users\n15 UAV-BS\nSC','UOS\n200 Users\n15 UAV-BS\nSC',
                 'Percept\n100 Users\n4 UAV-BS\nNSC','Percept\n200 Users\n4 UAV-BS\nNSC', 
                 'Percept\n100 Users\n4 UAV-BS\nSC','Percept\n200 Users\n4 UAV-BS\nSC',
                 'Percept\n100 Users\n8 UAV-BS\nNSC','Percept\n200 Users\n8 UAV-BS\nNSC', 
                 'Percept\n100 Users\n8 UAV-BS\nSC','Percept\n200 Users\n8 UAV-BS\nSC',
                 'Percept\n100 Users\n15 UAV-BS\nNSC','Percept\n200 Users\n15 UAV-BS\nNSC', 
                 'Percept\n100 Users\n15 UAV-BS\nSC','Percept\n200 Users\n15 UAV-BS\nSC']    

scen_names_BP_UOS = ['','UOS\n100 Users\n4 UAV-BS\nNo Small Cells','UOS\n200 Users\n4 UAV-BS\nNo Small Cells', 
                 'UOS\n100 Users\n4 UAV-BS\nSmall Cells','UOS\n200 Users\n4 UAV-BS\nSmall Cells',
                 'UOS\n100 Users\n8 UAV-BS\nNo Small Cells','UOS\n200 Users\n8 UAV-BS\nNo Small Cells', 
                 'UOS\n100 Users\n8 UAV-BS\nSmall Cells','UOS\n200 Users\n8 UAV-BS\nSmall Cells',
                 'UOS\n100 Users\n15 UAV-BS\nNo Small Cells','UOS\n200 Users\n15 UAV-BS\nNo Small Cells', 
                 'UOS\n100 Users\n15 UAV-BS\nSmall Cells','UOS\n200 Users\n15 UAV-BS\nSmall Cells']   

scen_names_BP_Percept = ['','Percept\n100 Users\n4 UAV-BS\nNo Small Cells','Percept\n200 Users\n4 UAV-BS\nNo Small Cells', 
                 'Percept\n100 Users\n4 UAV-BS\nSmall Cells','Percept\n200 Users\n4 UAV-BS\nSmall Cells',
                 'Percept\n100 Users\n8 UAV-BS\nNo Small Cells','Percept\n200 Users\n8 UAV-BS\nNo Small Cells', 
                 'Percept\n100 Users\n8 UAV-BS\nSmall Cells','Percept\n200 Users\n8 UAV-BS\nSmall Cells',
                 'Percept\n100 Users\n15 UAV-BS\nNo Small Cells','Percept\n200 Users\n15 UAV-BS\nNo Small Cells', 
                 'Percept\n100 Users\n15 UAV-BS\nSmall Cells','Percept\n200 Users\n15 UAV-BS\nSmall Cells'] 

qty_scen =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] #add qty of scenarios 
qty_scen2 =[1,2,3,4,5,6,7,8,9,10,11,12,13,14]

green_diamond = dict(markerfacecolor='g', marker='D')

# Create a figure instance
fig = plt.figure(1, figsize=(25, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data_to_plot[0:12], positions = qty_scen[1:13] ,patch_artist = True, flierprops=green_diamond, notch=True)
bp_percept = ax.boxplot(data_to_plot[12:], positions = qty_scen[13:], patch_artist = True, flierprops=green_diamond, notch=True)
for box in bp['boxes']:
    # change outline color
    box.set( color='#7c7f9e', linewidth=2)
    # change fill color
#    box.set( facecolor = '#1b9e77' )
    box.set( facecolor = '#cdd5e4' )
    
for box in bp_percept['boxes']:
    # change outline color
    box.set( color='#3867d6', linewidth=2)
    # change fill color
#    box.set( facecolor = '#1b9e77' )
    box.set( facecolor = '#45aaf2' )
            
#plt.title('Percentage users attended by UABS', fontsize=16)
plt.ylabel('Users Attended (%)', fontsize=14)
plt.xticks(np.arange(1,26),scen_names_BP,fontsize=10, ha= 'center') 
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.tight_layout(pad=3.0, w_pad=2, h_pad=1.0) 
#plt.grid(True)
plt.savefig("Boxplot_Users_perc_UABS_UOS_PERCEPT.pdf", format='pdf', dpi=1000)


# ---------------------------Create a figure instance for UOS Only-----------------------------------
fig_UOS = plt.figure(2, figsize=(14, 6))

# Create an axes instance
ax_uos = fig_UOS.add_subplot(111)

# Create the boxplot
bp_uos = ax_uos.boxplot(data_to_plot[0:12], positions = qty_scen2[1:13] ,patch_artist = True, flierprops=green_diamond, notch=True)

for box in bp_uos['boxes']:
    # change outline color
    box.set( color='#7c7f9e', linewidth=2)
    # change fill color
#    box.set( facecolor = '#1b9e77' )
    box.set( facecolor = '#cdd5e4' )
                
#plt.title('Percentage users attended by UABS', fontsize=16)
plt.ylabel('Users Attended (%)', fontsize=14)
plt.xticks(np.arange(1,14),scen_names_BP_UOS,fontsize=10, ha= 'center') 
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.tight_layout(pad=3.0, w_pad=2, h_pad=1.0) 
#plt.grid(True)
plt.savefig("Boxplot_Users_perc_UABS_UOS.pdf", format='pdf', dpi=1000)


# -----------------------Create a figure instance for Percept Only------------------------------------
fig_PERCEPT = plt.figure(3, figsize=(14, 6))

# Create an axes instance
ax_percept = fig_PERCEPT.add_subplot(111)

# Create the boxplot
bp_percept_only = ax_percept.boxplot(data_to_plot[12:], positions = qty_scen[13:], patch_artist = True, flierprops=green_diamond, notch=True)

for box in bp_percept_only['boxes']:
    # change outline color
    box.set( color='#3867d6', linewidth=2)
    # change fill color
#    box.set( facecolor = '#1b9e77' )
    box.set( facecolor = '#45aaf2' )
                
#plt.title('Percentage users attended by UABS', fontsize=16)
plt.ylabel('Users Attended (%)', fontsize=14)
plt.xticks(np.arange(13,26),scen_names_BP_Percept,fontsize=10, ha= 'center') 
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.tight_layout(pad=3.0, w_pad=2, h_pad=1.0) 
#plt.grid(True)
plt.savefig("Boxplot_Users_perc_UABS_PERCEPT.pdf", format='pdf', dpi=1000)


#//------------------Stack Plot ------------------\\

scen_names_SP = ['UOS 100 Users | 4 UAV-BS | No Small Cells','UOS 200 Users | 4 UAV-BS | No Small Cells', 
                 'UOS 100 Users | 4 UAV-BS | Small Cells','UOS 200 Users | 4 UAV-BS | Small Cells',
                 'UOS 100 Users | 8 UAV-BS | No Small Cells','UOS 200 Users | 8 UAV-BS | No Small Cells', 
                 'UOS 100 Users | 8 UAV-BS | Small Cells','UOS 200 Users | 8 UAV-BS | Small Cells',
                 'UOS 100 Users | 15 UAV-BS | No Small Cells','UOS 200 Users | 15 UAV-BS | No Small Cells', 
                 'UOS 100 Users | 15 UAV-BS | Small Cells','UOS 200 Users | 15 UAV-BS | Small Cells',
                 'Percept 100 Users | 4 UAV-BS | No Small Cells','Percept 200 Users | 4 UAV-BS | No Small Cells', 
                 'Percept 100 Users | 4 UAV-BS | Small Cells','Percept 200 Users | 4 UAV-BS | Small Cells',
                 'Percept 100 Users | 8 UAV-BS | No Small Cells','Percept 200 Users | 8 UAV-BS | No Small Cells', 
                 'Percept 100 Users | 8 UAV-BS | Small Cells','Percept 200 Users | 8 UAV-BS | Small Cells',
                 'Percept 100 Users | 15 UAV-BS | No Small Cells','Percept 200 Users | 15 UAV-BS | No Small Cells', 
                 'Percept 100 Users | 15 UAV-BS | Small Cells','Percept 200 Users | 15 UAV-BS | Small Cells']  

# stackplot_full = True
stackplot_full = False

if (stackplot_full == True):
    # Create a figure instance  
    fig2 = plt.figure(4, figsize=(9, 32)) #UOS
    
    # Create an axes instance
    #UOS
    ax2 = fig2.add_subplot(12,2,1)
    ax3 = fig2.add_subplot(12,2,2)
    ax4 = fig2.add_subplot(12,2,3)
    ax5 = fig2.add_subplot(12,2,4)
    ax6 = fig2.add_subplot(12,2,5)
    ax7 = fig2.add_subplot(12,2,6)
    ax8 = fig2.add_subplot(12,2,7)
    ax9 = fig2.add_subplot(12,2,8)
    ax10 = fig2.add_subplot(12,2,9)
    ax11 = fig2.add_subplot(12,2,10)
    ax12 = fig2.add_subplot(12,2,11)
    ax13 = fig2.add_subplot(12,2,12)
else:
    # Create a figure instance  
    fig2 = plt.figure(4, figsize=(9, 15)) #UOS
    
    # Create an axes instance
    #UOS
    ax2 = fig2.add_subplot(6,2,1)
    ax3 = fig2.add_subplot(6,2,2)
    ax4 = fig2.add_subplot(6,2,3)
    ax5 = fig2.add_subplot(6,2,4)
    ax6 = fig2.add_subplot(6,2,5)
    ax7 = fig2.add_subplot(6,2,6)
    ax8 = fig2.add_subplot(6,2,7)
    ax9 = fig2.add_subplot(6,2,8)
    ax10 = fig2.add_subplot(6,2,9)
    ax11 = fig2.add_subplot(6,2,10)
    ax12 = fig2.add_subplot(6,2,11)
    ax13 = fig2.add_subplot(6,2,12)

if (stackplot_full == True):
    #percept
    ax14 = fig2.add_subplot(12,2,13)
    ax15 = fig2.add_subplot(12,2,14)
    ax16 = fig2.add_subplot(12,2,15)
    ax17 = fig2.add_subplot(12,2,16)
    ax18 = fig2.add_subplot(12,2,17)
    ax19 = fig2.add_subplot(12,2,18)
    ax20 = fig2.add_subplot(12,2,19)
    ax21 = fig2.add_subplot(12,2,20)
    ax22 = fig2.add_subplot(12,2,21)
    ax23 = fig2.add_subplot(12,2,22)
    ax24 = fig2.add_subplot(12,2,23)
    ax25 = fig2.add_subplot(12,2,24)
else:
    fig3 = plt.figure(5, figsize=(9, 15)) #Percept
    #percept
    ax14 = fig3.add_subplot(6,2,1)
    ax15 = fig3.add_subplot(6,2,2)
    ax16 = fig3.add_subplot(6,2,3)
    ax17 = fig3.add_subplot(6,2,4)
    ax18 = fig3.add_subplot(6,2,5)
    ax19 = fig3.add_subplot(6,2,6)
    ax20 = fig3.add_subplot(6,2,7)
    ax21 = fig3.add_subplot(6,2,8)
    ax22 = fig3.add_subplot(6,2,9)
    ax23 = fig3.add_subplot(6,2,10)
    ax24 = fig3.add_subplot(6,2,11)
    ax25 = fig3.add_subplot(6,2,12)

# use a known color palette (see..)
pal = sns.color_palette("Set2")
uniquefuckingcolors  = ["#cdd5e4", "#7c7f9e"]
uniquefuckingcolors_Percept  = ["#45aaf2", "#3867d6"]

# Make the plot

#UOS - 4 UAV-BS - no small cells
#100 users
ax2.stackplot(range(0,32),data_to_plot2[0], data_to_plot[0], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors)
ax2.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax2.legend(loc='lower right')
ax2.set_title(scen_names_SP[0], fontsize=10)
#200 users
ax3.stackplot(range(0,32),data_to_plot2[1], data_to_plot[1], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors)
ax3.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax3.legend(loc='lower right')
ax3.set_title(scen_names_SP[1], fontsize=10)

#UOS - 4 UAV-BS - small cells
#100 users
ax4.stackplot(range(0,32),data_to_plot2[2], data_to_plot[2], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors)
ax4.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax4.legend(loc='lower right')
ax4.set_title(scen_names_SP[2], fontsize=10)
#200 users
ax5.stackplot(range(0,32),data_to_plot2[3], data_to_plot[3], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors)
ax5.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax5.legend(loc='lower right')
ax5.set_title(scen_names_SP[3], fontsize=10)

#UOS - 8 UAV-BS - no small cells
#100 users
ax6.stackplot(range(0,32),data_to_plot2[4], data_to_plot[4], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors)
ax6.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax6.legend(loc='lower right')
ax6.set_title(scen_names_SP[4], fontsize=10)
#200 users
ax7.stackplot(range(0,32),data_to_plot2[5], data_to_plot[5], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors)
ax7.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax7.legend(loc='lower right')
ax7.set_title(scen_names_SP[5], fontsize=10)

#UOS - 8 UAV-BS - small cells
#100 users
ax8.stackplot(range(0,32),data_to_plot2[6], data_to_plot[6], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors)
ax8.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax8.legend(loc='lower right')
ax8.set_title(scen_names_SP[6], fontsize=10)
#200 users
ax9.stackplot(range(0,32),data_to_plot2[7], data_to_plot[7], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors)
ax9.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax9.legend(loc='lower right')
ax9.set_title(scen_names_SP[7], fontsize=10)

#UOS - 15 UAV-BS - no small cells
#100 users
ax10.stackplot(range(0,32),data_to_plot2[8], data_to_plot[8], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors)
ax10.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax10.legend(loc='lower right')
ax10.set_title(scen_names_SP[8], fontsize=10)
#200 users
ax11.stackplot(range(0,32),data_to_plot2[9], data_to_plot[9], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors)
ax11.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax11.legend(loc='lower right')
ax11.set_title(scen_names_SP[9], fontsize=10)

#UOS - 15 UAV-BS - small cells
#100 users
ax12.stackplot(range(0,32),data_to_plot2[10], data_to_plot[10], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors)
ax12.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax12.legend(loc='lower right')
ax12.set_title(scen_names_SP[10], fontsize=10)
#200 users
ax13.stackplot(range(0,32),data_to_plot2[11], data_to_plot[11], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors)
ax13.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax13.legend(loc='lower right')
ax13.set_title(scen_names_SP[11], fontsize=10)

if (stackplot_full == False):
    fig2.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    fig2.tight_layout(pad=3.0, w_pad=2, h_pad=1.0)
    fig2.savefig("Stackplot_Users_perc_UABS_UOS.pdf", format='pdf', dpi=1000)


#Percept - 4 UAV-BS - no small cells
#100 users
ax14.stackplot(range(0,32),data_to_plot2[12], data_to_plot[12], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors_Percept)
ax14.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax14.legend(loc='lower right')
ax14.set_title(scen_names_SP[12], fontsize=10)
#200 users
ax15.stackplot(range(0,32),data_to_plot2[13], data_to_plot[13], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors_Percept)
ax15.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax15.legend(loc='lower right')
ax15.set_title(scen_names_SP[13], fontsize=10)

#Percept - 4 UAV-BS -  small cells
#100 users
ax16.stackplot(range(0,32),data_to_plot2[14], data_to_plot[14], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors_Percept)
ax16.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax16.legend(loc='lower right')
ax16.set_title(scen_names_SP[14], fontsize=10)
#200 users
ax17.stackplot(range(0,32),data_to_plot2[15], data_to_plot[15], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors_Percept)
ax17.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax17.legend(loc='lower right')
ax17.set_title(scen_names_SP[15], fontsize=10)


#Percept - 8 UAV-BS - no small cells
#100 users
ax18.stackplot(range(0,32),data_to_plot2[16], data_to_plot[16], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors_Percept)
ax18.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax18.legend(loc='lower right')
ax18.set_title(scen_names_SP[16], fontsize=10)
#200 users
ax19.stackplot(range(0,32),data_to_plot2[17], data_to_plot[17], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors_Percept)
ax19.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax19.legend(loc='lower right')
ax19.set_title(scen_names_SP[17], fontsize=10)

#Percept - 8 UAV-BS -  small cells
#100 users
ax20.stackplot(range(0,32),data_to_plot2[18], data_to_plot[18], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors_Percept)
ax20.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax20.legend(loc='lower right')
ax20.set_title(scen_names_SP[18], fontsize=10)
#200 users
ax21.stackplot(range(0,32),data_to_plot2[19], data_to_plot[19], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors_Percept)
ax21.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax21.legend(loc='lower right')
ax21.set_title(scen_names_SP[19], fontsize=10)


#Percept - 15 UAV-BS - no small cells
#100 users
ax22.stackplot(range(0,32),data_to_plot2[20], data_to_plot[20], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors_Percept)
ax22.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax22.legend(loc='lower right')
ax22.set_title(scen_names_SP[20], fontsize=10)
#200 users
ax23.stackplot(range(0,32),data_to_plot2[21], data_to_plot[21], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors_Percept)
ax23.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax23.legend(loc='lower right')
ax23.set_title(scen_names_SP[21], fontsize=10)

#Percept - 15 UAV-BS -  small cells
#100 users
ax24.set(ylim=(25, 100), xlim=(0, 33))
ax24.stackplot(range(0,32),data_to_plot2[22], data_to_plot[22], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors_Percept)
ax24.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax24.legend(loc='lower right')
ax24.set_title(scen_names_SP[22], fontsize=10)
#200 users
ax25.stackplot(range(0,32),data_to_plot2[23], data_to_plot[23], labels=['TBS','UAV-BS'],colors=uniquefuckingcolors_Percept)
ax25.set(xlabel='No. Simulations', ylabel='User Attended (%)')
ax25.legend(loc='lower right')
ax25.set_title(scen_names_SP[23], fontsize=10)


if (stackplot_full == True):

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    plt.tight_layout(pad=3.0, w_pad=2, h_pad=1.0)
    plt.savefig("Stackplot_Users_perc_UABS_PERCEPT_UOS.pdf", format='pdf', dpi=1000)
    #plt.show()
else:
    fig3.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    fig3.tight_layout(pad=3.0, w_pad=2, h_pad=1.0)
    fig3.savefig("Stackplot_Users_perc_UABS_PERCEPT.pdf", format='pdf', dpi=1000)
    #plt.show()

