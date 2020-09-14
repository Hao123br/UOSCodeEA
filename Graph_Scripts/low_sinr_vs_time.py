import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import glob

usepath = False
path = '/run/user/1000/gvfs/sftp:host=gercom.ddns.com,port=8372'
# scenarios_path = ['/home/emanuel/v2_master_thesis_UOS/100U_2_enB_No_UABS/UOSCode/results_thesis/',
#                 '/home/emanuel/v2_master_thesis_UOS/100U_2_enB_6_UABS/UOSCode/results_thesis/',
#                 '/home/emanuel/IEEE_Article/final_graphs/2enB_6UABS_100U/ueinfo/',  #PERCEPT SCENARIO
#                 '/home/emanuel/v2_master_thesis_UOS/200U_2_enB_No_UABS/UOSCode/results_thesis/',
#                 '/home/emanuel/v2_master_thesis_UOS/200U_2_enB_6_UABS/UOSCode/results_thesis/',
#                 '/home/emanuel/IEEE_Article/final_graphs/2enB_6UABS_200U/ueinfo/', #PERCEPT SCENARIO
#                 #4 eNBs scenarios
#                 '/home/emanuel/v2_master_thesis_UOS/100U_LTE_No_UABS/UOSCode/results_thesis/',
#                 '/home/emanuel/v2_master_thesis_UOS/100U_LTE_6_UABS/UOSCode/results_thesis/',
#                 '/home/emanuel/IEEE_Article/final_graphs/4enB_6UABS_100U/ueinfo/', #PERCEPT SCENARIO
#                 '/home/emanuel/v2_master_thesis_UOS/200U_LTE_No_UABS/UOSCode/results_thesis/',
#                 '/home/emanuel/v2_master_thesis_UOS/200U_LTE_6_UABS/UOSCode/results_thesis/',
#                 '/home/emanuel/IEEE_Article/final_graphs/4enB_6UABS_200U/ueinfo/'] #PERCEPT SCENARIO

scenarios_path = [                  
                  #no_smallcell
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/LTE/4enB_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_4UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_4UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_8UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_15UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_15UABS_100U/',
                  
                  #no_smallcell
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/LTE/4enB_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_4UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_4UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_8UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/LTE_and_UOS/UOS/4enB_15UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/No_Small_Cells/Percept/4enB_15UABS_200U/',
                  
                  #smallcell
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/LTE/4enB_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_4UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_4UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_8UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_15UABS_100U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_15UABS_100U/',
                  
                  #smallcells  
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/LTE/4enB_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_4UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_4UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_8UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_8UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/LTE_and_UOS/UOS/4enB_15UABS_200U/',
                  '/home/emanuel/Desktop/IEEE_Article/Sera_el_final_o_no/Small_Cells/Percept/4enB_15UABS_200U/']
                 
                 
No_scenarios = len(scenarios_path)

data_dict = dict()
data_dict['x'] = [x*5 for x in range(1,20)]
#interate through each scenario path
for i in range(0,No_scenarios):
    scenario = scenarios_path[i]
    qty_by_time = dict()
    mean_by_time = []
    prefix = ''
    if(usepath):
        prefix = path
    #get the path of all Qty files in this scenario
    files = glob.glob(prefix + scenario + 'Qty_UE_SINR*')
    print('y{}: {}'.format(i+1,len(files)))
    #iterate through each Qty file path
    for data_path in files:
        data = open(data_path, 'r')
        for line in data:
            sline = line.split(',')
            time = int(sline[0])
            qty = float(sline[1])
            if time in qty_by_time:
                qty_by_time[time] = qty_by_time[time] + qty
            else:
                qty_by_time[time] = qty
    for qty_sum in qty_by_time.values():
        mean_by_time.append(qty_sum/len(files))
    data_dict['y{0}'.format(i+1)] = mean_by_time

#ensure all lists have the same size
for mean_list in data_dict.values():
    mean_list.extend([0 for zero in range(19-len(mean_list))])

df = pd.DataFrame(data_dict)
print(df)
ax1=plt.subplot(4,2,1)
ax2=plt.subplot(4,2,2)
ax3=plt.subplot(4,2,3)
ax4=plt.subplot(4,2,4)

#Color palette
# Palette_Colors  = ["#cdd5e4", "#7c7f9e", "#3c4563","#a55eea","#8854d0","#26de81","#20bf6b"]
Palette_Colors  = ["#eb3b5a", "#7c7f9e", "#3c4563","#a55eea","#8854d0","#26de81","#20bf6b"] #with LTE red

legends = []
# multiple line plot
ax1.set(ylim=(0,20), xlim=(5, 95))

ax1.plot( 'x', 'y1', '', data=df, marker='', color= Palette_Colors[0], linewidth=2,linestyle='dashed', label='LTE') #100U-2eNBs-noUABS
ax1.plot( 'x', 'y2', '', data=df, marker='', color= Palette_Colors[1], linewidth=2,label='LTE + UOS 4 UAV-BS') #100U-2eNBs-6UABS
ax1.plot( 'x', 'y3', '', data=df, marker='', color= Palette_Colors[2], linewidth=2,label='LTE + Percept 4 UAV-BS') #PERCEPT SCENARIO
ax1.plot( 'x', 'y4', '', data=df, marker='', color= Palette_Colors[3], linewidth=2,label='LTE + UOS 8 UAV-BS') #100U-2eNBs-6UABS
ax1.plot( 'x', 'y5', '', data=df, marker='', color= Palette_Colors[4], linewidth=2,label='LTE + Percept 8 UAV-BS') #PERCEPT SCENARIO
ax1.plot( 'x', 'y6', '', data=df, marker='', color= Palette_Colors[5], linewidth=2,label='LTE + UOS 15 UAV-BS') #100U-2eNBs-6UABS
ax1.plot( 'x', 'y7', '', data=df, marker='', color= Palette_Colors[6], linewidth=2,label='LTE + Percept 15 UAV-BS') #PERCEPT SCENARIO
ax1.set(xlabel='Simulation time (s)', ylabel='Number of UEs')
ax1.xaxis.get_label().set_fontsize(14)
ax1.yaxis.get_label().set_fontsize(14)
# legends.append(ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2))
legends.append(ax1.legend(loc='upper center',bbox_to_anchor=(0.5, 0.97), ncol=2))
ax1.set_title('100 Users | No Small Cells',fontsize=14)

ax2.set(ylim=(0, 20), xlim=(5, 95))
ax2.plot( 'x', 'y8', '', data=df, marker='', color=Palette_Colors[0], linewidth=2,linestyle='dashed', label='LTE') #200U-2eNBs-noUABS
ax2.plot( 'x', 'y9', '', data=df, marker='', color= Palette_Colors[1], linewidth=2,label='LTE + UOS 4 UAV-BS') #200U-2eNBs-6UABS
ax2.plot( 'x', 'y10', '', data=df, marker='', color= Palette_Colors[2], linewidth=2,label='LTE + Percept 4 UAB-BS') #PERCEPT SCENARIO
ax2.plot( 'x', 'y11', '', data=df, marker='', color= Palette_Colors[3], linewidth=2,label='LTE + UOS 8 UAV-BS') #200U-2eNBs-6UABS
ax2.plot( 'x', 'y12', '', data=df, marker='', color= Palette_Colors[4], linewidth=2,label='LTE + Percept 8 UAB-BS') #PERCEPT SCENARIO
ax2.plot( 'x', 'y13', '', data=df, marker='', color= Palette_Colors[5], linewidth=2,label='LTE + UOS 15 UAV-BS') #200U-2eNBs-6UABS
ax2.plot( 'x', 'y14', '', data=df, marker='', color= Palette_Colors[6], linewidth=2,label='LTE + Percept 15 UAB-BS') #PERCEPT SCENARIO
ax2.set(xlabel='Simulation time (s)', ylabel='Number of UEs')
ax2.xaxis.get_label().set_fontsize(14)
ax2.yaxis.get_label().set_fontsize(14)
# legends.append(ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2))
legends.append(ax2.legend(loc='upper center',bbox_to_anchor=(0.5, 0.97), ncol=2))
ax2.set_title('200 Users | No Small Cells',fontsize=14)

ax3.set(ylim=(0, 20),xlim=(5, 95))
ax3.plot( 'x', 'y15', '', data=df, marker='', color=Palette_Colors[0], linewidth=2,linestyle='dashed', label='LTE') #100U-4eNBs-noUABS
ax3.plot( 'x', 'y16', '', data=df, marker='', color= Palette_Colors[1], linewidth=2,label='LTE + UOS 4 UAV-BS') #100U-4eNBs-6UABS
ax3.plot( 'x', 'y17', '', data=df, marker='', color= Palette_Colors[2], linewidth=2,label='LTE + Percept 4 UAV-BS') #PERCEPT SCENARIO
ax3.plot( 'x', 'y18', '', data=df, marker='', color= Palette_Colors[3], linewidth=2,label='LTE + UOS 8 UAV-BS') #100U-4eNBs-6UABS
ax3.plot( 'x', 'y19', '', data=df, marker='', color= Palette_Colors[4], linewidth=2,label='LTE + Percept 8 UAV-BS') #PERCEPT SCENARIO
ax3.plot( 'x', 'y20', '', data=df, marker='', color= Palette_Colors[5], linewidth=2,label='LTE + UOS 15 UAV-BS') #100U-4eNBs-6UABS
ax3.plot( 'x', 'y21', '', data=df, marker='', color= Palette_Colors[6], linewidth=2,label='LTE + Percept 15 UAV-BS') #PERCEPT SCENARIO
ax3.set(xlabel='Simulation time (s)', ylabel='Number of UEs')
ax3.xaxis.get_label().set_fontsize(14)
ax3.yaxis.get_label().set_fontsize(14)
# legends.append(ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2))
legends.append(ax3.legend(loc='upper center',bbox_to_anchor=(0.5, 0.97), ncol=2))
ax3.set_title('100 Users | Small Cells',fontsize=14)

ax4.set(ylim=(0, 20), xlim=(5, 95))
ax4.plot( 'x', 'y22', '', data=df, marker='', color=Palette_Colors[0], linewidth=2,linestyle='dashed', label='LTE') #200U-4eNBs-noUABS
ax4.plot( 'x', 'y23', '', data=df, marker='', color= Palette_Colors[1], linewidth=2,label='LTE + UOS 4 UAV-BS') #200U-4eNBs-6UABS
ax4.plot( 'x', 'y24', '', data=df, marker='', color= Palette_Colors[2], linewidth=2,label='LTE + Percept 4 UAV-BS') #PERCEPT SCENARIO
ax4.plot( 'x', 'y25', '', data=df, marker='', color= Palette_Colors[3], linewidth=2,label='LTE + UOS 8 UAV-BS') #200U-4eNBs-6UABS
ax4.plot( 'x', 'y26', '', data=df, marker='', color= Palette_Colors[4], linewidth=2,label='LTE + Percept 8 UAV-BS') #PERCEPT SCENARIO
ax4.plot( 'x', 'y27', '', data=df, marker='', color= Palette_Colors[5], linewidth=2,label='LTE + UOS 15 UAV-BS') #200U-4eNBs-6UABS
ax4.plot( 'x', 'y28', '', data=df, marker='', color= Palette_Colors[6], linewidth=2,label='LTE + Percept 15 UAV-BS') #PERCEPT SCENARIO
ax4.set(xlabel='Simulation time (s)', ylabel='Number of UEs')
ax4.xaxis.get_label().set_fontsize(14)
ax4.yaxis.get_label().set_fontsize(14)
# legends.append(ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2))
legends.append(ax4.legend(loc='upper center',bbox_to_anchor=(0.5, 0.97), ncol=2))
ax4.set_title('200 Users | Small Cells',fontsize=14)

#plt.ylabel('Number of UEs')
#plt.xlabel('Simulation time (s)')
#plt.suptitle('Average Low SINR Users')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
plt.subplots_adjust(right=1.95, top=4.0, hspace=0.3)
# plt.subplots_adjust(right=1.5, top=2.25, hspace=0.3)
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
# plt.tight_layout(pad=3.0, w_pad=2, h_pad=1.0)
plt.savefig('lineplot_Users_low_SINR_PERCEPT.png', dpi=1000, bbox_inches='tight', bbox_extra_artists=legends)
