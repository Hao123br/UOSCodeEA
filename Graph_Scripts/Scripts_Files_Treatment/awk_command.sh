#!/bin/bash    

QtyUsers=100
QtyUABS=15

for i in {0..1}
do

awk -f add.awk plot_PDr > plot_PDR_"$QtyUsers"_4_$QtyUABS
awk -f add.awk plot_APD > plot_APD_"$QtyUsers"_4_$QtyUABS
awk -f add.awk plot_thr > plot_TP_"$QtyUsers"_4_$QtyUABS
awk -f add.awk plot_jitter > plot_Jitter_"$QtyUsers"_4_$QtyUABS


done
