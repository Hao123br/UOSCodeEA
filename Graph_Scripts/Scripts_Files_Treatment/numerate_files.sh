#!/bin/bash

step=17


for i in {0..23}
do
	apd="APD.plt.~$i~"	
	jitter="Jitter.plt.~$i~"	
	tp="Throughput.plt.~$i~"	
	pdr="PDR.plt.~$i~"	

	sed -i "s/\(APD\).*\.png/\1_$i.png/g" $apd
	sed -i "s/\(Jitter\).*\.png/\1_$i.png/g" $jitter
	sed -i "s/\(Throughput\).*\.png/\1_$i.png/g" $tp
	sed -i "s/\(PDR\).*\.png/\1_$i.png/g" $pdr

done
