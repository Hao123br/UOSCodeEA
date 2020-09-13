#!/bin/bash

step=17


for i in {0..9}
do
	apd="APD_run_$i.plt"	
	jitter="Jitter_run_$i.plt"	
	tp="Throughput_run_$i.plt"	
	pdr="PDR_run_$i.plt"	

#	sed -i "s/\(run_\).*\.png/\1$i.png/g" $apd
#	sed -i "s/\(run_\).*\.png/\1$i.png/g" $jitter
#	sed -i "s/\(run_\).*\.png/\1$i.png/g" $tp
	sed -i "s/\(run_\).*\.png/\1$i.png/g" $pdr

done
