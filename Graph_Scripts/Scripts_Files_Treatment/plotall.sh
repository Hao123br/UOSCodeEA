#!/bin/bash
	for i in {0..23..1}
do
  cat "APD.plt.~$i~" >> plot_APD
  cat "Throughput.plt.~$i~" >> plot_thr
  cat "PDR.plt.~$i~" >> plot_PDr
  cat "Jitter.plt.~$i~" >> plot_jitter
done
