#!/bin/bash


for i in {0..59}
do
mv UE_info_UABS.~$i~ UE_info_UABS_RUN#$i

mv Quantity_UABS.~$i~ Quantity_UABS_RUN#$i

mv Qty_UE_SINR.~$i~ Qty_UE_SINR_RUN#$i


done
