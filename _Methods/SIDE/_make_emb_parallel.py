# -*- coding: utf-8 -*-
import os
import sys
# command_190731 = "python main_parallel.py --dataset 190803_n_SIDE_slAmi_withEdit_deg1 --network-file ../../_Data/slashdot_aminer_9_1.train " 
# command_0804 = "python main_parallel.py --dataset 190803_n_SIDE_epAmi_withEdit_deg1 --network-file ../../_Data/epinions_aminer_9_1.train " 
# command_0806 = "python main_parallel.py --dataset 190803_n_SIDE_sAmi_noEdit_deg1 --network-file ../../_Data/slashdot_aminer_9_1.train " 
# command_0806 += "--embed-dim 128 --deg1  --epochs-to-train 50" #" --window-size {} --epochs-to-train 50 --regularization-param {}"

# command_0807 = "python main.py --dataset 190807_n_SIDE_otc_noEdit_noDeg1 --network-file ../../_Data/bitcoin_otc_9_1.train " 
command_0807 = "python main.py --dataset 190808_n_SIDE_epiAmi_noEdit_withDeg1 --network-file ../../_Data/epinions_aminer_9_1.train "  
command_0807 += "--embed-dim 128 --epochs-to-train 20 --deg1" #" --window-size {} --epochs-to-train 50 --regularization-param {}"

window_size = [5]  # default 5
regul = [0.01]

for ws in window_size:
    for r in regul:
        print("ws {}, r {}".format(ws, r))
        os.system(command_0807.format(ws, r, ws, r))



