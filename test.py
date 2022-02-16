# -*- coding: utf-8 -*-
import os
import sys

gen_dataset = ["GNU","WIKI",'JUNG','EAT']
gen_emb_algo = ['side','stne']
gen_test_types = ['LP-uniform','LP-mixed','LP-biased']

# for virtual negative edges
#inp_algos = ["wrmf"] # "wrmf", "rwr", "app", "atp", "nerd"
#inp_ratios = [0.5]  # "0.6", "1.0", 2.0"
#extract_options = ["d16_shuffle"]

version = ["uniform_send"]


#py27: "adamic-adar", "node2vec", "sine_unsign", "sine", "signet", "deepwalk"
#py35: "graphgan", "slf_unsign", "slf", "beside_unsign", "beside", "sne_unsign", "sne"
#py37: "stne_unsign", "stne", "app"
#linux -> py27: "sgcn", "sgcn_unsign", "atp", "line"; py35: "side_unsign", "side", "dwns"


#fold_ids = ['u1',"u2","u3"]
num_embed = 128 # 128, 63, 129
emb_mani = ["ss", "st", "tt", "stconcat"]
emb_mani = ['stconcat']


dataset_idx = [2]
algo_idx = [2] # py27: 1,2 py37: 3,(4,5),6,7,8 linux: 9,10,11; 12,13
test_idx = [1,2,3]


directed_algorithms = [ "sne", "sne_unsign", "side", "side_unsign", "sine", "app", "atp", "nerd", "nerd_naive", "wrmf","hope"]


#################
dataset = []
for i in dataset_idx:
	dataset.append(gen_dataset[i-1])
emb_algo = []
for i in algo_idx:
	emb_algo.append(gen_emb_algo[i-1])
test_types = []
for i in test_idx:
	test_types.append(gen_test_types[i-1])
#################


for data in dataset:
	for algo in emb_algo:
		for test_type in test_types:
			arg = "python divine.py --dataset " + data + " --emb_algo " + algo + " --lp_task " + test_type + " --num_embed " + str(num_embed) +\
			' --vne_algo wrmf --theta 0.5 --selection_strategy local'
			os.system(arg)


