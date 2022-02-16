from tqdm import tqdm
import random
import os
import numpy as np
import utility

class VNE_Generator(object):
	def __init__(self, input_file, folder_path, test_file):
		self.input_file = input_file
		self.folder_path = folder_path
		self.test_file = test_file
		self.has_large_data = False

		self.edges = utility.read_edges_from_file(self.input_file)
		self.num_edges = len(self.edges)
		self.graph = utility.read_graph_from_file(self.input_file)
		self.degrees = utility.get_node_degrees(self.edges)
		self.in_degrees, self.out_degrees = utility.get_in_out_degrees(self.edges)
		self.num_nodes = len(self.degrees)

		print("==============Reading Graph============")
		print("num_nodes: {}".format(self.num_nodes))
		print("num_edges: {}".format(self.num_edges))
		print("=======================================")

	def add_vnes(self, vne_algo='wrmf', option='local', theta=0.5, emb_dim=16):
		random.seed(0)
		emb_path = self.input_file + '_{}.emb'.format(vne_algo) 
		output_path = self.input_file + "_signed_{}_{}_{}".format(vne_algo, option, theta)

		potential_vnes_filepath = self.folder_path + 'potential_VNEs_{}'.format(vne_algo)
		if 'local' in option:
			potential_vnes_filepath += '_local'
		elif 'global' in option:
			potential_vnes_filepath += '_global'

		# Consider the lower the degree of positivity is, the higher the degree of negativity is
		if os.path.isfile(potential_vnes_filepath):
			if 'local' in option:
				vnes = self.read_scores_from_file_in_dictionary(potential_vnes_filepath)
			elif 'global' in option:
				vnes = self.read_scores_from_file(potential_vnes_filepath)

		else:
			# If the embedding file (for calculating degree of positivity/negativity) does not exist, perform algorithm
			if not os.path.isfile(emb_path):
				os.chdir(os.getcwd() + '/_Methods/NeuRec-master')
				os.system("python main.py --recommender=WRMF --data.input.path=" + self.input_file + ' --data.test.input.path=' + self.test_file +  \
					" --splitter=given --epochs=50 --embedding_size=" + str(emb_dim) + " --data.output.path=" + emb_path + " --weight=uniform --data.index.remap=False --vne_option=general")
				os.chdir("../..")
				#_execute_methods.perform_algorithm(os.getcwd() + "/_Methods/", "", vne_algo, False, self.input_file, "", emb_path, self.emb_dim, self.num_nodes, "", "", "")
			
			src = np.zeros((self.num_nodes, emb_dim))
			tar = np.zeros((self.num_nodes, emb_dim))
			src, tar = utility.read_embeddings(emb_path, vne_algo, self.num_nodes, emb_dim, src_only=False)

			if 'local' in option:
				vnes = self.write_local_scores_to_file(potential_vnes_filepath, src, tar)
			elif "global" in option:
				vnes = self.write_global_scores_to_file(potential_vnes_filepath, src, tar)

		if 'local' in option:
			vnes = self.local_selection(vnes, float(theta), self.num_edges, self.num_nodes, self.out_degrees)
		elif 'global' in option:
			vnes = self.global_selection(vnes, float(theta), self.num_edges)

		self.write_vnes_to_file(output_path, vnes)

	def write_global_scores_to_file(self, file_path, src, tar):
		capacity = 40
		potential_vnes = []
		max_vnes = self.num_edges * capacity

		pbar = tqdm(total=self.num_nodes)
		for s in range(self.num_nodes):
			pbar.set_description("%d-iter" % (s+1))
			pbar.update(1)	
			neighbors = set(self.graph[s])
			neighbors.add(s)
			
			# [src, tar, degree of positivity]
			potential_vnes_per_src = [[s, t, np.dot(src[s], tar[t])] for t in range(self.num_nodes) if t not in neighbors]
			potential_vnes.extend(potential_vnes_per_src)

			if self.has_large_data == True and s == int(self.num_nodes/2):
				print("##### Very Large Dataset #####")
				potential_vnes.sort(key=lambda x: x[2])
				potential_vnes = potential_vnes[:max_vnes]

		potential_vnes.sort(key=lambda x: x[2])

		potential_vnes = potential_vnes[:max_vnes]
		self.write_scores_to_file(file_path, potential_vnes)
		pbar.close()
		return potential_vnes

	def write_local_scores_to_file(self, file_path, src, tar):
		capacity = 40
		potential_vnes = {}
		vne_degree = round(self.num_edges / self.num_nodes)
		max_vnes = vne_degree * capacity

		pbar = tqdm(total=self.num_nodes)
		for s in range(self.num_nodes):
			pbar.set_description("%d-iter" % (s+1))
			pbar.update(1)	
			neighbors = set(self.graph[s])
			neighbors.add(s)

			potential_vnes_per_src = [[t, np.dot(src[s], tar[t])] for t in range(self.num_nodes) if t not in neighbors]
			potential_vnes_per_src.sort(key=lambda x: x[1])
			
			# Shuffle nodes with same score (degree of positivity)
			potential_vnes_per_src = self.shuffle_nodes_with_same_scores(potential_vnes_per_src, max_vnes)
			potential_vnes_per_src = potential_vnes_per_src[:max_vnes]

			potential_vnes[s] = potential_vnes_per_src

		self.write_scores_dict_to_file(file_path, potential_vnes)
		pbar.close()
		return potential_vnes

	# shuffle nodes with same score
	def shuffle_nodes_with_same_scores(self, personalized_rel, coverage):
		new_personalized_rel = []
		rel_tmp = []
		flag = 0
		for rel in personalized_rel:
			if flag == 0:
				prev_score = rel[1]
				flag = 1
			cur_score = rel[1]
			if cur_score == prev_score:
				rel_tmp.append(rel)
			else:
				random.shuffle(rel_tmp)
				new_personalized_rel.extend(rel_tmp)
				rel_tmp = []
				rel_tmp.append(rel)
				prev_score = cur_score
				if len(new_personalized_rel) > coverage:
					break
		if rel_tmp:
			random.shuffle(rel_tmp)
			new_personalized_rel.extend(rel_tmp)
		return new_personalized_rel

	def global_selection(self, potential_vnes, ratio, num_edges):
	    # potential_vnes => [source, target, score (dop)]..]
	 
	    num_vnes = round(ratio * num_edges)
	    vnes = []
	    #potential_vnes.sort(key=lambda x: x[2])
	    cnt = 0
	    for edge in potential_vnes:
	        vnes.append(edge)
	        cnt += 1
	        if cnt >= num_vnes:
	            break

	    print("number of ori edges: {}, number of VNEs: {}".format(num_edges, len(relevance_)))

	    return vnes

	def local_selection(self, potential_vnes, ratio, num_edges, num_nodes, degrees):
	    # potential_vnes => {source: [target, score (dop)]..]
	    # vne_degrees = {source:[number of vnes per source, current #] ,..}
	 
	    num_vnes = round(ratio * num_edges)
	    min_degree = 0
	    max_degree = 0
	    m_d = 0
	    for d in degrees.values():
	        if d > m_d:
	            m_d = d
	    if max_degree == 0:
	        max_degree = m_d

	    vnes = []
	    #potential_vnes.sort(key=lambda x: x[2])
	    cnt = 0
	    vne_degrees = {}
	    vne_degree_per_src = round(ratio * num_edges / num_nodes)

	    for source, degree in degrees.items():
	        if degree >= min_degree and degree <= max_degree:
	            vne_degrees[source] = [vne_degree_per_src, 0]

	    for source, targets in potential_vnes.items():
	        for target in targets:
	            if vne_degrees.get(source) and vne_degrees.get(target[0]):
	                # Until each source get a pre-defined number of vnes
	                if vne_degrees[source][0] > vne_degrees[source][1]:
	                    vnes.append([source, target[0], target[1]])
	                    vne_degrees[source][1] += 1

	    print("number of ori edges: {}, DESIRED number of VNEs: {}, REAL number of VNEs: {}".format(num_edges, num_vnes, len(vnes)))

	    return vnes

	def write_scores_to_file(self, filename, edges):
	    with open(filename, "w+") as f:
	        for edge in edges:
	            f.writelines(str(edge[0])+"\t"+str(edge[1])+"\t"+str(edge[2])+"\n")

	def write_scores_dict_to_file(self, filename, edges):
	    with open(filename, "w+") as f:
	        for seed, targets in edges.items():
	            for target in targets:
                	f.writelines(str(seed)+"\t"+str(target[0])+"\t"+str(target[1])+"\n")

	def read_scores_from_file(self, filename):
	    with open(filename, "r") as f:
	        lines = f.readlines()
	        edges = [self.str_list_to_scores(line.split()) for line in lines]
	    return edges

	def read_scores_from_file_in_dictionary(self, filename):
	    edges = self.read_scores_from_file(filename)
	    d = {}
	    for edge in edges:
	        if d.get(edge[0]) is None:
	            d[edge[0]] = []
	        d[edge[0]].append([edge[1], edge[2]])
	    return d

	def str_list_to_scores(self, str_list):
		return [int(str_list[0]), int(str_list[1]), float(str_list[2])]

	def write_vnes_to_file(self, output_path, vnes, write_vnes=True):
		with open(output_path, "w+") as f:
			for edge in self.edges:
				f.writelines(str(edge[0])+"\t"+str(edge[1])+"\t1\n")
			for edge in vnes:
				f.writelines(str(edge[0])+"\t"+str(edge[1])+"\t-1\n")

		if write_vnes == True:
			with open(output_path+".dop", "w+") as f:
				for edge in vnes:
					f.writelines(str(edge[0])+"\t"+str(edge[1])+"\t"+str(edge[2])+"\n")