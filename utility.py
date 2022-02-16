import os
import numpy as np

def read_edges_from_file(filename): # all columns
    with open(filename, "r") as f: 
        lines = f.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]
    return edges

def str_list_to_int(str_list):
    return [int(item) for item in str_list]

def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def read_graph_from_file(file_path):
    edges = read_edges_from_file(file_path)
    d = {}
    for edge in edges:
        if d.get(edge[0]) is None:
            d[edge[0]] = []
        if d.get(edge[1]) is None:
            d[edge[1]] = []
        d[edge[0]].append(edge[1])
    return d

def get_node_degrees(edges):
    d = {}
    for edge in edges:
        if d.get(edge[0]) is None:
            d[edge[0]] = 0
        if d.get(edge[1]) is None:
            d[edge[1]] = 0
        d[edge[0]] += 1
        d[edge[1]] += 1
    return d

def get_in_out_degrees(edges):
    d_out = {}
    d_in = {}
    for edge in edges:
        if d_out.get(edge[0]) is None:
            d_out[edge[0]] = 0
        if d_out.get(edge[1]) is None:
            d_out[edge[1]] = 0
        if d_in.get(edge[0]) is None:
            d_in[edge[0]] = 0
        if d_in.get(edge[1]) is None:
            d_in[edge[1]] = 0

        d_out[edge[0]] += 1
        d_in[edge[1]] += 1
    return d_in, d_out

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory - " + directory)

def read_embeddings(filename, emb_algo, num_node, num_embed, src_only=True):

    embedding_matrix = []
    embedding_matrix.append(np.zeros((num_node, num_embed)))
    #embedding_matrix = np.random.rand(n_node, n_embed)

    if 'side' in emb_algo or 'wrmf' in emb_algo:
        # Read source embeddings
        with open(filename, "r") as f:
            lines = f.readlines()[1:]  # Skip the first line
            for line in lines:
                emd = line.split()
                embedding_matrix[0][int(emd[0]), :] = str_list_to_float(emd[1:])
        if src_only == False:
            # Read target embeddings
            embedding_matrix.append(np.zeros((num_node, num_embed)))
            with open(filename+'2', "r") as f:
                lines = f.readlines()[1:]  # Skip the first line
                for line in lines:
                    emd = line.split()
                    embedding_matrix[1][int(emd[0]), :] = str_list_to_float(emd[1:])

    elif 'stne' in emb_algo:
        # Read embeddings
        with open(filename, "r") as f:            
            lines = f.readlines()
            cnt = 0
            for line in lines:
                emd = line.split()
                embedding_matrix[0][cnt, :] = str_list_to_float(emd)
                cnt +=1
    return embedding_matrix


