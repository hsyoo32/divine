import os
import argparse
import link_prediction as lp
import utility as util
import numpy as np
from vne_generator import VNE_Generator

def parse_args():
    parser = argparse.ArgumentParser(description="Run Test.")

    parser.add_argument('--dataset', nargs='?', default='GNU',
                        help='Input dataset')   

    parser.add_argument('--emb_algo', nargs='?', default='stne',
                        help='Signed network embedding algorithm')

    parser.add_argument('--lp_task', nargs='?', default='LP-uniform',
                        help='Target task')

    parser.add_argument('--num_embed', type=int, default=128,
                        help='Dimensionality of embeddings')


    # For adding virtual negative edges
    parser.add_argument('--vne_algo', nargs='?', default='wrmf',
                        help='Algorithm for inferring degree of negativity')

    parser.add_argument('--theta', nargs='?', default='0.5',
                        help='Hyperparameter determining the number of VNEs')

    parser.add_argument('--selection_strategy', nargs='?', default='local',
                        help='Local or global')

    return parser.parse_args()

def test(args):
    # parameter settings
    data = args.dataset
    emb_algo = args.emb_algo
    lp_task = args.lp_task    
    num_embed = args.num_embed

    # parameters related to virtual negative edges
    vne_algo = args.vne_algo
    theta = args.theta
    selection_strategy = args.selection_strategy
    
    # path settings and folder creation
    file_path = os.getcwd() + "/_Data/" + data + "/"
    embed_path = os.getcwd() + "/_Emb/" + data + "/"
    result_path = os.getcwd() + "/_Results/" + data + "/" + lp_task + "/" #emb_algo + "/"
    algo_path = os.getcwd() + "/_Methods/"
    util.create_folder(file_path)
    util.create_folder(embed_path)
    util.create_folder(result_path)
    util.create_folder(algo_path)

    ori_filename = file_path + "{}_train".format(data)
    test_filename = file_path + "{}_test".format(data)
    train_filename = file_path + "{}_train_signed_{}_{}_{}".format(data, vne_algo, selection_strategy, theta)
    embed_filename = embed_path + "{}_{}_dim{}_{}_{}_{}.emb".format(data, emb_algo, num_embed, vne_algo, selection_strategy, theta)
    result_filename = result_path + "{}_{}_dim{}_{}_{}_{}.result".format(data, emb_algo, num_embed, vne_algo, selection_strategy, theta)

    # Add VNEs to the input network
    if not os.path.isfile(train_filename):
        vne_generator = VNE_Generator(ori_filename, file_path, test_filename)
        vne_generator.add_vnes(vne_algo, selection_strategy, float(theta))

    # Get the number of nodes
    train_edges_ = util.read_edges_from_file(ori_filename)
    nodes = set()
    for edge in train_edges_:
        nodes = nodes.union(set(edge[:2]))
    num_node = len(nodes)

    # Perform a signed NE method
    if not os.path.isfile(embed_filename): 
        perform_algorithm(algo_path, embed_path, emb_algo, train_filename, embed_filename, num_embed)

    # Read embeddings
    train_embeddings = []
    train_embeddings.append(np.zeros((num_node, num_embed)))
    if emb_algo == 'side':
        # SIDE outputs d-dimensional source and target embeddings for each node, respectively
        # Empirically, using only source embeddings lead to high link prediction accuracies
        src_only = True
        if src_only == False:
            train_embeddings.append(np.zeros((num_node, num_embed))) # for target embeddings
        train_embeddings = util.read_embeddings(embed_filename, emb_algo, num_node, num_embed, src_only=src_only)
    elif emb_algo == 'stne':
        # STNE outputs an unified d-dimensional embedding for each node
        train_embeddings = util.read_embeddings(embed_filename, emb_algo, num_node, num_embed)

    # Evaluation
    if 'LP-' in lp_task:
        # Train edges / test edges -> original unsigned network
        train_edges = util.read_edges_from_file(ori_filename)     
        test_edges = util.read_edges_from_file(test_filename)       
        
        if "LP-uniform" == lp_task:
            train_uncon_filename = file_path + '{}_train_uncon_ulp'.format(data)
            test_uncon_filename = file_path + '{}_test_uncon_ulp'.format(data)
        elif "LP-mixed" == lp_task:
            train_uncon_filename = file_path + '{}_train_uncon_mlp'.format(data)
            test_uncon_filename = file_path + '{}_test_uncon_mlp'.format(data)
        elif "LP-biased" == lp_task:
            train_uncon_filename = file_path + '{}_train_uncon_blp'.format(data)
            test_uncon_filename = file_path + '{}_test_uncon_blp'.format(data)
        train_edges_uncon = util.read_edges_from_file(train_uncon_filename)
        test_edges_uncon = util.read_edges_from_file(test_uncon_filename)
        lp.link_prediction(train_edges, test_edges, train_edges_uncon, test_edges_uncon, train_embeddings, result_filename, num_embed)

    print("Finish: perform evaluation - " + lp_task)

def perform_algorithm(algo_path, embed_path, algo, train_filename, embed_filename, num_embed):
    
    if "side" in algo: # linux only, py35, tensorflow1.1
        os.chdir(algo_path + "SIDE")
        #dataset = data + "_" + 
        dataset = os.path.basename(train_filename)
        print(dataset)
        
        epoch = 1
        extra_arg = " --signed --deg1"

        os.system("python main.py --network-file " + train_filename + " --dataset " + dataset + " --embed-dim " + str(num_embed) + \
            " --learning-rate 0.025 --num-walks 80 --walk-length 40 --window-size 5 --neg-sample-size 20 --epochs-to-train " + str(epoch) + extra_arg)

        # Revise embedding
        with open("./output/" + dataset + ".vocab", "r") as f1, open("./output/" + dataset +"_ep" + str(epoch-1) + ".emb", "r") as f2, \
         open("./output/" + dataset +"_ep" + str(epoch-1) + ".emb2", "r") as f3, open(embed_filename, "w+") as f4, open(embed_filename+"2", "w+") as f5:
            # Read node ids
            lines = f1.readlines()
            nodes = []
            for line in lines:
                splited_line = line.split(' ')
                idx = splited_line[0].replace("'","")
                idx = int(idx.replace("b",""))
                nodes.append(idx)
            # Read and write source embeddings
            lines = f2.readlines()
            f4.writelines(str(len(nodes)) + " " + str(num_embed) + "\n")
            index = 0
            for line in lines:
                if index in nodes:
                    f4.writelines(str(nodes[index]) + " " + str(line))
                index += 1
            # Read and write target embeddings
            lines = f3.readlines()
            f5.writelines(str(len(nodes)) + " " + str(num_embed) + "\n")
            index = 0
            for line in lines:
                if index in nodes:
                    f5.writelines(str(nodes[index]) + " " + str(line))
                index += 1


    elif "stne" in algo: #py37
        os.chdir(algo_path + "STNE-master")
        extra_arg = " --directed"
        # --window_size 10 --num_walks 80 --walk_len 40
        # --window_size 10 --num_walks 20 --walk_len 40
        # --window_size 5 --num_walks 20 --walk_len 10
        os.system("python src/main.py --input " + train_filename + " --outward-embedding-path " + embed_path + " --inward-embedding-path " + embed_path +\
            " --output " + embed_filename + " --dim " + str(int((num_embed-32)/2)) + " --n 5 --num_walks 20 --walk_len 40 --window_size 10 --learning-rate 0.025" +\
            " --m 1 --norm 0.01" + extra_arg)
    print("Finish: perform " + algo)

if __name__ == "__main__":
    args = parse_args()
    test(args)

