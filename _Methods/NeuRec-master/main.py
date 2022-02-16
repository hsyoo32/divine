import os
import random
import numpy as np
import tensorflow as tf
import importlib
from data.dataset import Dataset
from util import Configurator, tool
import tqdm # yc_add


np.random.seed(2018)
random.seed(2018)
tf.set_random_seed(2017)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_in_out_degrees(edges):
    d_out = {}
    d_in = {}
    for edge in edges:
        if d_out.get(edge[0]) is None:
            d_out[edge[0]] = 0
        if d_in.get(edge[1]) is None:
            d_in[edge[1]] = 0
        d_out[edge[0]] += 1
        d_in[edge[1]] += 1

        if d_out.get(edge[1]) is None:
            d_out[edge[1]] = 0
        if d_in.get(edge[0]) is None:
            d_in[edge[0]] = 0
    return d_in, d_out

def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]
    return edges

def str_list_to_int(str_list):
    return [int(item) for item in str_list]

if __name__ == "__main__":
    conf = Configurator("NeuRec.properties", default_section="hyperparameters")
    gpu_id = str(conf["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    recommender = conf["recommender"]
    # num_thread = int(conf["rec.number.thread"])

    # if Tool.get_available_gpus(gpu_id):
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    dataset = Dataset(conf)    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = conf["gpu_mem"]
    with tf.Session(config=config) as sess:
        if importlib.util.find_spec("model.general_recommender." + recommender) is not None:
            my_module = importlib.import_module("model.general_recommender." + recommender)
            
        elif importlib.util.find_spec("model.social_recommender." + recommender) is not None:
            
            my_module = importlib.import_module("model.social_recommender." + recommender)
            
        else:
            my_module = importlib.import_module("model.sequential_recommender." + recommender)
        
        MyClass = getattr(my_module, recommender)
        model = MyClass(sess, dataset, conf)

        model.build_graph()
        sess.run(tf.global_variables_initializer())
        model.train_model()
        

        # Added by hsyoo
        #ori_prefix, saved_prefix = dataset._get_data_path(conf)
        # recover original node ids         
        # if conf["data.index.remap"] == True:            
        #     with open(saved_prefix + ".user2id", "r") as f1, open(saved_prefix + ".item2id", "r") as f2:
        #         lines = f1.readlines()
        #         user2id = {}            
        #         for line in lines:
        #             user2id[int(line.split()[1])] = int(line.split()[0])
        #         lines = f2.readlines()
        #         item2id = {}
        #         for line in lines:
        #             item2id[int(line.split()[1])] = int(line.split()[0])

        # write relevance scores
        # edges = read_edges_from_file(ori_prefix)
        # d_in, d_out = get_in_out_degrees(edges)
        
        vne_option = conf["vne_option"]
        print("vne_option: {}".format(vne_option))
        output = conf["data.output.path"]

        if "general" in vne_option:
            with open(output, "w+") as f:
                f.writelines(str(model.num_users) + " " + str(conf["embedding_size"]) + "\n")
                for source_node in tqdm.tqdm(range(model.num_users)):
                    emb = model._cur_user_embeddings[source_node]
                    f.writelines(str(source_node) + " ")                    
                    for i in range(conf["embedding_size"]):
                        f.writelines(str(emb[i])+" ")
                    f.writelines("\n")
            with open(output+"2", "w+") as f:
                f.writelines(str(model.num_items) + " " + str(conf["embedding_size"]) + "\n")
                for target_node in tqdm.tqdm(range(model.num_items)):
                    emb = model._cur_item_embeddings[target_node]
                    f.writelines(str(target_node) + " ")                    
                    for i in range(conf["embedding_size"]):
                        f.writelines(str(emb[i])+" ")
                    f.writelines("\n")

        # else:   
        #     max_vne = -1 # global
        #     capacity = 10
        #     if "uniform" in vne_option:
        #         vne_degree = round(len(edges) / len(d_out))
        #         max_vne = vne_degree * capacity
        #         print("@@@@@@@@@@UNIFORM "+str(max_vne))
        #     elif "out" in vne_option:
        #         print("@@@@@@@@@@OUTPROP")
        #     elif "global" in vne_option:
        #         print("@@@@@@@@@@GLOBAL")
        #         max_vnes = len(edges) * capacity
        #         max_vne = -1
        #         num_missing_edges = len(d_out) * len(d_out)
        #         memory_limit = num_missing_edges/3
        #     else:
        #         print("##########WARNING: No option")

        #     li_gen = []
        #     # tqdm.tqdm(range(model.num_users))
        #     for source_node in tqdm.tqdm(range(model.num_users)):
        #         source_embed = model._cur_user_embeddings[source_node]
        #         relevances = np.matmul(source_embed, model._cur_item_embeddings.T)
        #         target_by_source = dataset.train_matrix[source_node].indices 
        #         if conf["data.index.remap"] == True:
        #             origin_source_id = user2id[source_node]

        #         li = []
        #         for target_node in range(len(relevances)):
        #             rel = relevances[target_node]
        #             if not target_node in target_by_source: # and not rel == 0: ###############
        #                 if conf["data.index.remap"] == True and not rel == 0: # ???
        #                     origin_target_id = item2id[target_node]
        #                     if not origin_source_id == origin_target_id:
        #                         li.append([origin_source_id, origin_target_id, rel])
        #                 else:
        #                     if not source_node == target_node:
        #                         li.append([source_node, target_node, rel])

        #         if "out" in vne_option:
        #             if conf["data.index.remap"] == True:
        #                 seed = origin_source_id
        #             else:
        #                 seed = source_node
        #             if d_out.get(seed) is not None:
        #                 vne_degree = d_out[seed]
        #             else:
        #                 print("@@@@@@@@@@@@@@@@@@@@@@@@WARNING@@@@@@@@@@@@@@@@@@@@@@@@@@")
        #                 vne_degree = 0
        #             max_vne = vne_degree * capacity
        #         elif "global" in vne_option:
        #             if len(li_gen) > memory_limit:
        #                 li_gen = li_gen[:max_vnes]

        #         if max_vne >= 0 and "receiver" not in vne_option:
        #             if "positive" in vne_option:
        #                 li.sort(key=lambda x: x[2], reverse=True)
        #             else:
        #                 li.sort(key=lambda x: x[2])
        #             li = li[:max_vne]
        #         li_gen.extend(li)
            

        #     if "receiver" in vne_option:
        #         li_gen.sort(key=lambda x: x[2])
        #         d = {}
        #         for s in range(model.num_users):
        #             d[s] = [max_vne, 0]
        #         li_receiver = []
        #         for edge in li_gen:
        #             if d[edge[1]][0] > d[edge[1]][1]:
        #                 d[edge[1]][1] += 1
        #                 li_receiver.append(edge)
        #         li_gen = li_receiver
            
        #     if "global" in vne_option:
        #         li_gen.sort(key=lambda x: x[2])
        #         li_gen = li_gen[:max_vnes]

        #     with open(output, "w+") as f:
        #         for l in li_gen:
        #             f.writelines(str(l[0])+ "\t" + str(l[1]) + "\t" + str(l[2]) + "\n")    



