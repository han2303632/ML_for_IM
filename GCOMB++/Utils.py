from networkx.readwrite  import json_graph
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import numpy as np
import torch
import pickle
import json

class Util:
    def __init__(self):
        pass

    def load_data(self, dataset_dir, num_k):
        '''
        top_nodes = []
        with open(dataset_dir + "top_node_%d.txt"%(num_k),'r') as f:
            for line in f:
                top_nodes = list(map(int, line.split()))
                break

        nodes_nbr = {}
        with open(dataset_dir + 'nodes_nbr.pickle','rb') as f:
            nodes_nbr = pickle.load(f)
        nodes_idx = {}
        with open(dataset_dir + 'nodes_idx.pickle','rb') as f:
            nodes_idx = pickle.load(f)
        nodes_features = {}
        with open(dataset_dir + 'nodes_features.pickle','rb') as f:
            nodes_features = pickle.load(f)
        '''
        nodes_info = {}
        with open(dataset_dir + 'nodes_info.pickle','rb') as f:
            nodes_info = pickle.load(f)

        top_nodes = nodes_info["top_nodes"]
        nodes_features = nodes_info["nodes_features"]
        nodes_nbr = nodes_info["nodes_nbr"]
        nodes_idx = nodes_info["nodes_idx"]


        # scaler = StandardScaler()
        # scaler.fit(nodes_features)
        # nodes_features = scaler.transform(nodes_features)
        nodes_features = F.normalize(torch.FloatTensor(nodes_features), p=2, dim=0)

        id2idx = nodes_idx
        idx2id = {}
        for key, value in id2idx.items():
            idx2id[value] = key

        return [id2idx[node] for node in top_nodes], nodes_nbr, torch.FloatTensor(nodes_features).cuda(), id2idx, idx2id, nodes_info["num_used_nodes"]


        
        



            
            







    '''
        
	m = len(main_graph)
	empty_column_for_cover = np.array([2], dtype='float64')
	print("deg")
	out_deg_wt_graph  = main_graph.out_degree( weight='weight')
	for k, v in embeddings_dict.items():
			print("k ", k)
			embeddings_dict[k]= np.array([1,1], dtype='float64')# np.concatenate((empty_column_for_cover, embeddings_dict[k]))#np.array([1,1], dtype='float64')#empty_column_for_cover#np.concatenate((empty_column_for_cover))#, empty_column_for_cover))
			embeddings_dict[k][0] =len( dict_node_sampled_neighbors[k])#main_graph.degree(k)
			embeddings_dict[k][1]= dict_sup_gs_scores[k]#main_graph.degree(k)#out_deg_wt_graph[k]#dict_sup_gs_scores[k]#main_graph.degree(k)
			print(k, out_deg_wt_graph[k])
	#		embeddings_dict[k][1]=dict_sup_gs_scores[k]#main_graph.degree(k)

	print(embeddings_dict)
	scaler=StandardScaler()
	temp_column_for_cover=np.ones((len(embeddings_dict), 2), dtype='float64')

	i=0
	dict_map_i_key={}
	for key, value in embeddings_dict.items():
		temp_column_for_cover[i]=value
		dict_map_i_key[i]=key
		i+=1

	scaler.fit(temp_column_for_cover)
	temp_column_for_cover_norm=None
	temp_column_for_cover_norm=scaler.transform(temp_column_for_cover)

	#
	# pca=PCA(n_components=2)
	# temp_column_for_cover_norm_pca = pca.fit_transform(temp_column_for_over_norm[:,1:])

	for index, value in enumerate(temp_column_for_cover_norm):
		true_node_id=dict_map_i_key[index]
		embeddings_dict[true_node_id][0]=value[0]#np.concatenate((np.array([temp_column_for_cover_norm[index][0]]),value))
		embeddings_dict[true_node_id][1]=value[1]
		#embeddings_dict[true_node_id][2]=value[2]
    '''

