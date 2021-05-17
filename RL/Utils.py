from networkx.readwrite  import json_graph
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import json

class Util:
    def __init__(self):
        pass

    def read_json_file(self, graph_dir):
        with open(graph_dir, 'r') as f:
            j_graph = json.load(f)
            return json_graph.node_link_graph(j_graph)

    def load_dataset(self, dataset_dir, num_k, sampling_freq):
        # load trained nodes
        top_nodes = []
        with open(dataset_dir + "large_graph_top_ten_percent.txt_%s_nbs_%s"%(str(num_k), str(sampling_freq)),'r') as f:
            for line in f:
                top_nodes = list(map(int, line.split()))
                break

        # load corresponding embedding
        embeddings_dict = {}
        with open(dataset_dir +"large_graph_embeddings.npy_%s_nbs_%s.pickle"%(str(num_k), str(sampling_freq)), 'rb') as handle:
            embeddings_dict = pickle.load(handle,encoding='iso-8859-1')

        # load graph
        # graph = self.read_json_file(dataset_dir + "large_graph-G.json")




         # load predicting score
        node2score = {}
        with open(dataset_dir + "large_graph_node_scores_supgs_%s_nbs_%s.pickle"%(str(num_k), str(sampling_freq)), 'rb') as handle:
            node2score = pickle.load(handle,encoding='iso-8859-1')

        # embedding: node_id -> score , not  index->score
         # load neighbors
        node2neighbors = {}
        dict_node_sampled_neighbors_file_name=dataset_dir + "large_graph-sampled_nbrs_for_rl.pickle"+"_"+ str(num_k)+"_nbs_"+str(sampling_freq)
        with open(dict_node_sampled_neighbors_file_name, 'rb') as handle:
            node2neighbors =pickle.load(handle,encoding='iso-8859-1')
        
        # graph_size = len(graph)
        # out_deg_wt_graph  = graph.out_degree( weight='weight')

        embeddings = np.ones((len(embeddings_dict), 2), dtype='float64')

        node2idx = {}
        idx2node ={}
        gcn_embedding = []

        for i, key in enumerate(embeddings_dict.keys()):
            embeddings[i][0] = len(node2neighbors[key])
            embeddings[i][1] = node2score[key]
            node2idx[key] = i
            idx2node[i] = key
            gcn_embedding.append(embeddings_dict[key])
        gcn_embedding.append(np.zeros(len(gcn_embedding[0])))

        # normalize
        scaler = StandardScaler()
        scaler.fit(embeddings)
        embedding_norm = scaler.transform(embeddings)

        top_nodes_idx = [node2idx[node] for node in top_nodes]

        # return graph, embedding_norm, graph_size, top_nodes_idx, node2neighbors, idx2node, node2idx
        # return None, embedding_norm, None, top_nodes_idx, node2neighbors, idx2node, node2idx
        return None, embedding_norm, None, top_nodes_idx, node2neighbors, idx2node, node2idx, gcn_embedding



            
            







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

