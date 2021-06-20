## 路径边权重之和收集算法


import json
from queue import Queue
from networkx.readwrite.gexf import teardown_module
import numpy as np
import math
import random
import pickle
import copy
from collections import defaultdict
from networkx.readwrite import json_graph
import infomap

class DatasetGenerator():

    def link_construct(self, links ,weight=0.1):
        result = []
        for source ,target in links:
            result.append({"source":source ,"target":target, "weight":weight})
        return result
    def node_construct(self, nodes):
        result = []
        for node in nodes:
            result.append({"id":node })
        return result

    def load_graph(self, graph_dir):
        graph = {}
        in_link = {}  # 谁指向我
        out_link = {}  # 我指向谁
        visited = {}
        edge_weight = {}
        self.node_weight_sum = defaultdict(lambda:0)
        nodes = set()
        score = {}
        self.im = infomap.Infomap("--two-level --flow-model directed")

        with open(graph_dir, 'br') as f:
            graph = json.load(f)
            # graph = {}

            # graph['links'] = self.link_construct([
            #     (1,2), (2,3), (3,4), (4,5)
            # ])
            # graph['nodes'] = self.node_construct([1,2,3,4,5])

            for dic in graph['nodes']:
                node_id = dic['id']
                visited[node_id] = False
                in_link[node_id] = set()
                out_link[node_id] = set()
                edge_weight[node_id] = {}
                score[node_id] = 0

                
            for dic in graph['links']:
                source = dic["source"]
                target = dic["target"]
                out_link[source].add(target)
                in_link[target].add(source)
                edge_weight[source][target] = dic["weight"]
                nodes.add(source)
                nodes.add(target)
                self.node_weight_sum[source] += edge_weight[source][target]

                self.im.add_link(dic['source'], dic['target'])


        # graph = json_graph.node_link_graph(graph)
                    
        return graph, nodes, edge_weight, in_link, out_link, visited, score

    def isolate(self, graph, node_id):
        for dic in graph['links']:
            source = dic["source"]
            target = dic["target"]
            
            if node_id == source or node_id == target:
                return False
        return True

    def dfs(self, out_link, node_score, edge_weight, visited, node):
        visited[node] = True
        for nbr in out_link[node]:
            weight = edge_weight[node][nbr]
            if visited[nbr] == False:
                self.dfs(out_link, node_score, edge_weight, visited, nbr)
            node_score[node] += (weight + weight * node_score[nbr])
        return node_score[node]

    def node_remove(self, Q, in_link, out_link, visited, score, edge_weight):
        removed_nodes = set()
        
        while(not Q.empty()):
            leaf = Q.get()
            in_link_copy = in_link[leaf].copy()
            # print("remove node", leaf, "node score", score[leaf])

            for node in in_link_copy:
                out_link[node].remove(leaf)
                in_link[leaf].remove(node)

                # 分数的传递
                score[node] += (edge_weight[node][leaf] + edge_weight[node][leaf] * score[leaf])
                if len(out_link[node]) == 0 and visited[node] == False:
                    visited[node] = True
                    Q.put(node)
                    removed_nodes.add(node)
        return removed_nodes
    def node_seperation(self, target_node, in_link, out_link, node_set, left_node_set, visited, edge_weight, score):
        # 删除所有有关target_node 的链接
        in_link_copy = in_link[target_node].copy()
        out_link_copy = out_link[target_node].copy()
        new_node_1 = target_node + 1000000000
        new_node_2 = target_node + 2000000000

        # link initialization
        in_link[new_node_1] = set()
        out_link[new_node_1] = set()
        in_link[new_node_2] = set()
        out_link[new_node_2] = set()

        # score initialization
        score[new_node_1] = score[target_node]
        score[new_node_2] = score[target_node]

        # edge_weight initialization
        edge_weight[new_node_1] = {}
        edge_weight[new_node_2] = {}


        
        for nbr in in_link_copy:
            in_link[target_node].remove(nbr)
            out_link[nbr].remove(target_node)
            in_link[new_node_1].add(nbr)
            out_link[nbr].add(new_node_1)

            edge_weight[nbr][new_node_1] = edge_weight[nbr][target_node]

            
        for nbr in out_link_copy:
            out_link[target_node].remove(nbr)
            in_link[nbr].remove(target_node)
            out_link[new_node_2].add(nbr)
            in_link[nbr].add(new_node_2)

            edge_weight[new_node_2][nbr] = edge_weight[target_node][nbr]
            
        visited[new_node_1] = False
        visited[new_node_2] = False
        del visited[target_node]
        
        node_set.remove(target_node)
        node_set.add(new_node_1)
        node_set.add(new_node_2)
        
        left_node_set.remove(target_node)
        left_node_set.add(new_node_1)
        left_node_set.add(new_node_2)




    def run2(self, graph_size):
        self.im.run()

        module_member = defaultdict(list)
        module_size = defaultdict(lambda: 0)
        print(len(list(self.im.nodes)))

        for node in self.im.nodes:
            module_member[node.module_id].append(node.node_id)
            module_size[node.module_id] += 1

        sorted_module_id = sorted(module_member.keys(), key=lambda x:module_size[x], reverse=True)
        used_modules = [module_member[id] for id in sorted_module_id[:int(math.sqrt(graph_size))]]

        return used_modules


    def run(self, graph, nodes, edge_weight, in_link, out_link, visited, score):

        Q = Queue()

        print("load graph")
        left_node_set = nodes.copy()

        for node in in_link.keys():
            if len(out_link[node]) == 0:   # 没有外向边或者外向边被清空
                Q.put(node)
                visited[node] = True
                left_node_set.remove(node)
            
        print("load complete")
        removed_nodes = self.node_remove(Q, in_link, out_link, visited, score, edge_weight)
        left_node_set -= removed_nodes


        print("number of left_nodes", len(left_node_set))
        print("number of total node", len(nodes))


        # 随机选择一个结点,对他的入度边和出度边进行一个拆分
        left_nodes = list(left_node_set)
        random.shuffle(left_nodes)

        for target_node in left_nodes:
            if visited[target_node] == True:
                continue
            if in_link[target_node] == set():
                continue

            self.node_seperation(target_node, in_link, out_link, nodes, left_node_set, visited, edge_weight, score)
            
            # target_node + "_2" 会被标记为已访问，因为它已经没有入边了
            only_in_node = target_node + 1000000000
            visited[only_in_node] == True
            Q.put(only_in_node)
            left_node_set.remove(only_in_node)
            removed_nodes = self.node_remove(Q, in_link, out_link, visited, score, edge_weight)
            left_node_set -= removed_nodes
            # -----------------------------------

            # print("number of new_node", len(left_node_set))

        combined_node_score = {}

        for node in nodes:
            if node >= 1000000000:
                origin_node = node % 1000000000
                # combined_node_score[origin_node] = max(score[1000000000 + origin_node] + score[2000000000 + origin_node]) /2
                combined_node_score[origin_node] = max([score[1000000000 + origin_node],score[2000000000 + origin_node]])
            else:
                combined_node_score[node] = score[node]

        return combined_node_score

        
  
        '''
        # top_score = [(node, combined_node_score[node]) for node in sorted_nodes[:100]]
        '''

        with open('top_node_score_far.pickle', 'wb') as f:
            pickle.dump(top_node_score_far,f, 2)
        with open('top_node_nbr_far.pickle', 'wb') as f:
            pickle.dump(top_node_nbr_far,f, 2)

    def bfs(self, nodes, hop_limit, out_link):
        queue = Queue()
        visited_nodes = []
        visited_map = defaultdict(lambda:False)

        for node_id in nodes:
            visited_map[node_id] = True
            queue.put((node_id,0))

        while( not queue.empty()):
            node_id, hop = queue.get()
            visited_nodes.append(node_id)
            # print("current node:%s"%(node_id))

            # 标记访问
            for adj_id in out_link[node_id]:
                # print("trying activate node:%s"%(adj_id))
                if visited_map[adj_id] == False and hop < hop_limit:
                    visited_map[adj_id] = True
                    queue.put((adj_id, hop+1))

        return visited_nodes


def get_top_nodes(node_type, *args):
    if node_type == "origin":
        top_nodes = []
        dataset_dir, num_k = args
        with open(dataset_dir + "top_node_%s.txt"%(str(num_k)),'r') as f:
            for line in f:
                top_nodes = list(map(int, line.split()))
                break
    elif node_type == 'path_weighted':
        node_score, out_link = args
        sorted_nodes = sorted(node_score.keys(), key=lambda x:node_score[x], reverse=True)
        influenced_nbr = set()

        count = 0
        top_nodes = []
        # for  node in sorted_nodes[:600]:
        for  node in sorted_nodes:
            if count >= 627:
                break

            if node in influenced_nbr:
                continue
            if len(set(top_nodes) & set(out_link[node])) != 0:
                continue

            top_nodes.append(node)
            influenced_nbr = influenced_nbr |  set(out_link[node])
            count += 1
    elif node_type == "sqrt_n":
        node_weight_sum, graph_size, out_link = args
        sorted_nodes = sorted(node_weight_sum.keys(), key=lambda x:node_weight_sum[x], reverse=True)

        required_num = int(math.sqrt(graph_size))
        # required_num = int(600)

        top_nodes = sorted_nodes[:required_num]

        influenced_nbr = set()

        count = 0
        top_nodes = []
        for  node in sorted_nodes:
            if count >= required_num:
                break

            if node in influenced_nbr:
                continue
            if len(set(top_nodes) & set(out_link[node])) != 0:
                continue

            top_nodes.append(node)
            influenced_nbr = influenced_nbr |  set(out_link[node])
            count += 1
        # for  node in sorted_nodes[:600]:
        '''
        for  node in sorted_nodes:
            if count >= required_num:
                break

            if node in influenced_nbr:
                continue

            two_hop_nbrs = set([node])
            for nbr in out_link[node]:
                two_hop_nbrs.update(out_link[nbr])
                two_hop_nbrs.add(nbr)

            if len(influenced_nbr & two_hop_nbrs) != 0:
                continue

            top_nodes.append(node)
            influenced_nbr = influenced_nbr |  two_hop_nbrs
            count += 1
            '''
    '''    
    elif node_type == 'community':
        nodes_weight_sum, modules = args
        top_nodes = []
        for members in modules:
            best = members[0]
            for node in members:
                if nodes_weight_sum[node] > nodes_weight_sum[best]:
                    best = node
            top_nodes.append(best)
    '''

    return top_nodes



if __name__ == '__main__':
    num_k = 20
    dataset_dir = "../GCOMB++/dataset/train/"
    

    generator = DatasetGenerator()

    graph, nodes, edge_weight, in_link, out_link, visited, score = generator.load_graph(dataset_dir + "large_graph-G.json")

    # 复制 out link
    out_link_copy = copy.deepcopy(out_link)

    # 计算结点score
    nodes_score = generator.run(graph, nodes, edge_weight, in_link, out_link_copy, visited, score)
    # used_module = generator.run2(len(nodes))
    del out_link_copy

    # top_nodes = get_top_nodes("path_weighted", nodes_score, out_link)
    # top_nodes = get_top_nodes("community", generator.node_weight_sum, used_module)
    top_nodes = get_top_nodes("sqrt_n", generator.node_weight_sum, len(nodes), out_link)
    # 返回k跳以内邻居
    nodes_at_hop = generator.bfs(top_nodes, 2, out_link)

    nodes_features = np.zeros((len(nodes_at_hop)+1,  3))
    nodes_nbr = {}
    nodes_idx = {}
    new_edge_weight = {}

    print("information gathering")
    for idx, node in enumerate(nodes_at_hop):
        nodes_idx[node] = idx
        node_score = nodes_score[node]
        node_score2 = 1 if node in top_nodes else 0
        node_weight_sum = generator.node_weight_sum[node]
        nodes_features[idx,:] = [node_score, node_weight_sum, node_score2]
        # nodes_features[idx,:] = [node_weight_sum, node_score2]

    for node in nodes_at_hop:
        nbrs_idx = []
        for nbr in out_link[node]:
            if nodes_idx.get(nbr) is not None:
                nbrs_idx.append(nodes_idx[nbr])
                new_edge_weight[(nodes_idx[node], nodes_idx[nbr])] = edge_weight[node][nbr]
        nodes_nbr[nodes_idx[node]] = nbrs_idx

    print("write start")
    nodes_info = {"top_nodes":top_nodes, "nodes_idx":nodes_idx, "nodes_nbr":nodes_nbr,
                    "nodes_features":nodes_features,"graph_size":len(nodes), 
                    "num_used_nodes":len(nodes_at_hop)+1, "edge_weight":new_edge_weight}

    with open(dataset_dir + "nodes_info.pickle", 'wb') as f:
        pickle.dump(nodes_info, f)
    

    print("write complete")





