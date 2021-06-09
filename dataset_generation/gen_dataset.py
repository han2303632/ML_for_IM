## 路径边权重之和收集算法


import json
from queue import Queue
from re import A
import time
import random
import pickle
import copy
from networkx.readwrite import json_graph

def link_construct(links ,weight=0.1):
    result = []
    for source ,target in links:
        result.append({"source":source ,"target":target, "weight":weight})
    return result
def node_construct(nodes):
    result = []
    for node in nodes:
        result.append({"id":node })
    return result

def load_graph(graph_dir):
    graph = {}
    in_link = {}  # 谁指向我
    out_link = {}  # 我指向谁
    visited = {}
    edge_weight = {}
    node_score = {}
    nodes = set()
    score = {}
    
    with open(graph_dir, 'br') as f:
        graph = json.load(f)
        # graph = {}

        # graph['links'] = link_construct([
        #     (1,2),(2,1), (1,3),(3,1), (2,3),(3,2), (2,4),(4,2),(3,4),(4,3)
        # ])
        # graph['nodes'] = node_construct([1,2,3,4])

        for dic in graph['nodes']:
            node_id = dic['id']
            visited[node_id] = False
            in_link[node_id] = set()
            out_link[node_id] = set()
            node_score[node_id] = 0
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

    graph = json_graph.node_link_graph(graph)
                
    return graph, nodes, node_score, edge_weight, in_link, out_link, visited, score

def isolate(graph, node_id):
    for dic in graph['links']:
        source = dic["source"]
        target = dic["target"]
        
        if node_id == source or node_id == target:
            return False
    return True

def dfs(out_link, node_score, edge_weight, visited, node):
    visited[node] = True
    for nbr in out_link[node]:
        weight = edge_weight[node][nbr]
        if visited[nbr] == False:
            dfs(out_link, node_score, edge_weight, visited, nbr)
        node_score[node] += (weight + weight * node_score[nbr])
    return node_score[node]

def node_remove(Q, in_link, out_link, visited, score):
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
def node_seperation(target_node, in_link, out_link, node_set, left_node_set, visited, edge_weight, score):
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









Q = Queue()

print("load graph")
graph, nodes, node_score, edge_weight, in_link, out_link, visited, score = load_graph("../RL/dataset/train/large_graph-G.json")   
left_node_set = nodes.copy()
out_link_copy = copy.deepcopy(out_link)

for node in in_link.keys():
    if len(out_link[node]) == 0:   # 没有外向边或者外向边被清空
        Q.put(node)
        visited[node] = True
        left_node_set.remove(node)
    
print("load complete")
removed_nodes = node_remove(Q, in_link, out_link, visited, score)
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

    node_seperation(target_node, in_link, out_link, nodes, left_node_set, visited, edge_weight, score)
    
    # target_node + "_2" 会被标记为已访问，因为它已经没有入边了
    only_in_node = target_node + 1000000000
    visited[only_in_node] == True
    Q.put(only_in_node)
    left_node_set.remove(only_in_node)
    removed_nodes = node_remove(Q, in_link, out_link, visited, score)
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

sorted_nodes = sorted(combined_node_score.keys(), key=lambda x:combined_node_score[x], reverse=True)

top_node_score_far= {}
top_node_nbr_far = {}
influenced_nbr = set()

count = 0
# for  node in sorted_nodes[:600]:
for  node in sorted_nodes:
    if count >= 600:
        break

    if node in influenced_nbr:
        continue

    top_node_score_far[node] = combined_node_score[node]
    top_node_nbr_far[node] = out_link_copy[node]
    influenced_nbr = influenced_nbr |  set(graph.neighbors(node))
    count += 1

print(sorted_nodes[:600])


# top_score = [(node, combined_node_score[node]) for node in sorted_nodes[:100]]

with open('top_node_score_far.pickle', 'wb') as f:
    pickle.dump(top_node_score_far,f, 2)
with open('top_node_nbr_far.pickle', 'wb') as f:
    pickle.dump(top_node_nbr_far,f, 2)