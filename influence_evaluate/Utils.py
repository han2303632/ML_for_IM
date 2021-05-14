import igraph as ig
from igraph import Graph
import snap
import numpy as np
import time
from queue import Queue
from multiprocess import Queue as mpQueue
from multiprocess import current_process, Process, freeze_support

MC_SIM_NUMBER = 10000  #dmonte carlo simulation times
NUMBER_OF_PROCESSES = 6


class Utils:
    def __init__(self):
        # freeze_support()
        pass


    def worker(self, input, output):
        while not input.empty():
            func, args = input.get()
            output.put(func(args))

    def mc_sim_single(self, G, edge_weight, seeds):
        print("start mc sim")
        visited = {}
        influenced_set = set()
        Q_list = Queue()

        # for node in G.Nodes():
        #     visited[node.GetId()] = False



        for seed in seeds:
            visited[seed] = True
            Q_list.put(seed)
            influenced_set.add(seed)

        while(not Q_list.empty()):
            node_id = Q_list.get()
            
            _, nodeVec = G.GetNodesAtHop(node_id, 1, True)

            for nbr in nodeVec:
                if visited.get(nbr) is None and np.random.rand() <= edge_weight[str(node_id) + "-" + str(nbr)]:
                    visited[nbr] = True
                    Q_list.put(nbr)
                    influenced_set.add(nbr)

        return len(influenced_set)

    def load_graph(self, graph_dir):
        G = snap.TNGraph.New()
        edge_weight = snap.TStrFltH()
        with open(graph_dir, 'r') as f:
            for line in f:
                source, dest, weight = line.split(' ')
                if(not G.IsNode(int(source))):
                    G.AddNode(int(source))
                if(not G.IsNode(int(dest))):
                    G.AddNode(int(dest))

                G.AddEdge(int(source), int(dest))
                edge_weight[source + "-" + dest] = float(weight)
        return G, edge_weight

    def load_seeds(self, seeds_dir):
        seeds = []
        with open(seeds_dir, 'r') as f:
            for line in f:
                seeds = list(map(int, line.strip().split(" ")))
                break
        return seeds

    # @input: graph, number of seeds k
    # finding the average influence in 10000 mc simulation
    def influence_evaluate(self, graph_dir, seeds_dir):
        G, edge_weight = self.load_graph(graph_dir)
        seeds = self.load_seeds(seeds_dir)

        '''
        TASK1 = [(mc_sim_single,(graph, edge_weight, seeds)) for i in range(mc_sim_number)]

        task_queue = Queue()
        done_queue = Queue()

        list(map(task_queue.put, TASK1))
        for i in range(NUMBER_OF_PROCESSES):
            Process(target=worker, args=(task_queue, done_queue)).start()

        print("submit complete")

        for i in range(len(TASK1)):
            print(done_queue.get()

            for i in range(mc_sim_number):
        '''
        
        result = 0
        for i in range(10):
            start = time.perf_counter()
            result += self.mc_sim_single(G, edge_weight, seeds)
            end = time.perf_counter()
            print("influence", result/(i+1), "running time:", end-start)

        print(result/10)
            
            

if __name__ == "__main__":
    utils = Utils()
    utils.influence_evaluate("./data/edges.txt", "./data/gcomb_result.txt")
        







