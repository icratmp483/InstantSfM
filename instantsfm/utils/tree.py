from queue import Queue
import numpy as np
import networkx as nx

from instantsfm.scene.defs import ViewGraph

def BFS(edges_list, root=0):
    visited = [False] * len(edges_list)
    parents = [-1] * len(edges_list)
    parents[root] = root

    q = Queue()
    visited[root] = True
    q.put(root)

    while not q.empty():
        current = q.get()
        for neighbor in edges_list[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parents[neighbor] = current
                q.put(neighbor)
    return parents

def MaximumSpanningTree(view_graph:ViewGraph, images):    
    num_images = len(images)
    G = nx.Graph()

    for pair in view_graph.image_pairs.values():
        if not pair.is_valid:
            continue
        image1, image2 = images[pair.image_id1], images[pair.image_id2]
        if not image1.is_registered or not image2.is_registered:
            continue
        idx1, idx2 = pair.image_id1, pair.image_id2

        G.add_edge(idx1, idx2, weight=len(pair.inliers))
    
    mst = nx.maximum_spanning_tree(G, weight='weight')
    edges_list = [[] for _ in range(num_images)]
    for u, v in mst.edges():
        edges_list[u].append(v)
        edges_list[v].append(u)
    
    parents_idx = BFS(edges_list)
    parents = [parents_idx[i] for i in range(len(parents_idx))]
    return parents, 0
