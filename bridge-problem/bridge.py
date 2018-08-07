from itertools import combinations

import networkx as nx
import matplotlib.pyplot as mp

import pandas as pd
import numpy as np

def generate_bridge_layout(n):
    """Generate undirected graph for bridge layout"""

    graph = nx.Graph()

    m = n+1

    for i in range(n+2):
        for j in range(n+1):
            if i < n+1:
                # vertical bridges
                graph.add_edge((i, j), (i+1, j))
            if j < n and i > 0 and i < n+1:
                # horizontal bridges
                graph.add_edge((i, j), (i, j+1))
            if i == 0:
                # paths between start node and south riverbank nodes
                graph.add_edge('start', (0, j))
            if i == n+1:
                # paths between north riverbank nodes and end node
                graph.add_edge((n+1, j), 'end')

    return graph

def nbridge(n):
    return n**2 + (n+1)**2

def generate_solutions(graph):
    # get list of bridges
    edges = []
    for edge in graph.edges:
        if edge[0] in ['start', 'end'] or edge[1] in ['start', 'end']:
            continue
        edges.append(edge)

    n_bridges = len(edges)

    solutions = []
    for n_bridges_removed in range(0, n_bridges+1):
        # loop over all combinations bridges with n_bridges_removed removed
        for bridges_removed in combinations(edges, n_bridges_removed):
            graph.remove_edges_from(bridges_removed) # remove the bridges
            result = nx.has_path(graph, 'start', 'end') # check if a path exists
            graph.add_edges_from(bridges_removed) # reinsert the bridges

            #append solution
            solutions.append(dict(path=result, n_bridges_removed=n_bridges_removed))

    return pd.DataFrame(solutions)


def draw_bridge_layout(graph, n, filename='graph.png', show=True):
    """draw bridge layout"""

    pos = dict()
    for node in graph.nodes:
        if node == 'start':
            pos[node] = [n/2, -1]
        elif node == 'end':
            pos[node] = [n/2, n+2]
        else:
            pos[node] = list(node)[::-1]

    nx.draw(graph, pos, with_labels=True, node_size=1000, node_color='g', font_size=10,
            font_color='w', font_weight='bold')
    mp.axis('equal')
    if show:
        mp.show()
    else:
        mp.savefig(filename, dpi=300)

def crossing_probability(probabilities, n, df):
    """probability of crossing the river where p is probability of bridge staying"""
    n_bridge = nbridge(n)

    df_temp = df[df['path']]

    fun = lambda x, y: np.sum(y**(n_bridge - x) * (1-y)**x)

    bridge_crossing_probabilities = np.empty(probabilities.size, dtype=np.float64)
    for i, p in enumerate(probabilities):
        bridge_crossing_probabilities[i] = fun(df_temp.n_bridges_removed, p)

    return bridge_crossing_probabilities

if __name__ == '__main__':

    N = 2

    BRIDGE_LAYOUT = generate_bridge_layout(N)
    BRIDGE_SOLUTION = generate_solutions(BRIDGE_LAYOUT)
