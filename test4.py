import random
from collections import deque
import copy

class GraphII:

    def __init__(self, nodes):
        self.graph = []
        # node numbered 0-1
        for node in range(nodes):
            self.graph.append([])
    
    def has_edge(self, src, dst):
        return src in self.graph[dst]
    
    def add_edge(self, src, dst):
        if not self.has_edge(src, dst):
            self.graph[src].append(dst)
            self.graph[dst].append(src)
    
    def get_graph(self):
        return self.graph
    
    def has_edges(self):
        return any(self.graph)
    
    def remove_vertex(self, v): # takes in a vertex v as parameter and removes it
        for neighbor in list(self.graph[v]):
            self.graph[neighbor].remove(v) # removes v from every neighbor of v
        self.graph[v] = [] # makes v's list of connections empty

    def get_edges(self):
        edges = []
        for u in range(len(self.graph)):
            for v in self.graph[u]:
                if u < v:
                    edges.append((u, v))
        return edges
        

def is_vertex_cover(G, C):
    for u in range(len(G.graph)):
        for v in G.graph[u]:
            if u not in C and v not in C:
                return False
    return True

def mvc_1(G):
    min_cover = set()
    cop = copy.deepcopy(G)
    while cop.has_edges(): # we keep adding vertices to set C while G has edges assuming
        v = max(range(len(cop.graph)), key=lambda x: len(cop.graph[x]))
        min_cover.add(v)
        cop.remove_vertex(v)
    return min_cover

def mvc_2(G):
    min_cover = set()
    cop = copy.deepcopy(G)
    while cop.has_edges():
        v = random.randint(0, len(cop.graph) - 1)
        if v not in min_cover and len(cop.graph[v]) > 0: # makes sure that every node we add at least adds 1 edge
            min_cover.add(v)
        cop.remove_vertex(v)
        
    return min_cover
        


def mvc_3(G):
    min_cover = set()
    cop = copy.deepcopy(G)
    while cop.get_edges() != []:
        print(cop.get_edges())
        u, v = random.choice(cop.get_edges()) # get a random edge, assign the nodes to u and v
        min_cover.add(u) # adding both vertices of the edge to set C
        min_cover.add(v)
        cop.remove_vertex(u) # removing the given vertices from our set
        cop.remove_vertex(v)
    return min_cover


def create_random_graph(nodes, edges):
    graph = GraphII(nodes)
    graphRep = graph.get_graph()
    # we're creating a graph of type 2:
    for i in range(0, edges):
        added_edge = False
        while not added_edge:
            r1, r2 = random.randint(0,nodes - 1), random.randint(0,nodes - 1)
            if r2 not in graphRep[r1]: #if an edge is succesfully added
                graph.add_edge(r1, r2)
                added_edge = True

    # your implementation for Part 4 goes here
    return graph
def add_to_each(sets, element):
    copy = sets.copy()
    for set in copy:
        set.append(element)
    return copy

def power_set(set):
    if set == []:
        return [[]]
    return power_set(set[1:]) + add_to_each(power_set(set[1:]), set[0])


def MVC(G):
    nodes = [i for i in range(len(G.graph))]
    subsets = power_set(nodes)
    min_cover = nodes
    for subset in subsets:
        if is_vertex_cover(G, subset):
            if len(subset) < len(min_cover):
                min_cover = subset
    return min_cover

oneEdgeMVC3 = []
l = create_random_graph(10, 10)
oneEdgeMVC3.append(mvc_3(l))

for i in range(0, 1000):
    h = create_random_graph(10, 10)
    oneEdgeMVC3.append(mvc_3(h))

print(oneEdgeMVC3)
