

from collections import deque
import random
import time
import timeit 
import matplotlib.pyplot as plt
import numpy as np
import copy

# Utility functions - some are implemented, others you must implement yourself.

# function to plot the bar graph and average runtimes of N trials
# Please note that this function only plots the graph and does not save it
# To save the graphs you must use plot.save(). Refer to matplotlib documentation
def draw_plot(run_arr, mean, title):
    x = np.arange(0, len(run_arr),1)
    fig=plt.figure(figsize=(20,8))
    plt.axhline(mean,color="red",linestyle="--",label="Avg")
    plt.bar(range(0, len(run_arr)), run_arr)
    plt.xlabel("Functions and Edge Counts")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.savefig(title)

def draw_plot2(run_arr, mean, title):
    x = np.arange(0, len(run_arr),1)
    fig=plt.figure(figsize=(20,8))
    plt.axhline(mean[0],color="red",linestyle="--",label="Avg")
    plt.axhline(mean[1],color="red",linestyle="--",label="Avg")
    plt.axhline(mean[2],color="red",linestyle="--",label="Avg")
    plt.bar(range(0, len(run_arr)), run_arr)
    plt.xlabel("Functions and Edge Counts")
    plt.ylabel("Proportion of MVC")
    plt.title(title)
    plt.savefig(title)

# function to generate random graphs 
# @args : nodes = number of nodes 
#       : edges = number of edges
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
# please select one representation from either Graph I or Graph II 
# for this assignment,
# please remove the representation you are not using

# graph implementation using hash map 
class GraphI:

    # using hash map
    def __init__(self, edges):
        self.graph = {}
        for x,y in edges:
            if x not in self.graph.keys():
                self.graph[x]=[]
            self.graph[x].append(y)

    def has_edge(self, src, dst):
        return dst in self.graph[src]

    def get_graph_size(self,):
        return len(self.graph)
    
    def get_graph(self,):
        return self.graph
    
    def has_cycle(self,):
        visited = [] # list of all the visited vertices
        previous = [] # this will tell us what node got us to the given node, so the second node will tell us
        # what node got us to 
        queue = deque() # queue to store the order of traversal 
        graphNew = self.get_graph()

        
        
        for node in graphNew: 
            previous.append(-1) # not sure what to set here
            visited.append(False)

        for node in graphNew[0]:
            queue.append(node)
            previous[node] = 0
            # add every node from source to queue
        visited[0] = True
        while queue: # while loop to run until the stack (queue) is empty
            q = queue.pop() # pop the next thing

            visited[q] = True

            for node in graphNew[q]: # otherwise, we loop through the given adjacency list
                # until we find the next node to visit

                if visited[node] and previous[q] != node: # if the node we're look at has been visited we skip it
                    return True
                elif visited[node]:
                    continue
                else: # otherwise we append it to our queue
                    queue.append(node)
                    previous[node] = q
            
        return False
    
    def is_connected(self,node1,node2):
        visited = [] # list of all the visited vertices
        previous = [] # this will tell us what node got us to the given node, so the second node will tell us
        # what node got us to 
        queue = deque() # queue to store the order of traversal 
        graphNew = self.get_graph()


        for node in graphNew: 
            previous.append(None) # not sure what to set here
            visited.append(False)
        
        for node in graphNew[node1]:
            queue.appendleft(node)
            # while the queue isn't empty

        visited[node1] = True # we start at the first node so we set visited to true

        while queue:
            # we go through the first element of the queue and we:
            q = queue.pop()

            if q == node2:
                return True

            # add all other visitable nodes to the queue
            # we also add all whatever node we're currently on to the previous node
            for node in graphNew[q]:
                if not visited[node]:
                    queue.appendleft(node) # not certain if this works
                    # we put the previous node here because
                    previous[node] = q #put node number here, not sure if this works
            
            visited[q] = True # When we're done with a given node we say we've visited the node

        # does a final check to see if the 2 nodes were both visited
        if visited[node1] is True and visited[node2] is True: 
            return True
        return False
    
        # added functions to help with part 6
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
        
    
# graph implementation using adjacency list   
class GraphII:
    # using adjacency list
    def __init__(self, nodes):
        self.graph = []
        # node numbered 0-1
        for node in range(nodes):
            self.graph.append([])
        
    def has_edge(self, src, dst):
        return src in self.graph[dst]
    
    def add_edge(self,src,dst):
        if not self.has_edge(src,dst):
            self.graph[src].append(dst)
            self.graph[dst].append(src)
    
    def get_graph(self,):
        return self.graph
    
    def has_cycle(self,):
        visited = [] # list of all the visited vertices
        previous = [] # this will tell us what node got us to the given node, so the second node will tell us
        # what node got us to 
        queue = deque() # queue to store the order of traversal 
        graphNew = self.get_graph()

        
        
        for node in graphNew: 
            previous.append(-1) # not sure what to set here
            visited.append(False)

        for node in graphNew[0]:
            queue.append(node)
            previous[node] = 0
            # add every node from source to queue
        visited[0] = True
        while queue: # while loop to run until the stack (queue) is empty
            q = queue.pop() # pop the next thing

            visited[q] = True

            for node in graphNew[q]: # otherwise, we loop through the given adjacency list
                # until we find the next node to visit
                if visited[node] and previous[q] != node: # if the node we're look at has been visited we skip it
                    return True
                elif visited[node]:
                    continue
                else: # otherwise we append it to our queue
                    queue.append(node)
                    previous[node] = q
            
        return False
    
    def is_connected(self,node1,node2):
        visited = [] # list of all the visited vertices
        previous = [] # this will tell us what node got us to the given node, so the second node will tell us
        # what node got us to 
        queue = deque() # queue to store the order of traversal 
        graphNew = self.get_graph()


        for node in graphNew: 
            previous.append(None) # not sure what to set here
            visited.append(False)
        
        for node in graphNew[node1]:
            queue.appendleft(node)
            # while the queue isn't empty

        visited[node1] = True # we start at the first node so we set visited to true

        while queue:
            # we go through the first element of the queue and we:
            q = queue.pop()

            if q == node2:
                return True

            # add all other visitable nodes to the queue
            # we also add all whatever node we're currently on to the previous node
            for node in graphNew[q]:
                if not visited[node]:
                    queue.appendleft(node) # not certain if this works
                    # we put the previous node here because
                    previous[node] = q #put node number here, not sure if this works
            
            visited[q] = True # When we're done with a given node we say we've visited the node

        # does a final check to see if the 2 nodes were both visited
        if visited[node1] is True and visited[node2] is True: 
            return True
        return False
    
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
    
def BFS_2(graph,src,dst):
    path = [] # stores the path of the graph
    visited = [] # list of all the visited vertices
    previous = [] # this will tell us what node got us to the given node, so the second node will tell us
    # what node got us to 
    queue = deque() # queue to store the order of traversal 
    graphNew = graph.get_graph()


    for node in graphNew: 
        previous.append(None) # not sure what to set here
        visited.append(False)
    
    for node in graphNew[src]:
        queue.appendleft(node)
        # while the queue isn't empty
    while queue:
        # we go through the first element of the queue and we:
        q = queue.pop()

        if q == dst:
            break

        # add all other visitable nodes to the queue
        # we also add all whatever node we're currently on to the previous node
        for node in graphNew[q]:
            if not visited[node]:
                queue.appendleft(node) # not certain if this works
                # we put the previous node here because
                previous[node] = q #put node number here, not sure if this works
        
        visited[q] = True # When we're done with a given node we say we've visited the node
    
    if visited[dst]: # if the destination has been visited, then we loop through the previous
        # array (which holds how we got the given array)
        temp = previous[dst]
        path.append(temp)
        while temp != src:

            path.append(previous[temp])
            temp = previous[temp]
        for i in range(0, (len(path)) // 2):
            path[i], path[len(path) - i - 1] = path[len(path) - i - 1], path[i]
    else:
        print("Path not found")
    path.append(dst)
    return path

def DFS_2(graph,src,dst):
    path = [] # stores the path of the graph
    visited = [] # list of all the visited vertices
    previous = [] # this will tell us what node got us to the given node, so the second node will tell us
    # what node got us to 
    queue = deque() # queue to store the order of traversal 
    graphNew = graph.get_graph()

    
    
    for node in graphNew: 
        previous.append(-1) # not sure what to set here
        visited.append(False)

    for node in graphNew[src]:
        queue.append(node)
        previous[node] = src
        # add every node from source to queue
    
    while queue: # while loop to run until the stack (queue) is empty
        q = queue.pop() # pop the next thing

        visited[q] = True

        if q == dst: # if the destination is q, we can skip the rest
            break


        for node in graphNew[q]: # otherwise, we loop through the given adjacency list
            # until we find the next node to visit

            if visited[node]: # if the node we're look at has been visited we skip it
                continue
            else: # otherwise we append it to our queue
                queue.append(node)
                previous[node] = q

    if visited[dst]: # if the destination has been visited, then we loop through the previous
        # array (which holds how we got the given array)
        temp = previous[dst]
        path.append(temp)
        while temp != src and temp != -1:
            path.append(previous[temp])
            temp = previous[temp]
        for i in range(0, (len(path)) // 2):
            path[i], path[len(path) - i - 1] = path[len(path) - i - 1], path[i]
    else:
        print("Path not found")
    path.append(dst)
    return path

def BFS_3(graph,src):
    path = [] # stores the path of the graph
    visited = [] # list of all the visited vertices
    previous = [] # this will tell us what node got us to the given node, so the second node will tell us
    # what node got us to 
    queue = deque() # queue to store the order of traversal 
    graphNew = graph.get_graph()


    for node in graphNew: 
        previous.append(-1) # not sure what to set here
        visited.append(False)
    
    for node in graphNew[src]:
        queue.appendleft(node)
        previous[node] = src
        # while the queue isn't empty
    while queue:
        # we go through the first element of the queue and we:
        q = queue.pop()

        # add all other visitable nodes to the queue
        # we also add all whatever node we're currently on to the previous node
        for node in graphNew[q]:
            if not visited[node]:
                queue.appendleft(node) # not certain if this works
                # we put the previous node here because
                previous[node] = q #put node number here, not sure if this works
        
        visited[q] = True # When we're done with a given node we say we've visited the node

    previous[src] = src
    for i in range(0, len(graphNew) - 1):

        temporaryArr = [] # temporary array stores every individual path to node i

        if visited[i]: # if the destination has been visited, then we loop through the previous
            # array (which holds how we got the given array)

            # we use temp to loop through the array by checking what node lead to the given node we're examining
            # eventually, assuming the given node was visited starting from the source, we'll get back to the start
            temp = previous[i]
            temporaryArr.append(temp)
            while temp != src and temp != -1: # Stops the while when we've completed the loop back to node i
                temporaryArr.append(previous[temp])
                temp = previous[temp]
            temporaryArr.reverse() # reverses the list since appending previous nodes creates a list in reverse order
        else:
            print("No path available to node " + str(i)) # error message
        if temporaryArr == [-1]: # case of -1, means there is no path
            temporaryArr = []
        path.append(temporaryArr) # appending path to the path array
    return path
    

def DFS_3(graph,src):
    path = [] # stores the path of the graph
    visited = [] # list of all the visited vertices
    previous = [] # this will tell us what node got us to the given node, so the second node will tell us
    # what node got us to 
    queue = deque() # queue to store the order of traversal 
    graphNew = graph.get_graph()
    
    for node in graphNew: 
        previous.append(-1) # not sure what to set here
        visited.append(False)

    queue.append(src) # adds src to the queue
    visited[src] = True
    
    while queue: # while loop to run until the stack (queue) is empty
        q = queue.pop() # pop the next thing

        visited[q] = True


        for node in graphNew[q]: # otherwise, we loop through the given adjacency list
            # until we find the next node to visit

            if visited[node]: # if the node we're look at has been visited we skip it
                continue
            else: # otherwise we append it to our queue
                queue.append(node)
                previous[node] = q
    # after this, we should've explored every node and know what node leads to said node
    # now we need to print out the path to every node

    for i in range(0, len(graphNew) - 1):

        temporaryArr = [] # temporary array stores every individual path to node i

        if visited[i]: # if the destination has been visited, then we loop through the previous
            # array (which holds how we got the given array)

            # we use temp to loop through the array by checking what node lead to the given node we're examining
            # eventually, assuming the given node was visited starting from the source, we'll get back to the start
            temp = previous[i]
            temporaryArr.append(temp)
            while temp != src and temp != -1: # Stops the while when we've completed the loop back to node i
                temporaryArr.append(previous[temp])
                temp = previous[temp]
            temporaryArr.reverse() # reverses the list since appending previous nodes creates a list in reverse order
        else:
            print("No path available to node " + str(i)) # error message
        if temporaryArr == [-1]: # case of -1, means there is no path
            temporaryArr = []
        path.append(temporaryArr) # appending path to the path array
    return path
#Utility functions to determine minimum vertex covers
def add_to_each(sets, element):
    copy = sets.copy()
    for set in copy:
        set.append(element)
    return copy

def power_set(set):
    if set == []:
        return [[]]
    return power_set(set[1:]) + add_to_each(power_set(set[1:]), set[0])

def is_vertex_cover(G, C):
    for start in range(0, len(G.graph)):
        for end in G.graph[int(start)]:
            if not(start in C or end in C):
                return False
    return True

def MVC(G):
    nodes = [i for i in range(len(G.graph))]
    subsets = power_set(nodes)
    min_cover = nodes
    for subset in subsets:
        if is_vertex_cover(G, subset):
            if len(subset) < len(min_cover):
                min_cover = subset
    return min_cover

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
        u, v = random.choice(cop.get_edges()) # get a random edge, assign the nodes to u and v
        min_cover.add(u) # adding both vertices of the edge to set C
        min_cover.add(v)
        cop.remove_vertex(u) # removing the given vertices from our set
        cop.remove_vertex(v)
    return min_cover

def experiment_1(iterations, graphs, nodes, edges):
    # current test values:
    #iterations, graphs, nodes, edges = 10, 100, 30, 30
    has_cycle = []
    for i in range(0, iterations):
        NumWithCycle = 0
        for i in range(0, graphs):
            graphy = create_random_graph(nodes, edges)
            if graphy.has_cycle():
                NumWithCycle += 1
        NumWithCycle = (NumWithCycle / graphs) * 100
        has_cycle.append(NumWithCycle)
    ave = 0
    for i in range(0, len(has_cycle) - 1):
        ave += has_cycle[i]
    ave = ave / len(has_cycle)
    print(ave)
    draw_plot(has_cycle, ave, "Percent of graphs with a cycle with 10 nodes and 10 edges")
    # your implementation for experiment in part 5 goes here
    return has_cycle

def graphGen(graphnum, nodes, edges):
    # current test values:
    #iterations, graphs, nodes, edges = 10, 100, 30, 30
    graphs = []
    for i in range(0, graphnum):
        graphy = create_random_graph(nodes, edges)
        graphs.append(graphy)

    # your implementation for experiment in part 5 goes here
    return graphs

def experiment_2():
    # part a

    oneEdge = graphGen(100, 6, 1)
    fiveEdges = graphGen(100, 6, 5)
    tenEdges = graphGen(100, 6, 10)
    fifteenEdges = graphGen(100, 6, 15)
    twentyEdges = graphGen(100, 6, 20)


    # part b
    # need to find the mvc of all these graphs?
    oneEdgeMVCL = []
    fiveEdgeMVCL = []
    tenEdgeMVCL = []
    fifteenEdgeMVCL = []
    twentyEdgeMVCL = []
    
    oneEdgeMVC = 0
    fiveEdgeMVC = 0
    tenEdgeMVC = 0
    fifteenEdgeMVC = 0
    twentyEdgeMVC = 0

    oneEdgeMVC1 = 0
    fiveEdgeMVC1 = 0
    tenEdgeMVC1 = 0
    fifteenEdgeMVC1 = 0
    twentyEdgeMVC1 = 0

    oneEdgeMVC2 = 0
    fiveEdgeMVC2 = 0
    tenEdgeMVC2 = 0
    fifteenEdgeMVC2 = 0
    twentyEdgeMVC2 = 0

    oneEdgeMVC3 = 0
    fiveEdgeMVC3 = 0
    tenEdgeMVC3 = 0
    fifteenEdgeMVC3 = 0
    twentyEdgeMVC3 = 0
    for i in range(0, len(oneEdge) - 1):
        oneEdgeMVCL.append(MVC(oneEdge[i]))
        oneEdgeMVC += len(oneEdgeMVCL[i])
        fiveEdgeMVCL.append(MVC(fiveEdges[i]))
        fiveEdgeMVC += len(fiveEdgeMVCL[i])
        tenEdgeMVCL.append(MVC(tenEdges[i]))
        tenEdgeMVC += len(tenEdgeMVCL[i])
        fifteenEdgeMVCL.append(MVC(fifteenEdges[i]))
        fifteenEdgeMVC += len(fifteenEdgeMVCL[i])
        twentyEdgeMVCL.append(MVC(twentyEdges[i]))
        twentyEdgeMVC += len(twentyEdgeMVCL[i])

        oneEdgeMVC1 += len(mvc_1(oneEdge[i]))
        fiveEdgeMVC1 += len(mvc_1(fiveEdges[i]))
        tenEdgeMVC1 += len(mvc_1(tenEdges[i]))
        fifteenEdgeMVC1 += len(mvc_1(fifteenEdges[i]))
        twentyEdgeMVC1 += len(mvc_1(twentyEdges[i]))

        oneEdgeMVC2 += len(mvc_2(oneEdge[i]))
        fiveEdgeMVC2 += len(mvc_2(fiveEdges[i]))
        tenEdgeMVC2 += len(mvc_2(tenEdges[i]))
        fifteenEdgeMVC2 += len(mvc_2(fifteenEdges[i]))
        twentyEdgeMVC2 += len(mvc_2(twentyEdges[i]))

        oneEdgeMVC3 += len(mvc_3(oneEdge[i]))
        fiveEdgeMVC3 += len(mvc_3(fiveEdges[i]))
        tenEdgeMVC3 += len(mvc_3(tenEdges[i]))
        fifteenEdgeMVC3 += len(mvc_3(fifteenEdges[i]))
        twentyEdgeMVC3 += len(mvc_3(twentyEdges[i]))

    graphArr = []
    graphMeans = []
    graphArr.append(oneEdgeMVC1 / oneEdgeMVC)
    graphArr.append(fiveEdgeMVC1 / fiveEdgeMVC)
    graphArr.append(tenEdgeMVC1 / tenEdgeMVC)
    graphArr.append(fifteenEdgeMVC1 / fifteenEdgeMVC)
    graphArr.append(twentyEdgeMVC1 / twentyEdgeMVC)
    graphMeans.append((graphArr[0] + graphArr[1] + graphArr[2] + graphArr[3] + graphArr[4]) / 5)

    graphArr.append(oneEdgeMVC2 / oneEdgeMVC)
    graphArr.append(fiveEdgeMVC2 / fiveEdgeMVC)
    graphArr.append(tenEdgeMVC2 / tenEdgeMVC)
    graphArr.append(fifteenEdgeMVC2 / fifteenEdgeMVC)
    graphArr.append(twentyEdgeMVC2 / twentyEdgeMVC)
    graphMeans.append((graphArr[5] + graphArr[6] + graphArr[7] + graphArr[8] + graphArr[9]) / 5)

    graphArr.append(oneEdgeMVC3 / oneEdgeMVC)
    graphArr.append(fiveEdgeMVC3 / fiveEdgeMVC)
    graphArr.append(tenEdgeMVC3 / tenEdgeMVC)
    graphArr.append(fifteenEdgeMVC3 / fifteenEdgeMVC)
    graphArr.append(twentyEdgeMVC3 / twentyEdgeMVC)
    graphMeans.append((graphArr[10] + graphArr[11] + graphArr[12] + graphArr[13] + graphArr[14]) / 5)

    
    print(graphArr)
    print(graphMeans)
    draw_plot2(graphArr, graphMeans, "Experiment 2")

    return True


def experiment_3():

    oneEdge = graphGen(100, 5, 14)
    fiveEdges = graphGen(100, 10, 15)
    tenEdges = graphGen(100, 15, 15)
    fifteenEdges = graphGen(100, 20, 15)


    # part b
    # need to find the mvc of all these graphs?
    oneEdgeMVCL = []
    fiveEdgeMVCL = []
    tenEdgeMVCL = []
    fifteenEdgeMVCL = []
    
    oneEdgeMVC = 0
    fiveEdgeMVC = 0
    tenEdgeMVC = 0
    fifteenEdgeMVC = 0

    oneEdgeMVC1 = 0
    fiveEdgeMVC1 = 0
    tenEdgeMVC1 = 0
    fifteenEdgeMVC1 = 0

    oneEdgeMVC2 = 0
    fiveEdgeMVC2 = 0
    tenEdgeMVC2 = 0
    fifteenEdgeMVC2 = 0

    oneEdgeMVC3 = 0
    fiveEdgeMVC3 = 0
    tenEdgeMVC3 = 0
    fifteenEdgeMVC3 = 0
    for i in range(0, len(oneEdge) - 1):
        print(i)
        oneEdgeMVCL.append(MVC(oneEdge[i]))
        oneEdgeMVC += len(oneEdgeMVCL[i])
        fiveEdgeMVCL.append(MVC(fiveEdges[i]))
        fiveEdgeMVC += len(fiveEdgeMVCL[i])
        tenEdgeMVCL.append(MVC(tenEdges[i]))
        tenEdgeMVC += len(tenEdgeMVCL[i])
        fifteenEdgeMVCL.append(MVC(fifteenEdges[i]))
        fifteenEdgeMVC += len(fifteenEdgeMVCL[i])

        oneEdgeMVC1 += len(mvc_1(oneEdge[i]))
        fiveEdgeMVC1 += len(mvc_1(fiveEdges[i]))
        tenEdgeMVC1 += len(mvc_1(tenEdges[i]))
        fifteenEdgeMVC1 += len(mvc_1(fifteenEdges[i]))

        oneEdgeMVC2 += len(mvc_2(oneEdge[i]))
        fiveEdgeMVC2 += len(mvc_2(fiveEdges[i]))
        tenEdgeMVC2 += len(mvc_2(tenEdges[i]))
        fifteenEdgeMVC2 += len(mvc_2(fifteenEdges[i]))

        oneEdgeMVC3 += len(mvc_3(oneEdge[i]))
        fiveEdgeMVC3 += len(mvc_3(fiveEdges[i]))
        tenEdgeMVC3 += len(mvc_3(tenEdges[i]))
        fifteenEdgeMVC3 += len(mvc_3(fifteenEdges[i]))

    graphArr = []
    graphMeans = []
    graphArr.append(oneEdgeMVC1 / oneEdgeMVC)
    graphArr.append(fiveEdgeMVC1 / fiveEdgeMVC)
    graphArr.append(tenEdgeMVC1 / tenEdgeMVC)
    graphArr.append(fifteenEdgeMVC1 / fifteenEdgeMVC)
    graphMeans.append((graphArr[0] + graphArr[1] + graphArr[2] + graphArr[3]) / 4)

    graphArr.append(oneEdgeMVC2 / oneEdgeMVC)
    graphArr.append(fiveEdgeMVC2 / fiveEdgeMVC)
    graphArr.append(tenEdgeMVC2 / tenEdgeMVC)
    graphArr.append(fifteenEdgeMVC2 / fifteenEdgeMVC)
    graphMeans.append((graphArr[4] + graphArr[5] + graphArr[6] + graphArr[7]) / 4)

    graphArr.append(oneEdgeMVC3 / oneEdgeMVC)
    graphArr.append(fiveEdgeMVC3 / fiveEdgeMVC)
    graphArr.append(tenEdgeMVC3 / tenEdgeMVC)
    graphArr.append(fifteenEdgeMVC3 / fifteenEdgeMVC)
    graphMeans.append((graphArr[8] + graphArr[9] + graphArr[10] + graphArr[11]) / 4)

    
    print(graphArr)
    print(graphMeans)
    draw_plot2(graphArr, graphMeans, "Experiment 3")

    return True

# Please feel free to include other experiments that support your answers. 
# Or any other experiments missing 
oneEdge = graphGen(100, 6, 10)
newL = []
for i in range(0, 100):
        newL.append(MVC(oneEdge[i]))

#experiment_1(25, 100, 10, 10)
#experiment_2()
experiment_3()