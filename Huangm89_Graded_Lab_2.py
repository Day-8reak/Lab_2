

from collections import deque
import random
import time
import timeit 
import matplotlib.pyplot as plt
import numpy as np

# Utility functions - some are implemented, others you must implement yourself.

# function to plot the bar graph and average runtimes of N trials
# Please note that this function only plots the graph and does not save it
# To save the graphs you must use plot.save(). Refer to matplotlib documentation
def draw_plot(run_arr, mean):
    x = np.arange(0, len(run_arr),1)
    fig=plt.figure(figsize=(20,8))
    plt.axhline(mean,color="red",linestyle="--",label="Avg")
    plt.xlabel("Iterations")
    plt.ylabel("Run time in ms order of 1e-6")
    plt.title("Run time for retrieval")
    plt.show()

# function to generate random graphs 
# @args : nodes = number of nodes 
#       : edges = number of edges
def create_random_graph(nodes, edges):
    graph = None

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
        # your implementation for Part 3 goes here
        return False
    
    def is_connected(self,node1,node2):
        # your implementation for Part 3 goes here
        return False
    
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
        # your implementation for Part 3 goes here
        # we simply use DFS and check for if we can ever run into a previously node 
        return False
    
    def is_connected(self,node1,node2):
        # your implementation for Part 3 goes here
        return False
    
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
            print(node)

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


        print(graphNew[q])
        for node in graphNew[q]: # otherwise, we loop through the given adjacency list
            # until we find the next node to visit
            print(node)

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
    for start in G.adj:
        for end in G.adj[start]:
            if not(start in C or end in C):
                return False
    return True

def MVC(G):
    nodes = [i for i in range(G.get_size())]
    subsets = power_set(nodes)
    min_cover = nodes
    for subset in subsets:
        if is_vertex_cover(G, subset):
            if len(subset) < len(min_cover):
                min_cover = subset
    return min_cover

def mvc_1(G):
    # Your implementation for part 6.a goes here
    min_cover=None

    return min_cover

def mvc_2(G):
    # Your implementation for part 6.b goes here
    min_cover=None

    return min_cover

def mvc_3(G):
    # Your implementation for part 6.c goes here
    min_cover=None

    return min_cover

def experiment_1():

    # your implementation for experiment in part 5 goes here
    return True

def experiment_2():

    # your implementation for experiment in part 7 goes here
    return True

def experiment_3():

    # your implementation for any other 
    # supplemental experiments you need to run goes here (e.g For question 7.c)
    return True

# Please feel free to include other experiments that support your answers. 
# Or any other experiments missing 