

from collections import deque


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

testGraph = GraphII(10)
testGraph.add_edge(0, 1)
testGraph.add_edge(5, 9)
testGraph.add_edge(2, 3)
testGraph.add_edge(4, 8)
testGraph.add_edge(1, 6)
testGraph.add_edge(6, 7)
testGraph.add_edge(7, 3)
testGraph.add_edge(2, 9)
testGraph.add_edge(8, 9)






print(testGraph.get_graph())




def BFS_2(graph,src,dst):
    path = [] # stores the path of the graph
    visited = [] # list of all the visited vertices
    previous = [] # this will tell us what node got us to the given node, so the second node will tell us
    # what node got us to 
    queue = deque() # queue to store the order of traversal 
    graphNew = graph.get_graph()


    for node in graphNew: 
        previous.append(0) # not sure what to set here
        visited.append(False)
    
    for node in graphNew[src]:
        queue.appendleft(node)
        # while the queue isn't empty
    while queue:
        # we go through the first element of the queue and we:
        q = queue.pop()

        # add all other visitable nodes to the queue
        # we also add all whatever node we're currently on to the previous node
        for node in graphNew[q]:
            if not visited[node]:
                queue.append(node) # not certain if this works
                # we put the previous node here because
                previous[node] = q #put node number here, not sure if this works
                visited[node] = True
        
        visited[q] = True # When we;re done with a given node we say we've visited the node
    
    if visited[dst]: # if the destination has been visited, then we loop through the previous
        # array (which holds how we got the given array)
        temp = previous[dst]
        path.append(temp)
        while temp != src:
            print(temp)
            path.append(previous[temp])
            temp = previous[temp]
        for i in range(0, (len(path)) // 2):
            path[i], path[len(path) - i - 1] = path[len(path) - i - 1], path[i]
    else:
        print("Path not found")
    path.append(dst)
    return path
"""
def BFS_2(graph, src, destination):
    queue = deque([[src]])  # queue shows order of traversal
    visited = set()
    
    while queue:
        path = queue.popleft() # we go through every instance in queue and pop.left into path
        node = path[-1] #????
        
        if node == destination:
            return path  # Return the first path found
        
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get_graph()[node]:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    
    return []  # Return empty if no path found"""

#print(BFS_2(testGraph, 0, 9))


queue = deque()


queue.append(1)
queue.append(2)
queue.append(3)
queue.append(4)
queue.append(5)
queue.append(6)

l = queue.pop()
print(l)