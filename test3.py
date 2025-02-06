from collections import deque
"""
queue = deque()
queue.append(1)
queue.append(2)
queue.append(3)
queue.append(4)
queue.append(5)
queue.append(6)
queue.append(7)
queue.append(8)
queue.append(9)
queue.append(10)

q = queue.pop()
print(type(q))
print(type("hello"))
print(type(queue))
"""


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




"""
testNewGraph = testGraph.get_graph()
print(type(testNewGraph[0]))
print(testNewGraph[0])
print(type(testGraph))

for node in testNewGraph[0]:
    print("Hello")
"""



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



#print(DFS_2(testGraph, 0, 4))

print(DFS_2(testGraph, 0, 2)