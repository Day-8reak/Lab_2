from collections import deque
import random


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
                print("We are currently checking node " + str(node))
                print("We are currently on node " + str(q))

                if visited[node] and previous[q] != node: # if the node we're look at has been visited we skip it
                    return True
                elif visited[node]:
                    continue
                else: # otherwise we append it to our queue
                    queue.append(node)
                    previous[node] = q
            print("for exited")
            
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

testGraph = GraphII(10)
testGraph.add_edge(0, 1)
#testGraph.add_edge(5, 9)
testGraph.add_edge(2, 3)
testGraph.add_edge(4, 8)
testGraph.add_edge(1, 6)
testGraph.add_edge(6, 7)
testGraph.add_edge(7, 3)
testGraph.add_edge(2, 9)
testGraph.add_edge(8, 9)
testGraph.add_edge(0, 7)


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

l = create_random_graph(10, 1)
print(l.get_graph())
print(len(MVC(l)))