
# pylint: disable-all

from enum import Enum
import queue
#import os
import pathlib
import csv
import numpy


import networkx as nx
import matplotlib.pyplot as plt


class Vertex(object):
    '''Representing a single vertex of a graph'''

    def __init__(self, key):

        self.__key = key
        self.__data_neighbours = {}
        self.__control_neighbours = {}
        self.__depth = 0 #represents how far from root

    def __str__(self):
        return 'Key: {}\nData Neighbors: {}\nControl Neighbours:{}'.format(
            self.__key,
            [i for i in self.__data_neighbours.keys()],
            [i for i in self.__control_neighbours.keys()]
        )
    
    # getters and setters. ONLY use these functions

    def get_data_connections(self):
        return [i for i in self.__data_neighbours.values()]
    
    def get_control_connections(self):
        return [i for i in self.__control_neighbours.values()]
    
    def get_connections(self):
        arr = list(self.get_data_connections())
        c = list(self.get_control_connections())
        
        for conn in c:
            if conn not in arr:
                arr.append(conn) 

        return arr # get all keys from all dicts

    def get_key(self):
        return self.__key
    
    def set_depth(self,depth_value):
        self.__depth = depth_value
        
    def get_depth(self):
        return self.__depth

    def set_data_connection(self,neighbour,weight=0): # neighbour = vertex() object, NOT key
        self.__data_neighbours[neighbour.get_key()] = neighbour #format is as {key:vertex(),key:vertex() etc}

    def set_control_connection(self,neighbour,weight=0):
        self.__control_neighbours[neighbour.get_key()] = neighbour

class actionType(Enum):
    noType = 0
    control = 1
    data = 2


class Graph(object):
    def __init__(self):
        self.__vertices = {}

    def __add_vertex(self, vertex):
        self.__vertices[vertex.get_key()] = vertex

    def get_vertex(self, key):
        try:
            return self.__vertices[key]
        except KeyError:
            return None

    def __contains__(self, key):
        if key is int:
            return key in self.__vertices.keys()
        if key is Vertex:
            return key in self.__vertices.values()
        return TypeError("Wrong type called in __contains__()")

    def __add_edge_type(self,edge_type,from_key,to_key,weight=0):
        if from_key not in self.__vertices:
            self.__add_vertex(Vertex(from_key))
        if to_key not in self.__vertices:
            self.__add_vertex(Vertex(to_key))
        
        if (edge_type == actionType.data):
            self.__vertices[from_key].set_data_connection(self.__vertices[to_key], weight)
        elif (edge_type == actionType.control):
            self.__vertices[from_key].set_control_connection(self.__vertices[to_key], weight)
        else:
            raise Exception("No action type chosen.")
        
    def add_data_edge(self,from_key,to_key,weight=0):
        self.__add_edge_type(actionType.data,from_key,to_key,weight)

    def add_control_edge(self,from_key,to_key,weight=0):
        self.__add_edge_type(actionType.control,from_key,to_key,weight)    

    def get_vertices(self):
        return list(self.__vertices.values())

    def __iter__(self):
        return iter(self.__vertices.values())
    
    def __str__(self):
        return '{}'.format(
            [str(vertex) for vertex in self.__vertices]
        )

    

class BehaviourTree(Graph):
    def __init__(self):
        self.__root_vertex = None
        self.__networkx_graph = None
        super().__init__()

    def get_graph_name(self):
        return NotImplementedError("You must create a new get_graph_name function when creating a new graph object")
    
    def set_root_vertex_by_key(self,vertex_key): #set vertex by key
        if (self.get_vertex(vertex_key) is None):
            raise Exception("Root vertex does not exist")    
        self.__root_vertex = vertex_key

    def get_root_vertex(self):
        return self.get_vertex(self.__root_vertex)
    
    def initialise_depth(self):
        root = self.get_root_vertex()
        if (root is None):
            raise Exception("Must initialise root vertex for BehaviourTree with initialise_root_vertex(vertex)")
        
        
        #breadth first search to find distance from root vertex
        visited = set()
        q = queue.Queue()
        q.put(root)

        while (not q.empty()):
            curr = q.get()
            current_vertex_depth = curr.get_depth()
            connected_vertices = curr.get_connections()
           
            for neighbour in connected_vertices:
                if neighbour.get_key() not in visited:
                    neighbour.set_depth(current_vertex_depth+1)
                    visited.add(neighbour)
                    q.put(neighbour)
    
    def create_unformatted_networkx_graph(self):
        if (self.__networkx_graph is not None):
            return self.__networkx_graph
        g = nx.DiGraph()
        vertices = self.get_vertices()
        for vertex in vertices:
            for neighbour in vertex.get_connections():
                g.add_edge(vertex.get_key(),neighbour.get_key())
        self.__networkx_graph = g
        return g
    
    def save_graph_to_file(self,parentFolder,fileName): # Do not include file extension?
        pF = pathlib.Path(parentFolder)
        parentDirectoryExists = pF.is_dir()
        if (parentDirectoryExists is False):
            print("Parent folder does not exist")
            return False
        
        if ( '.dat' not in fileName):
            fileName += '.dat'
            
        fullPath = parentFolder.joinpath(fileName) 
        fileAlreadyExists = fullPath.is_file()

        with open(fullPath.resolve(),'w') as file:
            writer = csv.writer(file,delimiter=',',lineterminator = '\n')
            writer.writerow(["from_vertex","to_vertex","vertex_type"])
            for vertex in self.get_vertices():
                for data_neighbour in vertex.get_data_connections():
                    writer.writerow([vertex.get_key(),data_neighbour.get_key(),actionType.data.value])
                for control_neighbour in vertex.get_control_connections():
                    writer.writerow([vertex.get_key(),control_neighbour.get_key(),actionType.control.value])
        
        return True
    
    def load_file_to_graph(self,filePath):

        if ( '.dat' not in filePath):
            filePath += '.dat'
        fP = pathlib.Path(filePath)
        fileAlreadyExists = fP.is_file()
        if (not fileAlreadyExists):
            raise FileNotFoundError("Could not find file to read to graph")
        with open(fP.resolve(),'r') as file:
            csvreader = csv.reader(file,delimiter=',',lineterminator = '\n')
            header = next(csvreader)
            for row in csvreader:
                from_key = int(row[0])
                to_key = int(row[1])
                action_type = actionType(int(row[2]))
                

                if (action_type == actionType.control):
                    self.add_control_edge(from_key,to_key)
                if (action_type == actionType.data):
                    self.add_data_edge(from_key,to_key)
        
    def get_position(self):
        raise NotImplementedError("Do not call traverse by BehaviourTree, traverse must be implemented by a child class")
    
        # Set margins for the axes so that nodes aren't clipped
        

    
class Circular(BehaviourTree):
    def get_position(self):
        return nx.circular_layout(self.create_unformatted_networkx_graph())

class Random(BehaviourTree):
    def get_position(self):
        return nx.random_layout(self.create_unformatted_networkx_graph())
    

    
    
class SimpleRulesV1(BehaviourTree):
    positions = {}

    def iterate_through_node(self,curr_node,current_x,current_y): #crx = curr position of node # returns rightmost x

        self.positions[curr_node.get_key()] = (current_x,current_y)
        currControlConnections = curr_node.get_control_connections()
        currDataConnections = curr_node.get_data_connections()

        child_rightmost_x = current_x
        child_bottommost_y = current_y

        control_connection_passed = 1
        data_connection_passed = 0


        for connection in currControlConnections:

            if connection.get_depth() > curr_node.get_depth() and connection.get_key() not in self.positions:
                control_connection_passed = 0
                tmp = self.iterate_through_node(connection,child_rightmost_x,child_bottommost_y+1)
                child_rightmost_x  = tmp[0] + 1

            
        for connection in currDataConnections:
 
            if connection.get_depth() > curr_node.get_depth() and connection.get_key() not in self.positions:
                tmp = self.iterate_through_node(connection,child_rightmost_x + 1,child_bottommost_y)
                
                child_rightmost_x  = tmp[0]
        
        return (child_rightmost_x,child_bottommost_y)

    def get_position(self):
        self.set_root_vertex_by_key(0)
        self.initialise_depth()
        self.iterate_through_node(self.get_root_vertex(),0,0)        
        return self.positions




