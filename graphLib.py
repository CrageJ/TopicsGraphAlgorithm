
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

class FormattedParams():
        def __init__(self,graph,color):
            self.graph = graph
            self.color = color

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
    

    def create_formatted_networkx_graph(self):
        g = nx.DiGraph()
        vertices = self.get_vertices()
        for vertex in vertices:
            for neighbour in vertex.get_control_connections():
                g.add_edge(vertex.get_key(),neighbour.get_key(),color = 'b')
            for neighbour in vertex.get_data_connections():
                g.add_edge(vertex.get_key(),neighbour.get_key(),color = 'g')
        colors = list(nx.get_edge_attributes(g,'color').values())
        
        ret = FormattedParams(g,colors)
        return ret
    
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
    


def flip(position_dict):
    max_x = 0
    max_y = 0

    for v in position_dict.values():
        max_x = max(max_x,v[0])
        max_y = max(max_y,v[1])
    
    for p in position_dict:
        prev_val = position_dict[p]
        position_dict[p] = (prev_val[0],max_y-prev_val[1])

    return position_dict
    


def adjustByRow(position_dict):

    max_x = 0
    max_y = 0

    for v in position_dict.values():
        max_x = max(max_x,v[0])
        max_y = max(max_y,v[1])

    #values list
    values_list = {}
    

    for key in position_dict:
        v = position_dict[key]
        y_v = v[1]
        x_v = v[0]


        #sort within 
        values_list[y_v] = values_list.get(y_v,[])

        index = 0
        #sort within values list, place item in correct spot
        for item in values_list[y_v]:
            if position_dict[item][0] < x_v:
                break

            index += 1

        values_list[y_v].insert(index,key)

    new_dict = {}

    for y_v in values_list:
        current_list = values_list[y_v]

        index = 0
        for vertex in current_list:
            x_scaling = float(1/(1 + len(values_list[y_v])))*(1+index)*max_x
            new_dict[vertex] = (y_v,x_scaling)
            index += 1

    return new_dict

    

    
    
class SimpleRulesV1(BehaviourTree):
    def __init__(self):
        self.positions = {}
        self.deepest_y_value = 0
        self.deepest_x_value = 0
        super().__init__()


    def iterate_through_node(self,curr_node,current_x,current_y): #crx = curr position of node # returns rightmost x
        self.positions[curr_node.get_key()] = (current_x,current_y)
        self.deepest_y_value = max(self.deepest_y_value,current_y)
        self.deepest_x_value = max(self.deepest_x_value,current_x)

        currControlConnections = curr_node.get_control_connections()
        currDataConnections = curr_node.get_data_connections()

        child_rightmost_x = current_x
        child_bottommost_y = current_y

        control_connection_passed = 1
        data_connection_passed = 0


        for connection in currControlConnections:

            if connection.get_depth() > curr_node.get_depth() and connection.get_key() not in self.positions:
                print('Node {} Processing {}'.format(curr_node.get_key(),connection.get_key()))
                control_connection_passed = 0
                tmp = self.iterate_through_node(connection,child_rightmost_x,child_bottommost_y+1)
                child_rightmost_x  = tmp[0] + 1

            
        for connection in currDataConnections:
 
            if connection.get_depth() > curr_node.get_depth() and connection.get_key() not in self.positions:
                tmp = self.iterate_through_node(connection,child_rightmost_x + 1,child_bottommost_y)
                print('Node {} Processing {}'.format(curr_node.get_key(),connection.get_key()))
                child_rightmost_x  = tmp[0]
        
        return (child_rightmost_x,child_bottommost_y)

    def get_position(self):
        self.set_root_vertex_by_key(0)
        self.initialise_depth()
        self.iterate_through_node(self.get_root_vertex(),0,0)   
        return flip(adjustByRow(self.positions))
    
def onSegment(px,py, qx,qy, rx,ry):
    if ( (qx < max(px, rx)) and (qx > min(px, rx)) and 
           (qy < max(py, ry)) and (qy > min(py, ry))):
        return True
    return False
  
def orientation(px,py, 
                qx,qy, 
                rx,ry):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ 
    # for details of below formula. 
      
    val = ((float(qy) - float(py)) * (float(rx) - float(qx))) - ((float(qx) - float(px)) * (float(ry) - float(qy)))
    if (val > 0):
          
        # Clockwise orientation
        return 1
    elif (val < 0):
          
        # Counterclockwise orientation
        return 2
    else:
          
        # Collinear orientation
        return 0
  
# The main function that returns true if 
# the line segment 'p1q1' and 'p2q2' intersect.
#lines go from (p#x,p#y) -> (q#x,q#y)
def doIntersect(p1x,p1y,
                q1x,q1y,
                p2x,p2y,
                q2x,q2y):
      
    # Find the 4 orientations required for 
    # the general and special cases
    o1 = orientation(p1x,p1y, q1x,q1y, p2x,p2y)
    o2 = orientation(p1x,p1y, q1x,q1y, q2x,q2y)
    o3 = orientation(p2x,p2y, q2x,q2y, p1x,p1y)
    o4 = orientation(p2x,p2y, q2x,q2y, q1x,q1y)
  
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
  
    # Special Cases
  
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1x,p1y, p2x,p2y,q1x,q1y)):
        return True
  
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1x,p1y, q2x,q2y,q1x,q1y)):
       return True
  
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and  onSegment(p2x,p2y,p1x,p1y,q2x,q2y)):
        return True
  
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2x,p2y,q1x,q1y,q2x,q2y)):
        return True
  
    # If none of the cases
    return False


class SimpleRulesV2(BehaviourTree):
    def __init__(self):
        self.positions = {}
        self.deepest_y_value = 0
        self.deepest_x_value = 0

        ##store line draw info here
        ##each entry is a (p1,p2) vertex pair representing line segment
        self.line_segments = []
        super().__init__()


    def iterate_through_node(self,curr_node,current_x,current_y,previous_node): #crx = curr position of node # returns rightmost x
        self.positions[curr_node.get_key()] = (current_x,current_y)
        curr_line_segment = (self.positions[previous_node.get_key()],self.positions[curr_node.get_key()])
        self.line_segments.append(curr_line_segment)
        self.deepest_y_value = max(self.deepest_y_value,current_y)
        self.deepest_x_value = max(self.deepest_x_value,current_x)

        currControlConnections = curr_node.get_control_connections()
        currDataConnections = curr_node.get_data_connections()

        child_rightmost_x = current_x
        child_bottommost_y = current_y

        control_connection_passed = 1
        data_connection_passed = 0


        for connection in currControlConnections:

            if connection.get_key() not in self.positions:
                #print('Node {} Processing {}'.format(curr_node.get_key(),connection.get_key()))
                control_connection_passed = 0
                tmp = self.iterate_through_node(connection,child_rightmost_x,current_y+1,curr_node)
                child_rightmost_x  = tmp[0] + 1

            
        lenCurrDataConnections = len(currDataConnections)
        if (lenCurrDataConnections == 1 and currDataConnections[0].get_key() not in self.positions):
            self.iterate_through_node(currDataConnections[0],child_rightmost_x+1,current_y,curr_node)
        else:
            child_bottommost_y += 1
            for connection in currDataConnections:
                if connection.get_key() not in self.positions:
                    self.iterate_through_node(connection,child_rightmost_x+1,current_y+1,curr_node)      
        return (child_rightmost_x,child_bottommost_y)

    def get_position(self):
        self.set_root_vertex_by_key(0)
        self.initialise_depth()
        root_v = self.get_root_vertex()
        self.iterate_through_node(root_v,0,0,root_v)        
        return flip(adjustByRow(self.positions))


class SimpleRulesV3(BehaviourTree):
    def __init__(self):
        self.positions = {}
        self.deepest_y_value = 0
        self.deepest_x_value = 0

        ##store line draw info here
        ##each entry is a (p1,p2) vertex pair representing line segment
        self.line_segments = []
        super().__init__()


    def iterate_through_node(self,curr_node,current_x,current_y,previous_node): #crx = curr position of node # returns rightmost x
        self.positions[curr_node.get_key()] = (current_x,current_y)
        curr_line_segment = (self.positions[previous_node.get_key()],self.positions[curr_node.get_key()])
        self.line_segments.append(curr_line_segment)
        self.deepest_y_value = max(self.deepest_y_value,current_y)
        self.deepest_x_value = max(self.deepest_x_value,current_x)

        currControlConnections = curr_node.get_control_connections()
        currDataConnections = curr_node.get_data_connections()

        child_rightmost_x = current_x
        child_bottommost_y = current_y

        control_connection_passed = 1
        data_connection_passed = 0


        for connection in currControlConnections:

            if connection.get_key() not in self.positions:
                #print('Node {} Processing {}'.format(curr_node.get_key(),connection.get_key()))
                control_connection_passed = 0
                tmp = self.iterate_through_node(connection,child_rightmost_x,current_y+1,curr_node)
                child_rightmost_x  = tmp[0] + 1

            
        for connection in currDataConnections:
 
            if connection.get_key() not in self.positions:

                does_intersect = False
                for line in self.line_segments:
                    if (line == (curr_line_segment)): #if current line, skip
                        continue
                    if doIntersect(child_rightmost_x+1,current_y,
                        current_x,current_y,
                        line[0][0],line[0][1],
                        line[1][0],line[1][1]): #thisnode/currentnode x previouslines
                        print('do_i:',child_rightmost_x+1,current_y,
                        current_x,current_y,
                        line[0][0],line[0][1],
                        line[1][0],line[1][1])
                        does_intersect = True
                        break
                    
                tmp = None

                if (does_intersect == True):
                    child_bottommost_y += 1
                    tmp = self.iterate_through_node(connection,child_rightmost_x + 1,child_bottommost_y,curr_node)
                else:
                    tmp = self.iterate_through_node(connection,child_rightmost_x + 1,current_y,curr_node)

                    #print('Node {} Processing {}'.format(curr_node.get_key(),connection.get_key()))
                    child_rightmost_x  = tmp[0]
        
        return (child_rightmost_x,child_bottommost_y)

    def get_position(self):
        self.set_root_vertex_by_key(0)
        self.initialise_depth()
        root_v = self.get_root_vertex()
        self.iterate_through_node(root_v,0,0,root_v)        
        return flip(adjustByRow(self.positions))



        










