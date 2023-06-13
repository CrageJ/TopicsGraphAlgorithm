
# pylint: disable-all

from enum import Enum
import queue
#import os
import pathlib
import csv
import numpy
import random
import statistics
import math
import copy
import array

import networkx as nx
import matplotlib.pyplot as plt
import shapely

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
        r = [i for i in self.__data_neighbours.values()]
        random.shuffle(r)
        return r
    
    def get_control_connections(self):
        r = [i for i in self.__control_neighbours.values()]
        random.shuffle(r)
        return r
    
    def get_connections(self):
        arr = list(self.get_data_connections())
        c = list(self.get_control_connections())
        
        for conn in c:
            if conn not in arr:
                arr.append(conn) 
        random.shuffle(arr)
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
        root.set_depth(0)
        q.put(root)

        while (not q.empty()):
            curr = q.get()
            current_vertex_depth = curr.get_depth()
            connected_vertices = curr.get_connections()
           
            for neighbour in connected_vertices:
                if neighbour.get_key() not in visited:
                    neighbour.set_depth(current_vertex_depth+1)
                    visited.add(neighbour.get_key())
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
        
    def get_positions(self):
        raise NotImplementedError("Do not call traverse by BehaviourTree, traverse must be implemented by a child class")
    
        # Set margins for the axes so that nodes aren't clipped



    #get lines of each 
def get_lines(graph,positions):
    #format = [((x1,y1),(x2,y2)),]
    lines = []
    for vertex in graph.get_vertices():
        for neighbour in vertex.get_connections():
            a = (positions[vertex.get_key()])
            b = (positions[neighbour.get_key()])
            coord = (a,b)
            
            lines.append( coord )

    return lines

#get cross over lines

def get_crossed_line_count(graph,positions):
    lines = get_lines(graph,positions)
    line_count = 0
    #test intesection with every line for every other line
    for left_index in range(0,len(lines)):
        left_line = lines[left_index]
        for right_index in range(left_index+1,len(lines)):
            right_line = lines[right_index]
            lines_intersect = doIntersect(left_line[0][0],left_line[0][1],left_line[1][0],left_line[1][1],\
                                right_line[0][0],right_line[0][1],right_line[1][0],right_line[1][1])
            if lines_intersect:
                print('a1({},{}) a2({},{})\nb1({},{}) b2({},{})'.format(left_line[0][0],left_line[0][1],left_line[1][0],left_line[1][1],\
                                right_line[0][0],right_line[0][1],right_line[1][0],right_line[1][1]))
                line_count += int(lines_intersect)

    return line_count

def get_line_lengths(graph,positions):
    lines = get_lines(graph,positions)
    line_lengths = []
    for line in lines:
        coord1 = line[0]
        coord2 = line[1]
        length = math.sqrt(((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2))
        line_lengths.append(length)

    return line_lengths

def get_line_length_stdev(graph,positions):
    line_lengths = get_line_lengths(graph,positions)
    stat = statistics.stdev(line_lengths)
    return stat

'''a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
np.dot(a, b)
array([[4, 1],
       [2, 2]])'''

# taken https://stackoverflow.com/questions/52516949/angle-between-two-non-intersecting-lines

#dot product
def get_angle(p1x,p1y,
                q1x,q1y,
                p2x,p2y,
                q2x,q2y):
    l1 = [(p1x,p1y), (q1x,q1y)]
    l2 = [(p2x, p2y), (q2x, q2y)]

    seg1 = numpy.array(l1)
    seg1 = seg1[1] - seg1[0]

    seg2 = numpy.array(l2)
    seg2 = seg2[1] - seg2[0]

    angle_l1 = numpy.angle(complex(*(seg1)),deg=False)
    angle_l2 = numpy.angle(complex(*(seg2)),deg=False)

    #result
    res = angle_l1 - angle_l2
    res = (res + numpy.pi/2) % numpy.pi - (numpy.pi/2)

    return res

# data lines
    #get lines of each 
def get_data_lines(graph,positions):
    #format = [((x1,y1),(x2,y2)),]
    lines = []
    for vertex in graph.get_vertices():
        for neighbour in vertex.get_data_connections():
            a = (positions[vertex.get_key()])
            b = (positions[neighbour.get_key()])
            coord = (a,b)
        
            lines.append( coord )

    return lines

def get_control_lines(graph,positions):
        #format = [((x1,y1),(x2,y2)),]
    lines = []
    for vertex in graph.get_vertices():
        for neighbour in vertex.get_control_connections():
            a = (positions[vertex.get_key()])
            b = (positions[neighbour.get_key()])
            coord = (a,b)
        
            lines.append( coord )

    return lines
    
    

def get_average_angles(graph,positions):
    lines = get_lines(graph,positions)

    angles = 0
    angles_count = 0

    # control = vertical
    for c in get_control_lines(graph,positions):
        current_angle = get_angle(c[0][0],c[0][1],
                                  c[1][0],c[1][1],
                                  0,0,
                                  0,1)
        angles += current_angle
        angles_count += 1

    #data = horizontal
    for d in get_data_lines(graph,positions):
        current_angle = get_angle(d[0][0],d[0][1],
                                  d[1][0],d[1][1],
                                  0,0,
                                  1,0)
        angles += current_angle
        angles_count += 1

    return angles/angles_count
    


    
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
            new_dict[vertex] = (x_scaling,y_v)
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
                #print('Node {} Processing {}'.format(curr_node.get_key(),connection.get_key()))
                control_connection_passed = 0
                tmp = self.iterate_through_node(connection,child_rightmost_x,child_bottommost_y+1)
                child_rightmost_x  = tmp[0] + 1

            
        for connection in currDataConnections:
 
            if connection.get_depth() > curr_node.get_depth() and connection.get_key() not in self.positions:
                tmp = self.iterate_through_node(connection,child_rightmost_x + 1,child_bottommost_y)
                #print('Node {} Processing {}'.format(curr_node.get_key(),connection.get_key()))
                child_rightmost_x  = tmp[0]
        
        return (child_rightmost_x,child_bottommost_y)
    def get_position(self):
        self.set_root_vertex_by_key(0)
        self.initialise_depth()
        root_v = self.get_root_vertex()
        self.iterate_through_node(root_v,0,0)        
        return flip(self.positions)


# The main function that returns true if 
# the line segment 'p1q1' and 'p2q2' intersect.
#lines go from (p#x,p#y) -> (q#x,q#y)
def doIntersect(p1x,p1y,
                q1x,q1y,
                p2x,p2y,
                q2x,q2y):
    p1 = shapely.LineString([shapely.Point(p1x,p1y),shapely.Point(q1x,q1y)])
    p2 = shapely.LineString([shapely.Point(p2x,p2y),shapely.Point(q2x,q2y)])
    intersects = (p1.intersects(p2))
    if (intersects == False):
        return False

    intersection =(p1.intersection(p2))
    if type(intersection) is shapely.LineString:
        print ("Linear Overlap (True)")
        return True
    if type(intersection) is shapely.Point:
        # if either of the points were on each others edges, it doesnt count
        cond1 = (p1x==p2x and p1y==p2y) # if p1 == q1 
        cond2 = (p1x==q2x and p1y==q2y)  # if p1 == q2
        cond3 = (q1x==p2x and q1y==p2y) # if q1 == p2
        cond4 = (q1x==q2x and q1y==q2y)  # if q1 == q2
    
        if cond1 or cond2 or cond3 or cond4:
            print ("Point Overlap (On edge of line segment) (False)")
            return False
        print ("Point Overlap (Not on edge of line segment) (True)")
        return True

    return ValueError("Error Processing Intersection")

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
                child_bottommost_y = tmp[1]

            
        lenCurrDataConnections = len(currDataConnections)
        if (lenCurrDataConnections == 1 and currDataConnections[0].get_key() not in self.positions):
            self.iterate_through_node(currDataConnections[0],child_rightmost_x+1,current_y,curr_node)
        else:
            
            for connection in currDataConnections:
                if connection.get_key() not in self.positions:
                    
                    tmp = self.iterate_through_node(connection,child_rightmost_x+1,child_bottommost_y,curr_node)     
                    child_rightmost_x = tmp[0] 
                    child_bottommost_y +=1 
                    #child_bottommost_y = tmp[1]
        return (child_rightmost_x,child_bottommost_y)

    def get_position(self):
        self.set_root_vertex_by_key(0)
        self.initialise_depth()
        root_v = self.get_root_vertex()
        self.iterate_through_node(root_v,0,0,root_v)        
        return flip(self.positions)


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


class LevelRankV1(BehaviourTree):
    def __init__(self):
        self.positions = {}
        self.levels = {}
        super().__init__()
    
    def get_levels(self):
        vertices = self.get_vertices()
        for vertex in vertices:
            arr = self.levels.get(vertex.get_depth(),[])
            
            arr.append(vertex)
            self.levels[vertex.get_depth()] = arr
    
    def order_levels(self):
        print (self.levels.items())
        for level, vertices in self.levels.items():
            vertex_count = 0
            for vertex in vertices:
                self.positions[vertex.get_key()] = (level,vertex_count)
                vertex_count += 1

    def get_position(self):
        self.set_root_vertex_by_key(0)
        self.initialise_depth()
        self.get_levels()
        self.order_levels()      
        return adjustByRow(flip(self.positions))
    


def generate_set_value(origin_vertex, dest_vertex):
    l = str(min(origin_vertex, dest_vertex))
    r = str(max(origin_vertex, dest_vertex))
    return (l + " " + r)

class LevelRankV2(BehaviourTree):
    def __init__(self):
        self.positions = {}
        self.levels = {}
        self.data_connection_set = set()
        self.control_connection_set = set()
        #self.lines = [] #in the format of [a,b] where vertex a is connected to b
        super().__init__()
    
    def get_levels(self):
        vertices = self.get_vertices()
        for vertex in vertices:
            curr_key = vertex.get_key()
            arr = self.levels.get(vertex.get_depth(),[])

            for data_vertex in vertex.get_data_connections():
                neighbour_key = data_vertex.get_key()
                #self.lines.append([curr_key,neighbour_key])
                self.data_connection_set.add(generate_set_value(curr_key,neighbour_key))

            for control_vertex in vertex.get_control_connections():
                neighbour_key = control_vertex.get_key()
                #self.lines.append([curr_key,neighbour_key])
                self.control_connection_set.add(generate_set_value(curr_key,neighbour_key))
                
            arr.append(vertex.get_key())
            self.levels[vertex.get_depth()] = arr
        
    

    #get the x and y coords of nodes from position list
    # current_level is key value
    def get_crossover_count(self,current_level,selected_level,
                            current_level_positions,selected_level_positions):
        
        crossover_count = 0
        lines = [] # stored as 
        for current_vertex in current_level:
            current_key = current_vertex

            current_position = current_level_positions[current_key]

            for selected_vertex in selected_level:
                selected_key = selected_vertex

                selected_position = selected_level_positions[selected_key]

                for line in lines:
                    #do intersect lines[0],lines[1],current,select
                    print("line", line)
                    print("curr", current_position[0],current_position[1],
                                                  selected_position[0],selected_position[1])

                    is_Intersecting = doIntersect(line[0][0],line[0][1],
                                                  line[1][0],line[1][1],
                                                  current_position[0],current_position[1],
                                                  selected_position[0],selected_position[1])
                    if is_Intersecting:
                        crossover_count += 1
                
                lines.append([[current_position[0],current_position[1]],[selected_position[0],selected_position[1]]])

        return crossover_count
    
    #RANDOM ITERATIONS
    random_iterations = 5 # can be modified as fit

    #counts the cross over of lines between current one and selected row 
    # current level = array NOT index
    # selected level = likewise
        
    
    def order_levels(self):
        print (self.levels.items())
        level_items = self.levels.items()
        for level, vertices in self.levels.items():
            vertex_count = 0
            for vertex in vertices:
                self.positions[vertex] = (level,vertex_count)
                vertex_count += 1

        first_level = True
        for level, vertices in level_items:
            # skip shuffling first layer 
            if first_level:
                first_level = False
                continue

            crossover_count_min = 999
            current_layout = copy.deepcopy(vertices)
            current_dict = {}
            for i in range(0,5):
                new_layout = copy.deepcopy(vertices)
                random.shuffle(new_layout)
                # create a new dictionary for this level
                new_dict = {}
                new_vertex_count = 0
                for n in new_layout:
                    new_dict[n] = (level,new_vertex_count)
                    new_vertex_count += 1

                    # get crosovers of this new line and previous line
                new_arrangement_crossovers = self.get_crossover_count(new_layout,self.levels[level-1],
                                                                      new_dict,self.positions)
                # if this arrangement casuses the fewest crossovers
                if new_arrangement_crossovers < crossover_count_min:
                    crossover_count_min = new_arrangement_crossovers
                    current_layout = new_layout
                    current_dict = new_dict

            # set appropriate variables for found localmin
            #change dict
            for k,v in current_dict.items():
                self.positions[k] = v

            # modify levels placement
            self.levels[level-1] = current_layout

        

    def get_position(self):
        self.set_root_vertex_by_key(0)
        self.initialise_depth()
        self.get_levels()
        self.order_levels()      
        return flip(self.positions)
    

class CombinedRankV1(BehaviourTree):
    def __init__(self):
        self.positions = {}
        self.deepest_y_value = 0
        self.deepest_x_value = 0

        ##store line draw info here
        ##each entry is a (p1,p2) vertex pair representing line segment
        self.line_segments = []

        self.vertex_to_pos = {}
        
        self.levels = {}
        self.data_connection_set = set()
        self.control_connection_set = set()
        #self.lines = [] #in the format of [a,b] where vertex a is connected to b
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

        cc = 0

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
            tmp = self.iterate_through_node(currDataConnections[0],child_rightmost_x+1,current_y,curr_node)
        else:
            for connection in currDataConnections:
                if connection.get_key() not in self.positions:
                    tmp = self.iterate_through_node(connection,child_rightmost_x+1,current_y,curr_node)    
                    child_rightmost_x = tmp[0]
                    cc += 1

        self.deepest_y_value =  max(self.deepest_y_value,child_bottommost_y)

        return (child_rightmost_x,child_bottommost_y)

    def get_levels(self):

        for vertex in self.get_vertices():
            curr_key = vertex.get_key()
            arr = self.levels.get(self.positions[curr_key][1],[])
            arr.append(curr_key)
            self.levels[self.positions[curr_key][1]] = arr

        for i in range(len(self.levels)):
            row = self.levels[i]
            row.sort(key=lambda x:self.positions[x][0],reverse=False)
            cc = 0
            for vertex in row:
                print(vertex)
                self.positions[vertex] = (self.positions[vertex][0],cc)
                cc += 1

    #get the x and y coords of nodes from position list
    # current_level is key value
    def get_crossover_count(self,current_level,selected_level,
                            current_level_positions,selected_level_positions):
        
        crossover_count = 0
        lines = [] # stored as 
        for current_vertex in current_level:
            current_key = current_vertex

            current_position = current_level_positions[current_key]

            for selected_vertex in selected_level:
                selected_key = selected_vertex

                selected_position = selected_level_positions[selected_key]

                for line in lines:
                    #do intersect lines[0],lines[1],current,select
                    print("line", line)
                    print("curr", current_position[0],current_position[1],
                                                  selected_position[0],selected_position[1])

                    is_Intersecting = doIntersect(line[0][0],line[0][1],
                                                  line[1][0],line[1][1],
                                                  current_position[0],current_position[1],
                                                  selected_position[0],selected_position[1])
                    if is_Intersecting:
                        crossover_count += 1
                
                lines.append([[current_position[0],current_position[1]],[selected_position[0],selected_position[1]]])

        return crossover_count
    
    #RANDOM ITERATIONS
    random_iterations = 5 # can be modified as fit

    #counts the cross over of lines between current one and selected row 
    # current level = array NOT index
    # selected level = likewise
        
    
    def order_levels(self):
        print (self.levels.items())
        level_items = self.levels.items()
        for level, vertices in self.levels.items():
            vertex_count = 0
            for vertex in vertices:
                self.positions[vertex] = (level,vertex_count)
                vertex_count += 1

        first_level = True
        for level, vertices in level_items:
            # skip shuffling first layer 
            if first_level:
                first_level = False
                continue

            crossover_count_min = 999
            current_layout = copy.deepcopy(vertices)
            current_dict = {}
            for i in range(0,5):
                new_layout = copy.deepcopy(vertices)
                random.shuffle(new_layout)
                # create a new dictionary for this level
                new_dict = {}
                new_vertex_count = 0
                for n in new_layout:
                    new_dict[n] = (level,new_vertex_count)
                    new_vertex_count += 1

                    # get crosovers of this new line and previous line
                new_arrangement_crossovers = self.get_crossover_count(new_layout,self.levels[level-1],
                                                                      new_dict,self.positions)
                # if this arrangement casuses the fewest crossovers
                if new_arrangement_crossovers < crossover_count_min:
                    crossover_count_min = new_arrangement_crossovers
                    current_layout = new_layout
                    current_dict = new_dict

            # set appropriate variables for found localmin
            #change dict
            for k,v in current_dict.items():
                self.positions[k] = v

            # modify levels placement
            self.levels[level-1] = current_layout
        

    def get_position(self):
        self.set_root_vertex_by_key(0)
        self.initialise_depth()
        self.iterate_through_node(self.get_root_vertex(),0,0,self.get_root_vertex())        
        self.get_levels()
        self.order_levels()      
        return adjustByRow(flip(self.positions))




class TestGraphIterator(BehaviourTree):
    def __init__(self):
        self.deepest_y_value = 0
        self.deepest_x_value = 0

        ##store line draw info here
        ##each entry is a (p1,p2) vertex pair representing line segment
        self.line_segments = []

        self.vertex_count = 0
        self.position_list = []
        super().__init__()


    def iterate_through_node(self,curr_node,current_x,current_y,previous_node,p): 
        positions = copy.copy(p)
        positions[curr_node.get_key()] = (current_x,current_y)
        if (len(positions) >= self.vertex_count):
            self.position_list.append(positions)
        curr_line_segment = (positions[previous_node.get_key()],positions[curr_node.get_key()])
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

            if connection.get_key() not in positions:
                #print('Node {} Processing {}'.format(curr_node.get_key(),connection.get_key()))
                control_connection_passed = 0
                tmp = self.iterate_through_node(connection,child_rightmost_x,current_y+1,curr_node,positions)
                child_rightmost_x  = tmp[0] + 1

            
        lenCurrDataConnections = len(currDataConnections)
        if (lenCurrDataConnections == 1 and currDataConnections[0].get_key() not in positions):
            self.iterate_through_node(currDataConnections[0],child_rightmost_x+1,current_y,curr_node,positions)
        else:
            child_bottommost_y += 1
            for connection in currDataConnections:
                if connection.get_key() not in positions:
                    self.iterate_through_node(connection,child_rightmost_x+1,current_y+1,curr_node,positions)      
        return (child_rightmost_x,child_bottommost_y)

    def get_position(self):
        self.vertex_count = len(self.get_vertices())
        self.set_root_vertex_by_key(0)
        self.initialise_depth()
        root_v = self.get_root_vertex()
        self.iterate_through_node(root_v,0,0,root_v,{})        
        return self.position_list
    






