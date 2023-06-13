# pylint: disable-all
import graphLib as g
import argparse 
import pathlib
import copy
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualise graphs using multiple graph display algorithms')

    algorithms = [ #must be the same name as the object name
        ('SimpleRulesV1','Output simple rule-based graphing algorithm'),
        ('Circular','Output circular graph'),
        ('Random', 'Output randomised graph'),
        ('SimpleRulesV2','Output V2 of simple rules'),
        ('SimpleRulesV3','Output V3 of simple rules'),
        ('LevelRankV1','Output V1 of level rank'),
        ('LevelRankV2','Output V2 of level rank'),
        ('CombinedRankV1','Output V2 of level rank'),
    ]

    parser.add_argument('inputLocation',type=str,
                       help='Absolute location of the folder to load examples')
    parser.add_argument('outputLocation',type=str,
                       help='Absolute location of the folder to save output image')

    algorithm_objects = {}
    algorithm_switches = []
    algorithm_count = 0
    algorithm_references = []

    algorithm_index = 0
    for algorithm in algorithms:
        algorithm_name = algorithms[algorithm_index][0]
        algorithm_description = algorithms[algorithm_index][1]
        #get object of algorithm name
        exec("algorithm_objects[{}] = g.{}()".format(algorithm_index,algorithm_name),locals())
        #add as argument to parser
        exec("parser.add_argument('--'+'{}', help='{}',action='store_true',default=False)".format(algorithm_name,algorithm_description),locals())
        algorithm_index += 1

    
    
    args = parser.parse_args()


    print(args)
    algorithm_references=[]

    #count number of args passed
    algorithm_iterator = 0
    algorithm_count = 0
    for algorithm in algorithms:
        is_algorithm_switch_passed = int(eval("args.{}".format(algorithm[0])))
        if (is_algorithm_switch_passed):
            algorithm_references.append((algorithm[0],algorithm_objects[algorithm_iterator])) # (name, object)
            algorithm_count += 1
        algorithm_iterator += 1

        

    folder_location = pathlib.Path(args.inputLocation)
    out_location = pathlib.Path(args.outputLocation)

    data_files = folder_location.glob('*.dat')
    file_count=0

    file_dirs = []

    for file in data_files:
        file_dirs.append(file) 
        file_count += 1

    print ("FileCount:{} AlgoCount:{} FILES:{}".format(file_count,algorithm_count,file_dirs))


    plot_count = 0

    plt.autoscale(tight=True)

    for file_number in range(file_count):
        file_dir = file_dirs[file_number]
        file_name = str(file_dir.stem)
        file_dir = str(file_dir.resolve())

        for algorithm_number in range(algorithm_count):
            plot_count += 1
            ax = plt.subplot(file_count,algorithm_count,plot_count)
            algorithm_name = algorithm_references[algorithm_number][0]
            algorithm_object = copy.deepcopy(algorithm_references[algorithm_number][1])
            positions = {}
            
            try:
                algorithm_object.load_file_to_graph(file_dir)
                positions = algorithm_object.get_position()
                formatted_graph = algorithm_object.create_formatted_networkx_graph()
                graph_colors = formatted_graph.color
                graph = formatted_graph.graph
                options = {
                    "font_size": 10,
                    "node_size": 100,
                    "node_color": "#D3D3D3",
                    "linewidths": 2,
                    "width": 2,
                    "with_labels" : True,
                    "edge_color" : graph_colors
                }
                #print(positions)
                a = g.get_crossed_line_count(algorithm_object,positions)
                b = g.get_line_length_stdev(algorithm_object,positions)
                c = g.get_average_angles(algorithm_object,positions)
                print('algorithm_name: {} crossed_line: {} stddev line length : {} average_angle: {}'.format(algorithm_name,a,b,c))
                # HEURISTIC VALUE CONSTS
                aa = 0.2
                bb = 0.4
                cc = 0.4
                print('HEURISTIC VALUE WITH ',a,b,c)
                print (aa*a + bb*b + cc*c)


                plt.title(file_name+'//'+algorithm_name)
                nx.draw(graph, pos=positions,ax=plt.gca(),**options)
            except BaseException as error:
                print("Error occured when printing algorithm ",algorithm_name," for ",file_name)


    plt.tight_layout()
    plt.ylabel('Graphs for various algorithms')

    graph_name = 'Graph'+'_'+str(int(time.time()))
    save_location = out_location / graph_name

    plt.savefig(save_location  )
    print("Saved fig to ", out_location.resolve() )
    try:
        plt.show()
    except BaseException as error:
        print("Can't show plot")
