# pylint: disable-all
import graphLib as g
import argparse 
import pathlib
import copy
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time

#call function as
#script.py examplefilelocation -
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualise graphs using multiple graph display algorithms')

    
    algorithms = (
        'simpleRulesV1',
        'circular',
        'random'
    )

    algorithm_objects = {
        algorithms[0] : g.SimpleRulesV1(), 
        algorithms[1] : g.Circular(),
        algorithms[2] : g.Random()
    }


    

    algorithm_switches = [('--'+algorithms[0],  'Output simple rule-based graphing algorithm'), #SimpleRulesV1 
                          ('--'+algorithms[1], 'Output circular graph'),   #Circular implementation
                          ('--'+algorithms[2],  'Output randomised graph')]            #Random
    
    parser.add_argument('inputLocation',type=str,
                       help='Absolute location of the folder to load examples')
    parser.add_argument('outputLocation',type=str,
                       help='Absolute location of the folder to save output')
    
    for a in algorithm_switches:
        parser.add_argument(a[0],help=a[1],action='store_true',default=False)
    
    args = parser.parse_args()

    print(args)

    #count number of args passed
    algorithm_count =\
    int(args.simpleRulesV1) +\
    int(args.circular)+\
    int(args.random)

    algorithm_references=[]

    if (args.simpleRulesV1):
        algorithm_references.append((algorithm_objects[algorithms[0]],algorithms[0]))
    if (args.circular):
        algorithm_references.append((algorithm_objects[algorithms[1]],algorithms[1]))    
    if (args.random):
        algorithm_references.append((algorithm_objects[algorithms[2]],algorithms[2]))    

    folder_location = pathlib.Path(args.inputLocation)
    out_location = pathlib.Path(args.outputLocation)

    data_files = folder_location.glob('*.dat')
    file_count=0

    file_dirs = []

    for file in data_files:
        file_dirs.append(file) 
        file_count += 1

    print (file_count)



    plot_count = 0

    for file_number in range(file_count):
        for algorithm_number in range(algorithm_count):
            plot_count += 1
            file_dir = file_dirs[file_number]
            file_name = str(file_dir.stem)
            file_dir = str(file_dir.resolve())
            algorithm_name = algorithm_references[algorithm_number][1]
            try:
                ax = plt.subplot(file_count,algorithm_count,plot_count)
                algorithm_object = copy.deepcopy(algorithm_references[algorithm_number][0])
                algorithm_name = algorithm_references[algorithm_number][1]
                algorithm_object.load_file_to_graph(file_dir)
                positions = algorithm_object.get_position()
                graph = algorithm_object.create_unformatted_networkx_graph()

                options = {
                    "font_size": 10,
                    "node_size": 300,
                    "node_color": "white",
                    "edgecolors": "black",
                    "linewidths": 2,
                    "width": 2,
                }
                
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



    
    
    









    
    
    
