# TopicsGraphAlgorithm
repo for adv. topics in computer science

# Requirements
Run on python 3.8.5 >=
Must python -m pip install: 
 * networkx[default]
 * matplotlib
 * pathlib

# Running the code:
python script.py [-h] [--simpleRulesV1] [--circular] [--random] inputLocation outputLocation

Where input/output location are the ABSOLUTE file paths.
Input: path to a folder of inputs
Output: path to save output

Each option (--xxx) enables an algorithm to be applied to all inputs.
Current algorithms developed:
 * simpleRulesV1

Rest are implemented by networkx

Do NOT run this inside wsl, as filenames will not be interpreted properly

Each input file represents a graph, where each file is a csv where each row represents a connection

In that row:
 * Row 1: vertex the connection comes from
 * Row 2: vertex the connection is going to
 * Row 3: type of connection, where 1 represents control and 2 represents data

The first line of the input must be a header.

# Notes:
Current iteration of the library dictates that in order for simpleRules to produce an output, all elements of the directed graphs must be reacheable from vertex 0

# TODO:
 * Heuristic measurer for how good the solution is
# TODO ALGORITHMS:
 * Randomness based graph
 * PCB arrangement graph


