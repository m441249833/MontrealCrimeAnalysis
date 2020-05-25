Steps:
    1. read all points from .shp file given
    2. collecting all points and plot on grid using hist2d method from Matplotlib
    3. running f(n) = g(n)+h(n) for scoring each node, and get the cheapest path.
    4. using numpy library to do statical calculation.
    5. plot the graph and display.
    
Libraries:
    Matplotlib.pyplot
    shapefile
    Numpy
    Collections
    Math

Test Cases:
    #test case1: -73.59 45.49 to -73.55 45.49 threshold = 0.5
    #test case2: -73.59 45.49 to -73.55 45.53 threshold = 0.5
    #test case3: -73.568 45.508 to -73.55 45.53 threshold = 0.5 (Supposed no path)

User input:
    1.threshold value
    2.initial state
    3.goal state

