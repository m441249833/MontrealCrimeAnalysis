#-------------------------------------------------------
# Assignment 1
# Written by Ke Ma 26701531
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

#test case1: -73.59 45.49 to -73.55 45.49 threshold = 0.5
#test case2: -73.59 45.49 to -73.55 45.53 threshold = 0.5
#test case3: -73.568 45.508 to -73.55 45.53 threshold = 0.5 (Supposed no path)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import shapefile
import numpy as np
from collections import defaultdict
import math

grid_size = 0.002
x1 = np.arange(-73.590,-73.550,grid_size)
y1 = np.arange(45.490,45.5301,grid_size)
shape = shapefile.Reader("Shape/crime_dt.shp",encoding='ISO-8859-1')
shapeRecords = shape.shapeRecords()
num_seq = []
grid_map= []

for i in range(len(x1)):
    col = []
    for j in range(len(y1)):
        col.append(0)
    grid_map.append(col)

x_coordinates=[]
y_coordinates=[]

#counting the density(crime rate) in each grid.
for k in range(len(shapeRecords)):
    x = float(shapeRecords[k].shape.__geo_interface__["coordinates"][0])
    y = float(shapeRecords[k].shape.__geo_interface__["coordinates"][1])
    x_coordinates.append(x)
    y_coordinates.append(y)
    x = int((x - (-73.590)) / grid_size)
    y = int((y - (45.490)) / grid_size)
    grid_map[y][x] +=1
#grid_map.reverse()
#display the grid number
# for row in grid_map:
#     print(row)

#storing all rate numbers in a list from the grid_map, and sort the list in descending order.
for i in range(len(grid_map)-1):
    for j in range(len(grid_map[i])-1):
        num_seq.append(grid_map[i][j])
num_seq = sorted(num_seq,reverse=True)

# promote user to input the threshold, and also verify if the threshold is valid.
valid_threshold = False
while (not valid_threshold):
    threshold = input("Please enter a threshold:")
    if (float(threshold) <= 1) and (float(threshold) >= 0):
        valid_threshold = True

# define the high crime block based on the top (1-threshold) crime rate numbers.

index = len(num_seq) - int(float(threshold) * len(num_seq)) - 1
high_num = num_seq [index]

# plot the grid using only 2 colors.
if threshold == 0.0 :
     plt.hist2d(x_coordinates, y_coordinates, bins=[x1, y1], cmap=ListedColormap(['yellow']))
elif threshold == 1.0:
    plt.hist2d(x_coordinates, y_coordinates, bins=[x1, y1], cmap=ListedColormap(['purple']))
else:
    plt.hist2d(x_coordinates, y_coordinates, bins=[x1, y1], cmap=ListedColormap(['purple', 'yellow']), vmin=0,vmax=2 * high_num)

#display statics
print("Total number of crime:",sum(num_seq))
print("Average:","{:.3f}".format(np.average(num_seq)))
print("Standard deviation:","{:.3f}".format(np.std(num_seq)))
print("High crime rate:",high_num)
#plt.show()


for i in range(len(grid_map)):
    for j in range(len(grid_map[i])):
        if grid_map[i][j] >= high_num:
            grid_map[i][j] = 1
        else:
            grid_map[i][j] = 0


def getNeighbours(point):
    l = []
    point_x = point[0]
    point_y = point[1]

    # up
    if point_y+1 <= len(grid_map)-1:
        if point_x == 0 or point_x == len(grid_map[0])-1:
            l.append((point_x,point_y+1))
        else:
            if not(grid_map[point_y][point_x] ==1 and grid_map[point_y][point_x-1] ==1):
                l.append((point_x,point_y+1))
    #down
    if point_y-1 >= 0:
        if point_x == 0 or point_x == len(grid_map[0])-1:
            l.append((point_x, point_y - 1))
        else:
            if not(grid_map[point_y-1][point_x] ==1 and grid_map[point_y-1][point_x-1] ==1):
                l.append((point_x,point_y-1))
    #left
    if point_x-1 >=0:
        if point_y == 0 or point_y == len(grid_map)-1:
            l.append((point_x-1,point_y))
        else:
            if not (grid_map[point_y][point_x-1] == 1 and grid_map[point_y-1][point_x-1] == 1):
                l.append((point_x-1,point_y))
    #right
    if point_x+1 <= len(grid_map[0])-1:
        if point_y == 0 or point_y == len(grid_map)-1:
            l.append((point_x+1,point_y))
        else:
            if not (grid_map[point_y][point_x] == 1 and grid_map[point_y-1][point_x] == 1):
                l.append((point_x + 1, point_y))
    #left-upper diagonal
    if point_x - 1 >=0 and point_y + 1 <=len(grid_map)-1:
        if grid_map[point_y][point_x-1] == 0:
            l.append((point_x-1,point_y+1))
    #right-upper diagonal
    if point_x + 1 <=len(grid_map[0])-1 and point_y+1<=len(grid_map)-1:
        if grid_map[point_y][point_x] == 0:
            l.append((point_x+1,point_y+1))
    #left-lower diagonal
    if point_x-1>=0 and point_y-1>=0:
        if grid_map[point_y-1][point_x-1] == 0:
            l.append((point_x-1,point_y-1))
    #right-lower diagonal
    if point_x+1 <= len(grid_map[0])-1 and point_y-1>=0:
        if grid_map[point_y-1][point_x] == 0:
            l.append((point_x+1,point_y-1))
    return l

#print(getNeighbours((4,1)))

def getCost(p1,p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    if x1 == x2 and y2-y1 ==1:
        if grid_map[y1][x1] ==1:
            return 1.3
        elif x1-1>=0 :
            if grid_map[y1][x1-1] ==1:
                return 1.3
            else:
                return 1
        else:
            return 1

    if x1 == x2 and y1-y2 == 1:
        if grid_map[y2][x2] ==1:
            return 1.3
        elif x1-1 >= 0 and y1-1 >=0:
            if grid_map[y1-1][x1-1] ==1:
                return 1.3
            else:
                return 1
        else:
            return 1
    if y1 == y2 and x1-x2 == 1:
        if grid_map[y2][x2] ==1:
            return 1.3
        elif y2-1>=0:
            if grid_map[y2-1][x2] ==1:
                return 1.3
            else:
                return 1
        else:
            return 1
    if y1 == y2 and x2-x1 == 1:
        if grid_map[y1][x1] == 1:
            return 1.3
        elif y1-1 >=0:
            if grid_map[y1-1][x1] ==1:
                return 1.3
            else:
                return 1
        else:
            return 1
    if abs(x1 - x2) ==1 and abs(y1 -y2) == 1:
        return 1.5

def construct_path(cameFrom,current):
    total_path = [current]
    while current in cameFrom.keys():
        current = cameFrom[current]
        total_path.insert(0,current)
    return total_path

def A_start(start, goal):
    # simple heuristic function estimate the cost from n to goal.
    def h(n):
        return abs(goal[0] - n[0]) + abs(goal[1] - n[1])

    # open set records the to be visited points
    openSet = [start]

    # a map records the nodes' parent node
    cameFrom = {}

    #g function records the cost from initial point to n.
    gScore = defaultdict(lambda: float("inf"))
    gScore[start] = 0

    # f function records the cost from start to n plus from n to the goal.
    fScore = defaultdict(lambda: float("inf"))
    fScore[start] = h(start)

    #visit each node in open set
    while openSet:
        lowest_fscore = math.inf
        current = None
        for point in openSet:
            if fScore[point] < lowest_fscore:
                lowest_fscore = fScore[point]
                current = point

        #if the current node is the goal, build the path
        if current == goal:
            return construct_path(cameFrom,current)

        #remove the visited node
        openSet.remove(current)
        neighbors = getNeighbours(current)
        for neighbor in neighbors:
            temp_gScore = gScore[current] + getCost(current,neighbor)
            if temp_gScore < gScore[neighbor]:

                cameFrom[neighbor] = current
                gScore[neighbor] = temp_gScore
                fScore[neighbor] = gScore[neighbor] + h(neighbor)
                if neighbor not in openSet:
                    openSet.append(neighbor)

    return None

#promote user input for initial point and goal point
point_valid = False
while not point_valid:
    initial_x,initial_y = input("Please enter the start point:").split()
    goal_x,goal_y = input("Please enter the end point:").split()
    initial_x,initial_y = float(initial_x),float(initial_y)
    goal_x,goal_y = float(goal_x),float(goal_y)
    if (initial_x>=-73.590 and initial_x<=-73.550) and (initial_y>=45.490 and initial_y<=45.530) and (goal_x>=-73.590 and goal_x<=-73.550) and (goal_y>=45.490 and goal_y<=45.530):
        point_valid = True
    else:
        print("Invalid points, please enter another one.")

#covert the input points to integer so it can fit into my algorithm.
initial_x = int(((initial_x - (-73.59)) / grid_size) + 0.01)
initial_y = int(((initial_y - (45.49)) / grid_size) +0.01)
goal_x = int(((goal_x - (-73.59)) / grid_size)+0.01)
goal_y = int(((goal_y - (45.49)) / grid_size)+0.01)

#invoke A* algorithm to generate the optimal path
final_path = A_start((initial_x,initial_y),(goal_x,goal_y))

if final_path != None :
    # calculate the total cost
    total_cost = 0
    for i in range(len(final_path) - 1):
        total_cost += getCost(final_path[i], final_path[i + 1])

    # print(final_path)

    # convert the coordinates to original coordinates.
    real_path = []
    path_x = []
    path_y = []
    for point in final_path:
        real_path.append((round(point[0] * grid_size + (-73.59), 3), round(point[1] * grid_size + 45.49, 3)))
        path_x.append(round(point[0] * grid_size + (-73.59), 3))
        path_y.append(round(point[1] * grid_size + 45.49, 3))

    print("Path:",real_path)
    print("path cost:", "{:.2f}".format(total_cost))

    plt.plot(path_x, path_y, color="red", linewidth=6)
else:
    print("Due to blocks, no path is found. Please change the map and try again")

#show the plot
plt.xticks(x1, rotation=90)
plt.yticks(y1)
plt.show()
print("Program terminated.")