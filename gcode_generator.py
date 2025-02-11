import numpy as np
import matplotlib.pyplot as plt

def define_shape():
    
    print("Please enter the number of polygons: ")
    num_polygons = int(input())
    
    # polygons = [polygon1, polygon2, polygon3, ...]
    # polygon  = [[vertex1, vertex2, ...], [hinge1, hinge2, ...]]
    # vertex   = [x, y]
    # hinge    = [side_number, hinge_orientation, bending_angle]
    polygons = []
    
    for i in range (num_polygons):
        print("\nPlease enter the number of vertices for polygon " + str(i) + ": ")
        num_vertices = int(input())
        vertices = []
        for j in range(num_vertices):
            print("\nthen please enter the vertices in counter-clockwise order.")
            print("\nPlease enter the x and y coordinates for vertex " + str(j) + ": ")
            x, y = input().split()
            x = int(x)
            y = int(y)
            vertices.append([x, y])
            
        print("\nPlease enter the number of sides for polygon " + str(i) + "that should be hinges: ")
        num_hinges = int(input())
        hinges = []
        for j in range(num_hinges):
            print("\nPlease enter the side number for hinge " + str(j) + ": ")
            side = int(input())
            print("\nPlease enter the orientation for hinge " + str(j) + "(m for mountain, v for valley): ")
            orientation = input()
            print("\nPlease enter the bending angle for hinge " + str(j) + "(0 to 90): ")
            angle = int(input())
            hinges.append([side, orientation, angle])
        
        polygons.append([vertices, hinges])
        
    return polygons




if __name__ == "__main__":
    
    polygons = define_shape()
    print(polygons)