import numpy as np

from geometry import build_prism
from cpo import generate_layer


def write_header_lines(file, layer_height, line_width, layer_count, mesh, start_x, start_y, start_z=0.3):
    with open(file, 'w') as f:
        header_lines = [
            f";layer_height = {layer_height}",
            f"\n;line_width = {line_width}",
            f"\n;layer_count = {layer_count}",
            f"\n;mesh = {mesh}"
        ]
        
        initialize_lines = [
            "\nG21 ;start of the code",
            "\nG1 Z15 F300",
            "\nG28 X0 Y0 ;Home",
            "\nG92 X0 Y0 ;Consider this as current",
            "\nG0 X50 Y50 F3000 ;Go-to Offset",  
            "\nG92 X0 Y0 ;Reset",
            "\n",
            f"\nG0 F3600 X{start_x:.3f} Y{start_y:.3f} Z{start_z:.3f} ;Go to start position",
            "\nM7",
            "\nG4 P150",
            "\n\n"
        ]
        
        f.writelines(header_lines)
        f.writelines(initialize_lines)
        

def write_finish_lines(file):
    with open(file, 'a') as f:
        finish_lines = [
            "\n\n;Finish",
            "\nM9",
            "\nG1 Z10.000"
            "\nG28 X-100 Y-100;Home",
        ]
        f.writelines(finish_lines)
        

def initialize_layer(xy, z, lines):
    lines.append(f"\nG1 X{xy[0]:.3f} Y{xy[1]:.3f} Z{z:.3f}")
    return lines


def write_move_to(file, x, y, z):
    lines = [
        f"\nM9",
        f"\nG1 Z10.000",
        f"\nG1 X{x:.3f} Y{y:.3f}",
        f"\nG1 Z{z:.3f}",
        f"\nM7",
        f"\nG4 P150"
    ]
    with open(file, 'a') as f:
        f.writelines(lines)
        

def write_layer(file, point_list, z, direction):
    """Draw single platform layer by concentrate infill

    Args:
        point_list (ndarray)  : np.array([[x1, y1], [x2, y2], ...])
        w (float)             : line width
        direction (int)       : 0 for out -> in (counter clockwise) 
                                1 for in -> out (clockwise)
    """
    
    if direction == 0:
        pl = point_list
    elif direction == 1:
        pl = []
        for point in point_list:
            pl.insert(0, point)
            
    lines = initialize_layer(pl[0], z, [])
    
    for p in pl:
        lines.append(f"\nG1 X{p[0]:.3f} Y{p[1]:.3f}")
        
    with open(file, 'a') as f:
        f.writelines(lines)
        
        
def go_to_next_layer(z, layer_height):
    z += layer_height
    
    
def generate_gcode(file, filename, polygons_final, outer_countour, direction, layer_height, line_width):
    write_header_lines(file, layer_height, line_width, 1, f'{filename}_{direction}', 0, 0)
    for polygon in polygons_final:
        write_layer(file, generate_layer(polygon[0]), 0.3, 1)

    write_layer(file, generate_layer(outer_countour), 0.7, 1)

    for polygon in polygons_final:
        prism = build_prism(polygon, direction)

        for layer_index, p in enumerate(prism):
            write_layer(file, generate_layer(p), 1.1+layer_index*0.4, layer_index%2)
            
    write_finish_lines(file)