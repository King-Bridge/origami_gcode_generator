import numpy as np

def calculate_arc(theta, Delta=0.8, mid_layer_count = 10, layer_height = 0.4):
    theta = np.deg2rad(theta)
    a = mid_layer_count * layer_height
    h = 2 * layer_height
    delta = layer_height
    r = Delta / theta
    b = a * np.tan(theta/2) + 1/np.cos(theta/2)*(h - (r - delta/2) * np.sin(theta/2))
    
    dl = []
    for i in range (mid_layer_count):
        x = np.sqrt(a ** 2 - (a - layer_height * (1/2 + i)) ** 2)
        dl.append(x*b/a)
        
    return dl


def build_prism_bottom(polygon, mid_layer_count):
    prism = []
    
    layer_count = mid_layer_count
    for edge in polygon[1]:
        if edge[1] == 0:
            layer_count /= 2
            break
        
    run_flag = False
    for edge in polygon[1]:
        if edge[1] == 1:
            run_flag = True
            break    
        
    if run_flag:    
        for i in range (int(layer_count)):
            pp = []
            for v in polygon[0]:
                pp.append(v)
            for j in range (len(polygon[1])):
                v_index = polygon[1][j][0]
                
                if polygon[1][j][1] == 0:
                    d = calculate_arc(polygon[1][j][2])[mid_layer_count - i - 1]
                else:
                    d = calculate_arc(polygon[1][j][2])[i]    
                
                dv = np.array(polygon[0][(v_index)%len(polygon[0])])- np.array(polygon[0][v_index-1])
                dv = [-dv[1], dv[0]]
                dv = np.array(dv)/np.linalg.norm(dv)
                pp[v_index] = (np.array(pp[v_index]) + dv*d).tolist()
                pp[v_index-1] = (np.array(pp[v_index-1]) + dv*d).tolist()
            prism.append(pp)
        
    return prism


def build_prism_top(polygon, mid_layer_count):
    prism = []
    
    layer_count = mid_layer_count
    for edge in polygon[1]:
        if edge[1] == 1:
            layer_count /= 2
            break
        
    run_flag = False
    for edge in polygon[1]:
        if edge[1] == 0:
            run_flag = True
            break    
        
    if run_flag:           
        for i in range (int(layer_count)):
            pp = []
            for v in polygon[0]:
                pp.append(v)
            for j in range (len(polygon[1])):
                v_index = polygon[1][j][0]
                
                if polygon[1][j][1] == 1:
                    d = calculate_arc(polygon[1][j][2])[mid_layer_count - i - 1]
                else:
                    d = calculate_arc(polygon[1][j][2])[i]    
                
                dv = np.array(polygon[0][(v_index)%len(polygon[0])])- np.array(polygon[0][v_index-1])
                dv = [-dv[1], dv[0]]
                dv = np.array(dv)/np.linalg.norm(dv)
                pp[v_index] = (np.array(pp[v_index]) + dv*d).tolist()
                pp[v_index-1] = (np.array(pp[v_index-1]) + dv*d).tolist()
            prism.append(pp)
            
    return prism


def build_prism(polygon, position, mid_layer_count=10):
    if position == 'bottom':
        prism = build_prism_bottom(polygon, mid_layer_count)
    elif position == 'top':
        prism = build_prism_top(polygon, mid_layer_count)
    return prism