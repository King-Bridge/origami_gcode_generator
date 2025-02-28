import numpy as np
import networkx as nx

layer_height = 0.4
layer_count = 12
delta = 0.8


def calculate_width(direction, theta):
    """
    Calculate the move-inside-length of an edge
    """
    def calculate_b(theta):
    
        t = theta * np.pi / 180 / 2   
        d = layer_height
        h = d * 2
        a = d * (layer_count - 2)
        b = a * np.tan(t) + 1/np.cos(t) * (h - (delta/(t*2) - d/2)*np.sin(t))

        return b
    
    if theta <= 90:
        b = calculate_b(theta)
        w = 2 * b + delta
        if direction == 1:
            w *= -1
        return w
    
    else:
        theta = theta / 2
        b = calculate_b(theta)
        w = 2 * (2 * b + delta) + delta
        if direction == 1:
            w *= -1
        return w
    

def is_same_vertex(v1, v2, tolerance=0.001):
    """
    Compare whether two vertices are the same
    """
    return abs(v1[0] - v2[0]) < tolerance and abs(v1[1] - v2[1]) < tolerance


def polygon_array_to_graph(polygon_array):
    """
    Convert polygon arrays to acyclic graphs
    """
    graph = nx.Graph()
    polygon_count = len(polygon_array)

    for i in range(polygon_count):
        graph.add_node(i)

    for i in range(polygon_count):
        for j in range(i + 1, polygon_count):  
            polygon1_vertices = polygon_array[i][0]
            polygon1_connections = polygon_array[i][1]
            polygon2_vertices = polygon_array[j][0]
            polygon2_connections = polygon_array[j][1]

            for conn1 in polygon1_connections:
                edge_index1 = int(conn1[0])
                v1_index1 = (edge_index1 - 1) % len(polygon1_vertices)
                v2_index1 = edge_index1 % len(polygon1_vertices)
                v1 = polygon1_vertices[v1_index1]
                v2 = polygon1_vertices[v2_index1]
                width1 = calculate_width(conn1[1], conn1[2])

                for conn2 in polygon2_connections:
                    edge_index2 = int(conn2[0])
                    v3_index2 = (edge_index2 - 1) % len(polygon2_vertices)
                    v4_index2 = edge_index2 % len(polygon2_vertices)
                    v3 = polygon2_vertices[v3_index2]
                    v4 = polygon2_vertices[v4_index2]
                    width2 = calculate_width(conn2[1], conn2[2])

                    if (is_same_vertex(v1, v3) and is_same_vertex(v2, v4)) or \
                       (is_same_vertex(v1, v4) and is_same_vertex(v2, v3)):
                        graph.add_edge(i, j, weight=width1)
                        break 
    return graph


def has_cycles(graph):
    """
    Determine whether the graph contains a ring
    """
    try:
        nx.find_cycle(graph)
        return True
    except nx.NetworkXNoCycle:
        return False
    
    
def get_subtree_nodes(graph, root, visited=None):
    """
    Gets all nodes in the subtree rooted at the given node
    """
    if visited is None:
        visited = set()
    subtree_nodes = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node not in visited:
            subtree_nodes.add(node)
            visited.add(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    stack.append(neighbor)
    return list(subtree_nodes)


def edge_vector(polygon_index, neighbor_index, graph, polygon_array):
    """
    Computes the vector perpendicular to the edges, 
    pointing in the direction from polygon_index to the polygon where neighbor_index is located
    """

    polygon1_vertices = polygon_array[polygon_index][0]
    polygon2_vertices = polygon_array[neighbor_index][0]

    v1,v2 = None, None
    for conn1 in polygon_array[polygon_index][1]:
        edge_index1 = int(conn1[0])
        v1_index1 = (edge_index1 - 1) % len(polygon1_vertices)
        v2_index1 = edge_index1 % len(polygon1_vertices)
        candidate_v1 = polygon1_vertices[v1_index1]
        candidate_v2 = polygon1_vertices[v2_index1]
        for conn2 in polygon_array[neighbor_index][1]:
            edge_index2 = int(conn2[0])
            neighbor_v1_index2 = (edge_index2 - 1) % len(polygon2_vertices)
            neighbor_v2_index2 = edge_index2 % len(polygon2_vertices)
            neighbor_v1 = polygon2_vertices[neighbor_v1_index2]
            neighbor_v2 = polygon2_vertices[neighbor_v2_index2]

            if (is_same_vertex(candidate_v1, neighbor_v1) and is_same_vertex(candidate_v2, neighbor_v2)) or \
                    (is_same_vertex(candidate_v1, neighbor_v2) and is_same_vertex(candidate_v2, neighbor_v1)):
                v1,v2 = candidate_v1, candidate_v2
                break

    if v1 is None or v2 is None:
        return np.array([0.0, 0.0])

    edge_direction = np.array(v2) - np.array(v1)

    normal_vector = np.array([-edge_direction[1], edge_direction[0]])

    norm = np.linalg.norm(normal_vector)
    if norm == 0:
        return np.array([0.0, 0.0])  
    normalized_vector = normal_vector / norm

    weight = graph[polygon_index][neighbor_index]['weight']

    return normalized_vector * abs(weight)


def dfs_tree_traversal(graph, root, polygon_array, visited=None, result=None):
    """
    Traverse the tree using depth-first search and store the results in a high-dimensional list
    """
    if visited is None:
        visited = set()
    if result is None:
        result = []

    visited.add(root)

    for neighbor in graph.neighbors(root):
        if neighbor not in visited:
            edge_data = graph.get_edge_data(root, neighbor)
            weight = edge_data['weight']

            # 计算向量
            vector = edge_vector(root, neighbor, graph, polygon_array)

            subtree_nodes = get_subtree_nodes(graph, neighbor, visited.copy())
            result.append([[root, neighbor], weight, subtree_nodes, vector.tolist()])

            dfs_tree_traversal(graph, neighbor, polygon_array, visited, result)

    return result


def transform_polygons(polygon_array, edges):
    """
    Translate polygon vertices in a subtree.

    Args.
        polygon_array: Array of raw polygons.
        edges: output of dfs_tree_traversal (list of edges).

    Returns.
        The modified polygon array.
    """
    transformed_polygon_array = [
        [np.array(vertices).tolist(), connections] for vertices, connections in polygon_array
    ]

    for edge in edges:
        _, _, subtree_nodes, translation_vector = edge
        translation_vector = np.array(translation_vector) 

        for node_index in subtree_nodes:
            transformed_polygon_array[node_index][0] = (np.array(transformed_polygon_array[node_index][0]) - translation_vector).tolist()
    return transformed_polygon_array


def remove_duplicate_polygons(polygon_array, tolerance=1e-6):
    """
    Remove duplicate polygons from the polygon array 
    (consider the case where the vertices are in a different order).

    Args.
        polygon_array: The original polygon array.
        tolerance: Tolerance to be used when comparing floating point numbers.

    Returns.
        A new polygon array with unique polygons.
    """

    unique_polygons = []
    seen = set()

    for polygon in polygon_array:
        vertices = polygon[0]
        np_vertices = np.array(vertices)
        sorted_vertices = np_vertices[np.lexsort(np.transpose(np_vertices))]
        hashable_vertices = tuple(tuple(v) for v in sorted_vertices)

        if hashable_vertices not in seen:
            is_duplicate = False
            for seen_polygon in unique_polygons:
                seen_np_vertices = np.array(seen_polygon[0])
                seen_sorted_vertices = seen_np_vertices[np.lexsort(np.transpose(seen_np_vertices))]
                if np.allclose(sorted_vertices, seen_sorted_vertices, atol=tolerance):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_polygons.append(polygon)
                seen.add(hashable_vertices) 

    return unique_polygons


def translate_connected_edges(polygon_array):
    """
    Processes the edges of each polygon that are connected to other polygons, inserting new vertices and updating the connection information.

    Args.
        polygon_array: Array of raw polygons.

    Returns.
        The modified polygon array.
    """
    
    polygons_island = []

    transformed_polygon_array = []
    for polygon_index, (vertices, connections) in enumerate(polygon_array):
        new_vertices = vertices[:]
        new_connections = []  

        insert_info = [] 
        vertex_count = len(vertices)

        for conn in connections:

            edge_index = int(conn[0])
            theta = conn[2]
            if theta > 90:
                theta /= 2
                
            width = np.abs(calculate_width(conn[1], theta))  
            translation_distance = (width - delta) / 2.0

            v1_index = (edge_index - 1) % vertex_count
            v2_index = edge_index % vertex_count
            v1 = vertices[v1_index]
            v2 = vertices[v2_index]

            edge_direction = np.array(v2) - np.array(v1)
            normal_vector = np.array([edge_direction[1], - edge_direction[0]])

            norm = np.linalg.norm(normal_vector)
            if norm == 0:
                print('Warning: Normal vector is zero. Maybe two vertices are duplicate.')
                continue 
            normalized_vector = normal_vector / norm
            translation_vector = normalized_vector * translation_distance

            new_v1 = np.array(v1) + translation_vector
            new_v2 = np.array(v2) + translation_vector

            insert_info.append((v1_index, new_v1.tolist()))
            insert_info.append((v1_index, new_v2.tolist()))
            # if count == 0:
            if conn[2] > 90:
                vi1 = v1 + normalized_vector * (translation_distance + delta)
                vi2 = v2 + normalized_vector * (translation_distance + delta)
                vi3 = v2 + normalized_vector * (np.abs(calculate_width(conn[1], conn[2])) - translation_distance - delta)
                vi4 = v1 + normalized_vector * (np.abs(calculate_width(conn[1], conn[2])) - translation_distance - delta)
                
                polygons_island.append([[vi2.tolist(), vi1.tolist(), vi4.tolist(), vi3.tolist()], [[1, conn[1], conn[2]], [3, conn[1], conn[2]]]])
                
        offset = 1

        for i, insert_vertex in insert_info:
            new_vertices.insert(i + offset, insert_vertex)
            offset += 1

        offset = 1
        for conn in connections:
            edge_index = int(conn[0])
            new_connections.append([edge_index + offset, conn[1], conn[2]]) 
            offset += 2

        transformed_polygon_array.append([new_vertices, new_connections])
        

    transformed_polygon_array += remove_duplicate_polygons(polygons_island)
    
    for i in range (len(transformed_polygon_array)):
        for j in range (len(transformed_polygon_array[i][1])):
            if transformed_polygon_array[i][1][j][2] > 90:
                transformed_polygon_array[i][1][j][2] /= 2
                
    return transformed_polygon_array


def add_hinges(polygons):
    """
    Add hinges to the origin polygons
    """
    graph = polygon_array_to_graph(polygons)
    
    if not has_cycles(graph):
        edges = dfs_tree_traversal(graph, 0, polygons)
        polygons_modified = transform_polygons(polygons, edges)
        polygons_final =  translate_connected_edges(polygons_modified)
        return polygons_final

    else:
        print('The origami pattern contains cycles, please manually make gcode file.')
        
    
def translate_to_continent(polygon_array):
    """
    Processes the edges of each polygon that are connected to other polygons
    inserting new vertices and updating the connection information.
    """

    transformed_polygon_array = []
    for polygon_index, (vertices, connections) in enumerate(polygon_array):
        new_vertices = vertices[:] 
        new_connections = []  

        insert_info = []  
        vertex_count = len(vertices)

        for conn in connections:
    
            edge_index = int(conn[0])
                
            width = np.abs(calculate_width(conn[1], conn[2]))
            translation_distance = width / 2.0

            v1_index = (edge_index - 1) % vertex_count
            v2_index = edge_index % vertex_count
            v1 = vertices[v1_index]
            v2 = vertices[v2_index]

            edge_direction = np.array(v2) - np.array(v1)
            normal_vector = np.array([edge_direction[1], - edge_direction[0]])

            norm = np.linalg.norm(normal_vector)
            if norm == 0:
                print('Warning: Normal vector is zero. Maybe two vertices are duplicate.')
                continue 
            normalized_vector = normal_vector / norm
            translation_vector = normalized_vector * translation_distance

            new_v1 = np.array(v1) + translation_vector
            new_v2 = np.array(v2) + translation_vector
            insert_info.append((v1_index, new_v1.tolist()))
            insert_info.append((v1_index, new_v2.tolist()))
            
        offset = 1

        for i, insert_vertex in insert_info:
            new_vertices.insert(i + offset, insert_vertex)
            offset += 1

        offset = 1
        for conn in connections:
            edge_index = int(conn[0])
            new_connections.append([edge_index + offset, conn[1], conn[2]]) 
            offset += 2

        transformed_polygon_array.append([new_vertices, new_connections])
        
    return transformed_polygon_array


def translate_to_continent_for_surface(polygon_array):

    transformed_polygon_array = []
    for polygon_index, (vertices, connections) in enumerate(polygon_array):
        new_vertices = vertices[:]
        new_connections = [] 

        insert_info = []
        vertex_count = len(vertices)

        for conn in connections:

            edge_index = int(conn[0])
                
            width = np.abs(calculate_width(conn[1], conn[2])) 
            translation_distance = delta / 2

            v1_index = (edge_index - 1) % vertex_count
            v2_index = edge_index % vertex_count
            v1 = vertices[v1_index]
            v2 = vertices[v2_index]

            edge_direction = np.array(v2) - np.array(v1)
            normal_vector = np.array([edge_direction[1], - edge_direction[0]])

            norm = np.linalg.norm(normal_vector)
            if norm == 0:
                print('Warning: Normal vector is zero. Maybe two vertices are duplicate.')
                continue 
            normalized_vector = normal_vector / norm
            translation_vector = normalized_vector * translation_distance

            new_v1 = np.array(v1) + translation_vector
            new_v2 = np.array(v2) + translation_vector

            insert_info.append((v1_index, new_v1.tolist()))
            insert_info.append((v1_index, new_v2.tolist()))

        offset = 1
        for i, insert_vertex in insert_info:
            new_vertices.insert(i + offset, insert_vertex)
            offset += 1

        offset = 1
        for conn in connections:
            edge_index = int(conn[0])
            new_connections.append([edge_index + offset, conn[1], conn[2]]) 
            offset += 2

        transformed_polygon_array.append([new_vertices, new_connections])
        
    return transformed_polygon_array


def extract_outer_polygon_vertices(polygon_array):
    """
    Extracts the vertex information of large polygons from the polygon array and arranges them in contour order (independent of polar coordinates).

    Args.
        polygon_array: the original polygon array, satisfying the above conditions.

    Returns.
        A list of the coordinates of the vertices of the polygon, in contour order.
    """

    start_polygon_index = 0
    start_vertex_index = 0
    start_vertex = None

    for i, polygon in enumerate(polygon_array):
        is_shared_polygon = False
        for other_polygon in polygon_array:
            if other_polygon is polygon:
                continue
            for conn1 in polygon[1]:
                edge_index1 = int(conn1[0])
                polygon1_vertices = polygon[0]
                v1_index1 = (edge_index1 - 1) % len(polygon1_vertices)
                v2_index1 = edge_index1 % len(polygon1_vertices)
                candidate_v1 = polygon1_vertices[v1_index1]
                candidate_v2 = polygon1_vertices[v2_index1]
                for conn2 in other_polygon[1]:
                    edge_index2 = int(conn2[0])
                    other_vertices = other_polygon[0]
                    neighbor_v1_index2 = (edge_index2 - 1) % len(other_vertices)
                    neighbor_v2_index2 = edge_index2 % len(other_vertices)
                    neighbor_v1 = other_vertices[neighbor_v1_index2]
                    neighbor_v2 = other_vertices[neighbor_v2_index2]
                    if (is_same_vertex(candidate_v1, neighbor_v1) and is_same_vertex(candidate_v2, neighbor_v2)) or \
                        (is_same_vertex(candidate_v1, neighbor_v2) and is_same_vertex(candidate_v2, neighbor_v1)):
                        is_shared_polygon = True 
                        break
                if is_shared_polygon:
                    break
            if not is_shared_polygon:
                start_polygon_index = i
                break
        if not is_shared_polygon:
            break

    vertices = polygon_array[start_polygon_index][0]
    for j, vertex in enumerate(vertices):
        is_shared = False
        for conn in polygon_array[start_polygon_index][1]:
            edge_index = int(conn[0])
            v1_index = (edge_index - 1) % len(vertices)
            v2_index = edge_index % len(vertices)
            v1 = vertices[v1_index]
            v2 = vertices[v2_index]
            if (is_same_vertex(vertex, v1)) or (is_same_vertex(vertex,v2)):
                is_shared = True
                break 
        if not is_shared:
            start_vertex_index = j
            start_vertex = vertex
            break

    if start_vertex is None:
        print("Warning: No starting vertex found.")
        return [] 

    outer_polygon_vertices = []
    current_polygon_index = start_polygon_index
    current_vertex_index = start_vertex_index

    first_polygon_index = start_polygon_index
    first_vertex_index = start_vertex_index
    first_vertex = polygon_array[start_polygon_index][0][start_vertex_index]

    while True:
        current_polygon = polygon_array[current_polygon_index]
        vertices = current_polygon[0]
        vertex = vertices[current_vertex_index]

        if not outer_polygon_vertices or not is_same_vertex(vertex, outer_polygon_vertices[-1]):
            outer_polygon_vertices.append(vertex)

        next_vertex_index = (current_vertex_index + 1) % len(vertices)
        next_vertex = vertices[next_vertex_index]

        shared_polygon_index = None
        shared_edge_index = None

        for other_polygon_index, other_polygon in enumerate(polygon_array):
            if other_polygon_index == current_polygon_index:
                continue

            other_vertices = other_polygon[0]
            for conn in other_polygon[1]:
                edge_index = int(conn[0])
                v1_index = (edge_index - 1) % len(other_vertices)
                v2_index = edge_index % len(other_vertices)
                v1 = other_vertices[v1_index]
                v2 = other_vertices[v2_index]
                if (is_same_vertex(vertex, v2) and is_same_vertex(next_vertex, v1)):
                    shared_polygon_index = other_polygon_index
                    shared_edge_index = edge_index
                    break
                elif (is_same_vertex(vertex, v1) and is_same_vertex(next_vertex,v2)):
                    shared_polygon_index = other_polygon_index
                    shared_edge_index = edge_index
                    break
            if shared_polygon_index is not None:
                break

        if shared_polygon_index is None:
            current_vertex_index = next_vertex_index
        else:

            current_polygon_index = shared_polygon_index
            other_vertices = polygon_array[current_polygon_index][0]
            current_vertex_index = None
            for v_index, v in enumerate(other_vertices):
                if (is_same_vertex(vertex, v)):
                    current_vertex_index = v_index
                    break

            if current_vertex_index is None:
               raise ValueError("Cannot find the vertex in the shared polygon.")

        if current_polygon_index == first_polygon_index and is_same_vertex(polygon_array[current_polygon_index][0][current_vertex_index], first_vertex):
            break

    return outer_polygon_vertices


def find_outer_countour(polygons):
    graph = polygon_array_to_graph(polygons)
    
    if not has_cycles(graph):
        edges = dfs_tree_traversal(graph, 0, polygons)
        polygons_modified = transform_polygons(polygons, edges)
        outer_polygon_vertices = extract_outer_polygon_vertices(translate_to_continent(polygons_modified))
        return outer_polygon_vertices

    else:
        print('The origami pattern contains cycles, please manually make gcode file.')
        

def merge_sublists_with_common_elements(data):
    """
    Merge sublists with the same elements.

    Parameters:
        data: list containing the sublists.

    Return Value:
        The merged list.
    """

    def find_connected_sublist_index(current_index, merged_groups):

        current_set = set(data[current_index])
        for i, group in enumerate(merged_groups):
            if current_set.intersection(group):  
                return i
        return -1  

    merged_groups = []  

    for i in range(len(data)):
        connected_group_index = find_connected_sublist_index(i, merged_groups)

        if connected_group_index != -1:
            merged_groups[connected_group_index].update(data[i])
        else:
            merged_groups.append(set(data[i]))

    result = [list(group) for group in merged_groups]
    return result


def add_missing_numbers(data, n):

    all_elements = set()
    for sublist in data:
        all_elements.update(sublist)  

    missing_numbers = []
    for k in range(n):
        if k not in all_elements:
            missing_numbers.append([k])  

    return data + missing_numbers  


def generate_surface_polygons(polygons_final, direction):
    graph_surface = polygon_array_to_graph(translate_to_continent_for_surface(polygons_final))
    
    if not has_cycles(graph_surface):
        edges_surface = dfs_tree_traversal(graph_surface, 0, polygons_final)

    else:
        print('The origami pattern contains cycles, please manually make gcode file.')
        
    merge_list = []
    
    if direction == 'bottom':
        for edge in edges_surface:
            if edge[1] > 0:
                merge_list.append(edge[0])
    elif direction == 'top':
        for edge in edges_surface:
            if edge[1] < 0:
                merge_list.append(edge[0])
    merge_list = merge_sublists_with_common_elements(merge_list)
    merge_list = add_missing_numbers(merge_list, len(polygons_final))
    polygons_merged = []
    for polygon_list in merge_list:
        if len(polygon_list) == 1:
            polygons_merged.append(polygons_final[polygon_list[0]][0])
            continue
        polygon_to_merge = []
        for index in polygon_list:
            polygon_to_merge.append(polygons_final[index])
        polygon_to_merge = translate_to_continent_for_surface(polygon_to_merge)
        polygons_merged.append(extract_outer_polygon_vertices(polygon_to_merge))
    
    return polygons_merged