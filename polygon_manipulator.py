import numpy as np
import networkx as nx

layer_height = 0.4
layer_count = 12
delta = 0.8


def calculate_b(theta):
    
    t = theta * np.pi / 180 / 2   
    d = layer_height
    h = d * 2
    a = d * (layer_count - 2)
    b = a * np.tan(t) + 1/np.cos(t) * (h - (delta/(t*2) - d/2)*np.sin(t))

    return b


def calculate_width(direction, theta):
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
    比较两个顶点是否相同。
    """
    return abs(v1[0] - v2[0]) < tolerance and abs(v1[1] - v2[1]) < tolerance


def polygon_array_to_graph(polygon_array):
    """
    将多边形数组转换为无环图。
    """
    graph = nx.Graph()
    polygon_count = len(polygon_array)

    # 添加节点（多边形）到图中
    for i in range(polygon_count):
        graph.add_node(i)

    # 找到连接的边并添加到图中
    for i in range(polygon_count):
        for j in range(i + 1, polygon_count):  # 避免重复和自身连接
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

                    # 更严格的连接判断：比较端点坐标
                    if (is_same_vertex(v1, v3) and is_same_vertex(v2, v4)) or \
                       (is_same_vertex(v1, v4) and is_same_vertex(v2, v3)):
                        graph.add_edge(i, j, weight=width1)
                        break  # 找到了连接，不再遍历
    return graph


def has_cycles(graph):
    """
    判断图是否包含环。
    """
    try:
        nx.find_cycle(graph)
        return True
    except nx.NetworkXNoCycle:
        return False
    
    
def get_subtree_nodes(graph, root, visited=None):
    """
    获取以给定节点为根的子树中的所有节点。
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
    计算垂直于边的向量，方向从 polygon_index 指向 neighbor_index 所在的多边形。
    """

    polygon1_vertices = polygon_array[polygon_index][0]
    polygon2_vertices = polygon_array[neighbor_index][0]
    #寻找公共边
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

            # 匹配端点
            if (is_same_vertex(candidate_v1, neighbor_v1) and is_same_vertex(candidate_v2, neighbor_v2)) or \
                    (is_same_vertex(candidate_v1, neighbor_v2) and is_same_vertex(candidate_v2, neighbor_v1)):
                v1,v2 = candidate_v1, candidate_v2
                break

    if v1 is None or v2 is None:
        return np.array([0.0, 0.0])

    # 计算边的方向向量
    edge_direction = np.array(v2) - np.array(v1)
    # 垂直向量 (交换 x 和 y，然后取反一个)
    normal_vector = np.array([-edge_direction[1], edge_direction[0]])
    # 归一化向量
    norm = np.linalg.norm(normal_vector)
    if norm == 0:
        return np.array([0.0, 0.0])  # 避免除以零
    normalized_vector = normal_vector / norm

    weight = graph[polygon_index][neighbor_index]['weight']
    # 乘以权重的绝对值来确定向量的大小
    return normalized_vector * abs(weight)


def dfs_tree_traversal(graph, root, polygon_array, visited=None, result=None):
    """
    使用深度优先搜索遍历树，并将结果存储在高维列表中。
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
            result.append([[root, neighbor], weight, subtree_nodes, vector.tolist()])  # 存储结果，并将 NumPy 数组转换为列表

            dfs_tree_traversal(graph, neighbor, polygon_array, visited, result)  # 递归调用

    return result


def transform_polygons(polygon_array, edges):
    """
    对子树中的多边形顶点进行平移。

    Args:
        polygon_array: 原始多边形数组。
        edges:  dfs_tree_traversal 函数的输出结果 (edges 列表)。

    Returns:
        修改后的多边形数组。
    """
    transformed_polygon_array = [
        [np.array(vertices).tolist(), connections] for vertices, connections in polygon_array
    ]

    for edge in edges:
        _, _, subtree_nodes, translation_vector = edge
        translation_vector = np.array(translation_vector)  # 将平移向量转换为 NumPy 数组

        for node_index in subtree_nodes:
            # 将对应多边形的顶点坐标都加上平移向量
            transformed_polygon_array[node_index][0] = (np.array(transformed_polygon_array[node_index][0]) - translation_vector).tolist() #转换为list

    return transformed_polygon_array


def translate_connected_edges(polygon_array):
    """
    将每个多边形与别的多边形相连的边进行处理，插入新顶点并更新连接信息。

    Args:
        polygon_array: 原始多边形数组。

    Returns:
        修改后的多边形数组。
    """

    transformed_polygon_array = []
    for polygon_index, (vertices, connections) in enumerate(polygon_array):
        new_vertices = vertices[:]  # 复制顶点列表，避免修改原始数据
        new_connections = []  # 创建新的连接信息列表

        insert_info = []  # 存储要插入的顶点信息 (索引，顶点坐标)
        vertex_count = len(vertices)

        # count = 0
        for conn in connections:
        # for i in range (len(connections)):
        #     conn = connections[-(i+1)]
            edge_index = int(conn[0])
            width = np.abs(calculate_width(conn[1], conn[2]))  # 使用绝对值
            translation_distance = (width - delta) / 2.0

            # 获取边的顶点坐标
            v1_index = (edge_index - 1) % vertex_count
            v2_index = edge_index % vertex_count
            v1 = vertices[v1_index]
            v2 = vertices[v2_index]

            # 计算新的顶点位置
            edge_direction = np.array(v2) - np.array(v1)
            normal_vector = np.array([edge_direction[1], - edge_direction[0]])

            # # 确保法向量指向多边形外部
            # center = np.mean(np.array(vertices), axis=0)
            # midpoint = (np.array(v1) + np.array(v2)) / 2.0
            # to_center = center - midpoint

            # if np.dot(normal_vector, to_center) > 0:
            #     normal_vector = -normal_vector

            # 归一化向量
            norm = np.linalg.norm(normal_vector)
            if norm == 0:
                print('Warning: Normal vector is zero. Maybe two vertices are duplicate.')
                continue  # 避免除以零
            normalized_vector = normal_vector / norm
            translation_vector = normalized_vector * translation_distance

            # 计算新的顶点位置
            new_v1 = np.array(v1) + translation_vector
            new_v2 = np.array(v2) + translation_vector

            # 存储插入信息（在 v2 之前插入 new_v1，然后在 v2 之前插入 new_v2）
            insert_info.append((v1_index, new_v1.tolist()))
            insert_info.append((v1_index, new_v2.tolist()))
            # if count == 0:
            # count += 1
            
        # print(insert_info)


        # 插入新顶点
        offset = 1
        # for i, insert_vertex in sorted(insert_info):  # 确保索引从小到大排列
        for i, insert_vertex in insert_info:
            new_vertices.insert(i + offset, insert_vertex)
            offset += 1

        # 更新连接信息
        offset = 1
        for conn in connections:
            edge_index = int(conn[0])
            #由于顶点插入到边的第二个顶点，因此坐标不会发生改变
            new_connections.append([edge_index + offset, conn[1], conn[2]]) #添加未修改的连接信息
            offset += 2

        transformed_polygon_array.append([new_vertices, new_connections])

    return transformed_polygon_array


def add_hinges(polygons):
    graph = polygon_array_to_graph(polygons)
    
    if not has_cycles(graph):
        edges = dfs_tree_traversal(graph, 0, polygons)
        polygons_modified = transform_polygons(polygons, edges)
        polygons_final =  translate_connected_edges(polygons_modified)
        return polygons_final

    else:
        print('The origami pattern contains cycles, please manually make gcode file.')