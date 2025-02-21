from shapely.geometry.polygon import LinearRing, Polygon
from shapely.geometry import Point
import numpy as np


def simplify_polygon_vertices(vertices, tolerance=1e-6):
    """
    简化多边形的顶点列表，删除共线的顶点（改进版本，不强制保留第一个顶点）。

    Args:
        vertices: 一个列表，包含多边形的顶点坐标，例如 [[0, 0], [10, 0], [10, 10], [0, 10]]
        tolerance: 浮点数，用于判断点是否共线的容差值。

    Returns:
        一个列表，包含简化后的顶点坐标。
    """

    if len(vertices) < 3:
        return vertices

    simplified_vertices = []
    n = len(vertices)

    for i in range(n):
        # 形成一个滑动窗口，包含三个连续的点
        p1 = Point(vertices[(i - 1) % n])  # 前一个点
        p2 = Point(vertices[i])          # 当前点
        p3 = Point(vertices[(i + 1) % n])  # 下一个点

        # 计算面积
        polygon = Polygon([p1, p2, p3])
        area = polygon.area

        # 如果面积足够小，则认为当前点可以删除
        if abs(area) > tolerance:
            simplified_vertices.append(vertices[i])

    # 如果简化后只剩两个点，需要重新添加一个点，否则无法构成多边形
    if len(simplified_vertices) < 3 and len(vertices) >= 3:
        return vertices  # 无法简化，返回原顶点列表

    return simplified_vertices


def judge_angle(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    vt = np.array([v1[1], -v1[0]])
    return np.sign(np.dot(vt, v2))


def rotate_polygon(polygon):
    rotate_flag = True
    while rotate_flag:
        rotate_flag = False
        p1 = polygon[-1]
        p2 = polygon[0]
        p3 = polygon[1]
        if judge_angle(p1, p2, p3) > 0:
            polygon.insert(0, polygon.pop())
            rotate_flag = True
            
    return polygon


def generate_layer(polygon):
    polygon = simplify_polygon_vertices(polygon, tolerance=1e-6)
    polygon = rotate_polygon(polygon)
    ring_outer = LinearRing(polygon+[polygon[0]])
    # ring_outer = LinearRing([[0,0],[10,0],[10,10],[5,5],[0,10]])
    ring_outer_list = [ring_outer.parallel_offset(0.2, side='left', join_style='mitre')]

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_aspect(aspect=1)
    # plot_line(ax, ring_outer, "blue")
    while(1):
        ring_outer_offset = ring_outer_list[-1].parallel_offset(0.4, side='left', join_style='mitre')
        # plot_line(ax, ring_outer_offset, "green")
        if Polygon(ring_outer_offset).area < 0.1:
            break
        ring_outer_list.append(ring_outer_offset)
    # plt.show()

    point_list = []
    for ring in ring_outer_list:
        for point in ring.coords:
            point_list.append(point)
            
    return point_list