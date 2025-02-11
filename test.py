import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_polygons(polygons_data):
    """
    绘制多边形图形，根据指定信息高亮特定边，并在边上标记数值。
    多个多边形共用同一条边时，只标记一次。
    现在边和数字都标记为指定颜色。

    Args:
        polygons_data: 包含多边形数据的列表，每个多边形包含顶点坐标和高亮边信息。
                       高亮边信息现在包含边的索引、颜色代码和标记值。
    """

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # 设置坐标轴范围
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    for polygon in polygons_data:
        vertices = polygon[0]
        for x, y in vertices:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)

    marked_edges = set()  # 用于存储已标记的边，避免重复标记

    for polygon in polygons_data:
        vertices = polygon[0]
        highlight_edges = polygon[1]

        # 绘制多边形边
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            color = 'black'  # 默认颜色为黑色
            edge_key = tuple(sorted(((x1, y1), (x2, y2))))  # 创建唯一标识边的元组
            text_color = 'black' # 默认文本颜色为黑色


            # 检查是否需要高亮当前边
            for edge_info in highlight_edges:
                edge_index = int(edge_info[0]) - 1
                if i == edge_index:
                    color_code = int(edge_info[1])
                    if color_code == 1:
                        color = 'blue'
                    elif color_code == 0:
                        color = 'red'
                    
                    if edge_key not in marked_edges:  # 添加检查边是否已被标记
                        value = edge_info[2]  # 获取要标记的值

                        # 计算边的中心点位置，用于放置文本
                        mid_x = (x1 + x2) / 2
                        mid_y = (y1 + y2) / 2

                        # 计算旋转角度，让文本与边平行
                        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                        
                        # 计算旋转角度，让文本与边平行
                        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                        # 修改旋转角度使文本冲上
                        angle -= 90 # 减去90度，使文本冲上

                        # 在边的中心点添加文本
                        ax.text(mid_x, mid_y, str(value), color=text_color, ha='center', va='center', rotation=angle, fontweight='bold')

                        marked_edges.add(edge_key)  # 将边添加到已标记集合中
                        break  # 找到并处理高亮边后，退出 edge_info 循环
            
            print([x1, x2], [y1, y2], color, text_color)
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2) # 绘制边的颜色在此处改变



    plt.title("Polygons with Highlighted Edges and Values")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

# 示例数据 (包含标记值, 共享一条边)
polygons_data = [
    [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], [[2.0, 1.0, 90.0]]],
    [[[10.0, 0.0], [20.0, 0.0], [20.0, 10.0], [10.0, 10.0]], [[4.0, 1.0, 90.0]]]
]

draw_polygons(polygons_data)