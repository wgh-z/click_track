# Description: 用于判断点和多边形是否在多边形内部

# 点是否在矩形内部
def point_in_rect(point, rect):  # xy, xyxy
    is_in = False
    if point is not None:
        if rect[0] < point[0] < rect[2] and rect[1] < point[1] < rect[3]:
            is_in = True
    return is_in


# 点是否在多边形内部
def point_in_polygon(point, poly_points):
    is_in = False
    i = -1
    points_num = len(poly_points)
    j = points_num - 1
    while i < points_num - 1:
        i += 1
        # 测试对应的点
        # print(i, poly[i], j, poly[j])

        if ((poly_points[i][0] <= point[0] < poly_points[j][0]) or
                (poly_points[j][0] <= point[0] < poly_points[i][0])):
            if (point[1] < (poly_points[j][1] - poly_points[i][1]) * (point[0] - poly_points[i][0]) / (
                    poly_points[j][0] - poly_points[i][0]) + poly_points[i][1]):
                is_in = not is_in
        j = i
    return is_in


# 多边形是否在多边形内部:
def ploy_in_polygon(poly1, poly2):
    for point in poly1:
        if not point_in_polygon(point, poly2):
            return False
    return True
