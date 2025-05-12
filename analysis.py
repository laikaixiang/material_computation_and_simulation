import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.measure import find_contours, approximate_polygon
from skimage.segmentation import find_boundaries
import math
from count import visible_binary

# 设置字体为黑体
plt.rcParams['font.family'] = 'SimHei'

def calculate_angles(polygon):
    """计算多边形各个内角"""
    angles = []
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        p3 = polygon[(i + 2) % n]

        # 计算向量
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        # 计算角度（0-180度）
        angle = np.degrees(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
        angle = angle + 360 if angle < 0 else angle
        angle = 360 - angle if angle > 180 else angle
        angles.append(angle)
    return angles


def analyze_grain_topology(labeled_array, grain_id):
    """分析单个晶粒的拓扑结构"""
    # 提取当前晶粒的掩膜
    mask = (labeled_array == grain_id)

    # 找到边界
    boundaries = find_boundaries(mask, mode='outer')
    coords = np.column_stack(np.where(boundaries))

    # 找到轮廓（可能有多条，取最长的）
    contours = find_contours(mask, 0.5)
    if not contours:
        return None
    main_contour = max(contours, key=len)

    # 多边形近似（简化顶点）
    polygon = approximate_polygon(main_contour, tolerance=1.0)

    # 计算拓扑特征
    num_sides = len(polygon)
    angles = calculate_angles(polygon)

    return {
        'num_sides': num_sides,
        'angles': angles,
        'polygon': polygon,
        'contour': main_contour
    }


def analyze_grains(csv_file_path, threshold=0.72, visible_analyze=False):
    # 1. 读取CSV文件
    data = pd.read_csv(csv_file_path, header=None).values

    # 2. 二值化处理
    binary_data = (data > threshold).astype(int)

    # 3. 标记连通区域
    labeled_array, num_features = ndimage.label(binary_data)

    print(f"总晶粒数量: {num_features}")

    # 4. 计算每个晶粒的面积
    grain_areas = np.bincount(labeled_array.ravel())[1:]
    sorted_indices = np.argsort(grain_areas)[::-1]
    sorted_areas = grain_areas[sorted_indices]

    # 5. 拓扑分析
    topology_results = {}
    # if analyze_topology:
    print("\n正在进行拓扑分析...")
    for grain_id in range(1, num_features + 1):
        result = analyze_grain_topology(labeled_array, grain_id)
        if result:
            topology_results[grain_id] = result

    # 6. 输出结果
    print("\n晶粒面积排序结果（从大到小）:")
    for i, idx in enumerate(sorted_indices, 1):
        grain_id = idx + 1
        area = grain_areas[idx]
        topo_info = ""
        if grain_id in topology_results:
            topo_info = f", 边数: {topology_results[grain_id]['num_sides']}"
        print(f"第{i}位: 晶粒{grain_id}, 面积: {area} 像素{topo_info}")

    # 7. 可视化
    if visible_analyze:
        # 7.1 原始数据
        # plt.subplot(1, 3, 1)
        plt.figure(figsize=(6, 6))
        plt.imshow(data, cmap='viridis')
        plt.title('原始数据')
        plt.colorbar()
        plt.show()

        # 7.2 标记的晶粒
        # plt.figure(figsize=(6, 6))
        # colors = plt.cm.get_cmap('tab20', num_features + 1)(np.arange(num_features + 1))
        # colors[0] = [0, 0, 0, 1]  # 背景
        # cmap = mcolors.ListedColormap(colors)
        # norm = mcolors.BoundaryNorm(np.arange(num_features + 2) - 0.5, cmap.N)
        # img = plt.imshow(labeled_array, cmap=cmap, norm=norm)
        # plt.title('标记的晶粒')
        # plt.colorbar(img, ticks=np.arange(1, num_features + 1), label='晶粒编号')
        # plt.show()
        visible_binary(num_features, binary_data, labeled_array, 1, sorted_indices)

        # 7.3 拓扑结构可视化（显示边数和角度）
        plt.figure(figsize=(6, 6))
        plt.imshow(labeled_array > 0, cmap='gray')  # 背景

        # 显示前20个最大晶粒的拓扑结构（避免图像过于拥挤）
        max_grains_to_show = min(20, num_features)
        for grain_id in sorted_indices[:max_grains_to_show] + 1:
            if grain_id in topology_results:
                result = topology_results[grain_id]
                polygon = result['polygon']
                plt.plot(polygon[:, 1], polygon[:, 0], 'r-', linewidth=1)

                # 标记边数和平均角度
                center = np.mean(polygon, axis=0)
                avg_angle = np.mean(result['angles'])
                plt.text(center[1], center[0],
                         f"{result['num_sides']}边\n{avg_angle:.1f}°",
                         color='red', ha='center', va='center',
                         fontsize=8, fontweight='bold')

        plt.title('晶粒拓扑结构（边数和平均角度）')
        plt.show()

    return num_features, sorted_areas, topology_results


# 使用示例
if __name__ == "__main__":
    csv_file_path = "time_15000.csv"  # 替换为您的CSV文件路径
    num_grains, areas, topology = analyze_grains(csv_file_path, visible_analyze=True)

    # 输出前10个晶粒的详细拓扑信息
    print("\n前10个晶粒的详细拓扑信息:")
    for grain_id in list(topology.keys())[:10]:
        info = topology[grain_id]
        print(
            f"晶粒{grain_id}: {info['num_sides']}边, 角度范围: {min(info['angles']):.1f}°-{max(info['angles']):.1f}°, 平均角度: {np.mean(info['angles']):.1f}°")