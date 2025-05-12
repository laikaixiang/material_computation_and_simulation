import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from count import visible_binary, visible_colorful

# 设置字体为黑体
plt.rcParams['font.family'] = 'SimHei'

def analyze_grains(csv_file_path, threshold=0.72, show_top_rate = 1, visible = "binary"):
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

    # 5. 输出结果
    # print("\n晶粒面积排序结果（从大到小）:")
    # for i, area in enumerate(sorted_areas, 1):
    #     print(f"第{i}位: {area} 像素")


    # 6.1 晶粒分布可视化--彩色
    if visible == "colorful":
        visible_colorful(num_features, labeled_array, show_top_rate, sorted_indices)
        # 显示面积分布直方图
        plt.figure(figsize=(8, 4))
        plt.hist(grain_areas, bins=50, color='steelblue', edgecolor='white')
        plt.xlabel('晶粒面积（像素）')
        plt.ylabel('频数')
        plt.title('晶粒面积分布直方图')
        plt.axvline(np.median(grain_areas), color='red', linestyle='--', label=f'中位数: {np.median(grain_areas):.1f}')
        plt.legend()
        plt.show()

    # 6.2 晶粒分布可视化--黑白
    elif visible == "binary":
        visible_binary(num_features, binary_data, labeled_array, show_top_rate, sorted_indices)
        # 显示面积分布直方图
        plt.figure(figsize=(8, 4))
        plt.hist(grain_areas, bins=50, color='steelblue', edgecolor='white')
        plt.xlabel('晶粒面积（像素）')
        plt.ylabel('频数')
        plt.title('晶粒面积分布直方图')
        plt.axvline(np.median(grain_areas), color='red', linestyle='--', label=f'中位数: {np.median(grain_areas):.1f}')
        plt.legend()
        plt.show()

    # 6.3 不可视化
    else:
        pass

    return num_features, sorted_areas



# 使用示例
if __name__ == "__main__":
    csv_file_path = "time_15000.csv"
    threshold_list = [1-i/100 for i in range(0,30,1)]
    len_list = []
    for threshold in threshold_list:
        num_grains, areas = analyze_grains(csv_file_path, threshold = threshold, visible=False)
        print(len(areas))
        len_list.append(len(areas))
    print(len_list)
    plt.figure()
    plt.plot(threshold_list, len_list)
    plt.show()
    print("最优参数为0.72")