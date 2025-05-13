import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis import analyze_grains

# 设置字体为黑体
plt.rcParams['font.family'] = 'SimSun'

def get_time_intervals(csv_files):
    """根据时间步长范围确定分析间隔"""
    # time_points = [int(f.split('_')[1].split('.')[0]) for f in csv_files]
    # max_time = max(time_points)
    #
    # intervals = []
    # if max_time <= 10:
    #     intervals = list(range(1, max_time + 1))
    # elif max_time <= 100:
    #     intervals = list(range(1, 11)) + list(range(20, 101, 10))
    # elif max_time <= 1000:
    #     intervals = list(range(1, 101, 10)) + list(range(200, 1001, 100))
    # else:
    #     intervals = list(range(1, 1001, 100)) + list(range(2000, max_time + 1, 1000))
    #
    # # 确保只包含实际存在的时间点
    # existing_intervals = [t for t in intervals if f"time_{t}.csv" in csv_files]
    existing_intervals = [int(file_name[5:-4]) for file_name in csv_files]
    return existing_intervals


def load_selected_data(data_dir, intervals):
    """加载选定时间点的数据"""
    all_results = []

    for t in intervals:
        csv_file = f"time_{t}.csv"
        csv_path = os.path.join(data_dir, csv_file)

        try:
            num_grains, areas, topology = analyze_grains(csv_path, threshold=0.72,
                                                         print_result=False,
                                                         visible_analyze=False)

            # 为每个晶粒添加面积信息到topology字典中
            for grain_id in topology.keys():
                topology[grain_id]['area'] = areas[grain_id - 1]  # 假设ID从1开始

            all_results.append({
                'time': t,
                'num_grains': num_grains,
                'avg_area': np.mean(areas),
                'area_distribution': areas,
                'topology': topology
            })
        except Exception as e:
            print(f"Error analyzing {csv_file}: {str(e)}")

    return all_results


def dynamic_analysis(all_results):
    """根据时间阶段采用不同的分析策略"""
    early_stage = [r for r in all_results if r['time'] <= 100]
    mid_stage = [r for r in all_results if 100 < r['time'] <= 1000]
    late_stage = [r for r in all_results if r['time'] > 1000]

    analysis_results = {}

    # 初期阶段(密集采样): 详细分析晶粒形成过程
    if early_stage:
        analysis_results['early'] = analyze_formation_stage(early_stage)

    # 中期阶段: 分析拓扑结构演变
    if mid_stage:
        analysis_results['mid'] = analyze_topology_evolution(mid_stage)

    # 后期阶段: 分析稳态生长动力学
    if late_stage:
        analysis_results['late'] = analyze_steady_state(late_stage)

    return analysis_results


def analyze_formation_stage(results):
    """分析晶粒形成初期阶段"""
    formation_data = {
        'times': [r['time'] for r in results],
        'nucleation_rate': [],  # 晶粒形成速率
        'size_distribution': []  # 尺寸分布变化
    }

    # 计算晶粒形成速率
    for i in range(1, len(results)):
        delta_t = results[i]['time'] - results[i - 1]['time']
        delta_n = results[i]['num_grains'] - results[i - 1]['num_grains']
        formation_data['nucleation_rate'].append(delta_n / delta_t)

    return formation_data


def analyze_topology_evolution(results):
    """分析拓扑结构演变阶段"""
    topology_data = {
        'times': [r['time'] for r in results],
        'avg_sides': [],
        'side_distribution': [],
        'angle_distribution': []
    }

    for r in results:
        topology = r['topology']
        sides = [v['num_sides'] for v in topology.values()]
        angles = np.concatenate([v['angles'] for v in topology.values()])

        topology_data['avg_sides'].append(np.mean(sides))
        topology_data['side_distribution'].append(pd.Series(sides).value_counts().sort_index())
        topology_data['angle_distribution'].append(angles)

    return topology_data


def analyze_steady_state(results):
    """分析稳态生长阶段"""
    steady_data = {
        'times': [r['time'] for r in results],
        'growth_rates': [],
        'size_topology_correlation': []
    }

    # 计算平均生长速率
    for i in range(1, len(results)):
        delta_t = results[i]['time'] - results[i - 1]['time']
        delta_area = results[i]['avg_area'] - results[i - 1]['avg_area']
        steady_data['growth_rates'].append(delta_area / delta_t)

    # 计算尺寸-拓扑相关性(类似Lewis定律)
    for r in results:
        topology = r['topology']
        size_topology = []

        for grain_id, props in topology.items():
            # 直接从props中获取面积
            area = props['area']
            size_topology.append({
                'area': area,
                'sides': props['num_sides']
            })

        steady_data['size_topology_correlation'].append(size_topology)

    return steady_data


def plot_dynamic_results(analysis_results):
    """根据不同阶段的特点进行可视化（独立图表，中文标签）"""

    # 1. 晶粒成核速率（初期阶段）
    if 'early' in analysis_results:
        plt.figure(figsize=(10, 6))
        times = analysis_results['early']['times']
        nucleation_rates = analysis_results['early']['nucleation_rate']
        plt.plot(times[1:], nucleation_rates, 'bo-')
        plt.xlabel('时间（初期阶段）')
        plt.ylabel('晶粒成核速率')
        plt.title('初期阶段晶粒成核速率变化')
        plt.grid(True)
        plt.show()

    # 2. 平均边数随时间变化
    if 'mid' in analysis_results:
        plt.figure(figsize=(10, 6))
        times = analysis_results['mid']['times']
        avg_sides = analysis_results['mid']['avg_sides']
        plt.plot(times, avg_sides, 'ro-')
        plt.axhline(y=6, color='k', linestyle='--', label='理论平均值=6')
        plt.xlabel('时间')
        plt.ylabel('平均边数')
        plt.title('晶粒平均边数随时间变化')
        plt.legend()
        plt.grid(True)
        plt.savefig('spectral_analysis_results.svg', dpi=300, format='svg')
        plt.show()

    # 3. 边数分布（选择代表性时间点）
    if 'mid' in analysis_results:
        plt.figure(figsize=(10, 6))
        side_dists = analysis_results['mid']['side_distribution']
        times = analysis_results['mid']['times']
        selected_indices = [0, len(side_dists) // 2, -1]  # 首、中、末时间点

        for idx in selected_indices:
            dist = side_dists[idx]
            time = times[idx]
            plt.plot(dist.index, dist.values, 'o-', label=f't={time}')

        plt.xlabel('边数')
        plt.ylabel('出现频率')
        plt.title('晶粒边数分布（对照论文图3）')
        plt.legend(title='时间点')
        plt.grid(True)
        plt.show()

    # 4. 生长速率（稳态阶段）
    if 'late' in analysis_results:
        plt.figure(figsize=(10, 6))
        times = analysis_results['late']['times'][1:]
        growth_rates = analysis_results['late']['growth_rates']
        plt.plot(times, growth_rates, 'go-')
        plt.xlabel('时间（稳态阶段）')
        plt.ylabel('平均生长速率（像素/时间步）')
        plt.title('稳态阶段晶粒平均生长速率')
        plt.grid(True)
        plt.show()

    # 5. 尺寸-拓扑相关性图（类似Lewis定律）
    if 'late' in analysis_results and len(analysis_results['late']['size_topology_correlation']) > 0:
        plot_lewis_law_relationship(analysis_results['late']['size_topology_correlation'][-1])


def plot_lewis_law_relationship(size_topology_data):
    """绘制类似论文图7的Lewis定律关系图（独立图表，中文标签）"""
    areas = [d['area'] for d in size_topology_data]
    sides = [d['sides'] for d in size_topology_data]

    # 按边数分组计算平均面积
    df = pd.DataFrame({'边数': sides, '面积': areas})
    avg_areas = df.groupby('边数')['面积'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_areas.index, avg_areas.values, 'bo-', label='模拟数据')

    # 尝试线性拟合(忽略n<5的数据点，如论文所示)
    fit_df = avg_areas[avg_areas.index >= 5].reset_index()
    if len(fit_df) > 2:
        coeffs = np.polyfit(fit_df['边数'], fit_df['面积'], 1)
        poly = np.poly1d(coeffs)
        x_fit = np.linspace(5, max(fit_df['边数']), 100)
        plt.plot(x_fit, poly(x_fit), 'r--',
                 label=f'线性拟合: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}')

    plt.xlabel('晶粒边数（n）')
    plt.ylabel('平均晶粒面积（像素）')
    plt.title('晶粒面积与边数关系（对照论文图7）')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    data_dir = "data/"
    csv_files = sorted([f for f in os.listdir(data_dir) if f.startswith('time_') and f.endswith('.csv')],
                       key=lambda x: int(x.split('_')[1].split('.')[0]))

    # 1. 根据时间步长范围确定分析间隔
    selected_intervals = get_time_intervals(csv_files)
    print(f"Selected time intervals for analysis: {selected_intervals}")

    # 2. 加载选定时间点的数据
    all_results = load_selected_data(data_dir, selected_intervals)

    # 3. 动态分析不同阶段
    analysis_results = dynamic_analysis(all_results)

    # 4. 可视化结果
    plot_dynamic_results(analysis_results)

    return analysis_results


if __name__ == "__main__":
    results = main()