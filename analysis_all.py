import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis import analyze_grains
import seaborn as sns
import pickle

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
            num_grains, areas, topology = analyze_grains(csv_path, threshold=0.8,
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
    mid_stage = [r for r in all_results if 100 < r['time'] <= 5000]
    late_stage = [r for r in all_results if r['time'] > 5000]

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
    data = {
        'times': [r['time'] for r in results],
        'avg_sides': [],
        'side_distribution': [], # sides
        'angle_distribution': [], # angle
        'growth_rates': [],
        'size_topology_correlation': [], # area
        'num_grains': []
    }

    for r in results:
        topology = r['topology']
        sides = [v['num_sides'] for v in topology.values()]
        angles = np.concatenate([v['angles'] for v in topology.values()])

        data['avg_sides'].append(np.mean(sides)) # 平均边数
        data['side_distribution'].append(pd.Series(sides).value_counts().sort_index()) # 边数分布
        data['angle_distribution'].append(angles) # 角度分布
        data['num_grains'].append(r['num_grains']) # 晶粒数量

        # 计算平均生长速率
        for i in range(1, len(results)):
            delta_t = results[i]['time'] - results[i - 1]['time']
            delta_area = results[i]['avg_area'] - results[i - 1]['avg_area']
            data['growth_rates'].append(delta_area / delta_t)

        # area data面积数据
        size_topology = []
        for grain_id, props in topology.items():
            # 直接从props中获取面积
            area = props['area']
            size_topology.append({
                'area': area,
                'sides': props['num_sides']
            })

        data['size_topology_correlation'].append(size_topology)

    # 计算平均生长速率
    for i in range(1, len(results)):
        delta_t = results[i]['time'] - results[i - 1]['time']
        delta_area = results[i]['avg_area'] - results[i - 1]['avg_area']
        data['growth_rates'].append(delta_area / delta_t)

    return data


def analyze_steady_state(results):
    """分析稳态生长阶段"""
    data = {
        'times': [r['time'] for r in results],
        'avg_sides': [],
        'side_distribution': [],  # sides
        'angle_distribution': [],  # angle
        'growth_rates': [],
        'size_topology_correlation': [],
        'num_grains': []
    }

    # 计算尺寸-拓扑相关性(类似Lewis定律)
    for r in results:
        topology = r['topology']
        sides = [v['num_sides'] for v in topology.values()]
        angles = np.concatenate([v['angles'] for v in topology.values()])

        data['avg_sides'].append(np.mean(sides))  # 平均边数
        data['side_distribution'].append(pd.Series(sides).value_counts().sort_index())  # 边数分布
        data['angle_distribution'].append(angles)  # 角度分布
        data['num_grains'].append(r['num_grains'])

        # area data面积数据
        size_topology = []
        for grain_id, props in topology.items():
            # 直接从props中获取面积
            area = props['area']
            size_topology.append({
                'area': area,
                'sides': props['num_sides']
            })

        data['size_topology_correlation'].append(size_topology)

    # 计算平均生长速率
    for i in range(1, len(results)):
        delta_t = results[i]['time'] - results[i - 1]['time']
        delta_area = results[i]['avg_area'] - results[i - 1]['avg_area']
        data['growth_rates'].append(delta_area / delta_t)

    return data


def plot_dynamic_results(analysis_results):
    """根据不同阶段的特点进行可视化（独立图表，中文标签）"""
    image_dir = "images/"

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

        # 保存
        image_path = os.path.join(image_dir, '初期阶段晶粒成核速率变化.svg')
        plt.savefig(image_path, dpi=300, format='svg')

        # 展示
        plt.show()

    # 2. 平均边数随时间变化
    if 'mid' in analysis_results or 'late' in analysis_results:
        plt.figure(figsize=(12, 7))

        # 准备数据
        mid_times = analysis_results['mid']['times'] if 'mid' in analysis_results else []
        mid_avg_sides = analysis_results['mid']['avg_sides'] if 'mid' in analysis_results else []
        late_times = analysis_results['late']['times'] if 'late' in analysis_results else []
        late_avg_sides = analysis_results['late']['avg_sides'] if 'late' in analysis_results else []

        # 合并数据
        all_times = np.concatenate((mid_times, late_times))
        all_avg_sides = np.concatenate((mid_avg_sides, late_avg_sides))

        # 绘制曲线（中期用红色，后期用蓝色）
        if 'mid' in analysis_results:
            plt.plot(
                mid_times, mid_avg_sides,
                marker='o',  # 圆点标记
                linestyle='-',  # 实线
                color='r',  # 统一颜色
                linewidth=2,
                markersize=8,
                label='中期数据',
                alpha=0.8
            )
        if 'late' in analysis_results:
            plt.plot(
                late_times, late_avg_sides,
                marker='^',  # 方块标记
                linestyle='-',  # 实线（保持与中期一致）
                color='r',  # 统一颜色
                linewidth=2,
                markersize=8,
                label='后期数据',
                alpha=0.8
            )

        # 连起来
        plt.plot([mid_times[-1], late_times[0]],
                 [mid_avg_sides[-1], late_avg_sides[0]],
                 linestyle='-',  # 实线
                 color='r',  # 统一颜色
                 linewidth=2,
                 alpha=0.8
                 )

        # 理论参考线
        plt.axhline(y=6, color='gray', linestyle='--', label='理论平均值=6')
        plt.axhline(y=np.mean(all_avg_sides), color='k', linestyle='--', label=f'实际平均值={np.mean(all_avg_sides):2f}')

        # # 标记阶段分界
        # if 'mid' in analysis_results and 'late' in analysis_results:
        #     transition_time = mid_times[-1]
        #     plt.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.6)
        #     plt.text(transition_time, max(all_avg_sides) * 0.95,
        #              '中期→后期', ha='center', va='center',
        #              bbox=dict(facecolor='white', alpha=0.8))

        # # 添加趋势线（可选）
        # if len(all_times) > 2:
        #     coeffs = np.polyfit(all_times, all_avg_sides, 1)
        #     poly = np.poly1d(coeffs)
        #     plt.plot(all_times, poly(all_times), 'g--', linewidth=1.5,
        #              label=f'整体趋势: y={coeffs[0]:.3f}x+{coeffs[1]:.2f}')

        # 图表装饰
        plt.xlabel('时间（模拟步数）', fontsize=12)
        plt.ylabel('平均边数', fontsize=12)
        plt.title('晶粒平均边数随时间变化', fontsize=14, pad=20)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10, loc='best')

        # 保存图像
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, '晶粒平均边数随时间变化(全阶段).svg')
        plt.savefig(image_path, dpi=300, format='svg', bbox_inches='tight')

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
        plt.title('晶粒边数分布')
        plt.legend(title='时间点')
        plt.grid(True)

        # 保存
        image_path = os.path.join(image_dir, '晶粒边数分布.svg')
        plt.savefig(image_path, dpi=300, format='svg')

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

        # 保存
        image_path = os.path.join(image_dir, '稳态阶段晶粒平均生长速率.svg')
        plt.savefig(image_path, dpi=300, format='svg')

        plt.show()

    # 5. 尺寸-拓扑相关性图（类似Lewis定律）
    if 'late' in analysis_results and len(analysis_results['late']['size_topology_correlation']) > 0:
        plot_lewis_law_relationship(analysis_results['late']['size_topology_correlation'][-1])

    # 6. 平均晶体粒径变化（中期和后期阶段）
    if 'mid' in analysis_results or 'late' in analysis_results:
        plt.figure(figsize=(12, 7))

        # 收集所有时间点和对应的平均粒径
        all_times = []
        all_avg_sizes = []

        if 'mid' in analysis_results:
            for r in analysis_results['mid']['size_topology_correlation']:
                areas = [d['area'] for d in r]
                avg_size = np.mean(np.sqrt(areas))
                all_avg_sizes.append(avg_size)
            all_times.extend(analysis_results['mid']['times'])

        if 'late' in analysis_results:
            for r in analysis_results['late']['size_topology_correlation']:
                areas = [d['area'] for d in r]
                avg_size = np.mean(np.sqrt(areas))
                all_avg_sizes.append(avg_size)
            all_times.extend(analysis_results['late']['times'])

        # 转换为numpy数组便于计算
        times_array = np.array(all_times)
        sizes_array = np.array(all_avg_sizes)

        # 绘制原始数据点
        plt.plot(times_array, sizes_array, 'mo-', linewidth=2, markersize=8,
                 label='原始数据', alpha=0.7)

        # 线性拟合（可以选择对中期和后期分别拟合）
        if len(times_array) > 1:
            # 整体拟合
            coeffs = np.polyfit(times_array, sizes_array, 2)
            poly = np.poly1d(coeffs)
            fit_line = poly(times_array)

            # 计算R平方值
            residuals = sizes_array - fit_line
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((sizes_array - np.mean(sizes_array)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # 绘制拟合线
            plt.plot(times_array, fit_line, 'b--', linewidth=2,
                     label=f'二次函数拟合: y = {coeffs[0]:.4e}x² + {coeffs[1]:.2e} + {coeffs[2]:.2f}\nR² = {r_squared:.3f}')

            # 可选：对中期和后期分别拟合
            if 'mid' in analysis_results and 'late' in analysis_results:
                mid_end_idx = len(analysis_results['mid']['times'])

                # 中期拟合
                mid_coeffs = np.polyfit(times_array[:mid_end_idx], sizes_array[:mid_end_idx], 1)
                mid_poly = np.poly1d(mid_coeffs)
                plt.plot(times_array[:mid_end_idx], mid_poly(times_array[:mid_end_idx]),
                         'g:', linewidth=2,
                         label=f'中期拟合: y = {mid_coeffs[0]:.4f}x + {mid_coeffs[1]:.2f}')

                # 后期拟合
                late_coeffs = np.polyfit(times_array[mid_end_idx:], sizes_array[mid_end_idx:], 1)
                late_poly = np.poly1d(late_coeffs)
                plt.plot(times_array[mid_end_idx:], late_poly(times_array[mid_end_idx:]),
                         'r:', linewidth=2,
                         label=f'后期拟合: y = {late_coeffs[0]:.4f}x + {late_coeffs[1]:.2f}')

        # 图表装饰
        plt.xlabel('时间（模拟步数）', fontsize=12)
        plt.ylabel('平均晶体粒径（像素）', fontsize=12)
        plt.title('晶体粒径随时间变化及线性拟合分析', fontsize=14, pad=20)
        plt.grid(True, linestyle='--', alpha=0.6)

        # 添加阶段标记
        if 'mid' in analysis_results and 'late' in analysis_results:
            transition_time = analysis_results['mid']['times'][-1]
            plt.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.5)
            plt.text(transition_time, np.max(sizes_array) * 0.95,
                     '中期→后期', ha='center', va='center',
                     backgroundcolor='white', fontsize=10)

        # 添加拟合方程和统计信息
        plt.legend(fontsize=10, loc='upper left')
        plt.tight_layout()

        # 保存
        image_path = os.path.join(image_dir, '晶体粒径随时间变化.svg')
        plt.savefig(image_path, dpi=300, format='svg')

        plt.show()

    # 7. 晶粒个数随时间变化（中期和后期阶段）
    if 'mid' in analysis_results or 'late' in analysis_results:
        plt.figure(figsize=(12, 7))

        # 准备数据
        mid_times = analysis_results['mid']['times'] if 'mid' in analysis_results else []
        mid_counts = analysis_results['mid']['num_grains'] if 'mid' in analysis_results else []
        late_times = analysis_results['late']['times'] if 'late' in analysis_results else []
        late_counts = analysis_results['late']['num_grains'] if 'late' in analysis_results else []

        # 合并数据
        all_times = np.concatenate((mid_times, late_times))
        all_counts = np.concatenate((mid_counts, late_counts))

        # 绘制曲线
        plt.plot(all_times, all_counts, 'co-', linewidth=2, markersize=8,
                 label='晶粒个数', alpha=0.8)

        # 添加趋势线（二次多项式拟合）
        # if len(all_times) > 2:
        #     coeffs = np.polyfit(all_times, all_counts, 2)
        #     poly = np.poly1d(coeffs)
        #     fit_x = np.linspace(min(all_times), max(all_times), 100)
        #     fit_y = poly(fit_x)
        #
        #     # 计算R平方
        #     residuals = all_counts - poly(all_times)
        #     ss_res = np.sum(residuals ** 2)
        #     ss_tot = np.sum((all_counts - np.mean(all_counts)) ** 2)
        #     r_squared = 1 - (ss_res / ss_tot)
        #
        #     plt.plot(fit_x, fit_y, 'm--', linewidth=2,
        #              label=f'趋势线 (R²={r_squared:.3f})')

        # 标记阶段分界
        if 'mid' in analysis_results and 'late' in analysis_results:
            transition_time = mid_times[-1]
            # plt.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.6)
            # plt.text(transition_time, max(all_counts) * 0.9,
            #          '中期→后期', ha='center', va='center',
            #          bbox=dict(facecolor='white', alpha=0.8))

        # 图表装饰
        plt.xlabel('时间（模拟步数）', fontsize=12)
        plt.ylabel('晶粒数量', fontsize=12)
        plt.title('中期和后期晶粒数量随时间变化', fontsize=14, pad=20)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10, loc='upper right')

        # 添加关键点标注
        if len(all_times) > 0:
            plt.annotate(f'初始: {all_counts[0]}个',
                         xy=(all_times[0], all_counts[0]),
                         xytext=(10, 20), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->"))

            plt.annotate(f'最终: {all_counts[-1]}个',
                         xy=(all_times[-1], all_counts[-1]),
                         xytext=(-50, -30), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->"))

        plt.tight_layout()

        # 保存
        image_path = os.path.join(image_dir, '中期和后期晶粒数量随时间变化.svg')
        plt.savefig(image_path, dpi=300, format='svg')

        plt.show()

    # 8. 面积分布分析（智能选择代表性轮次）
    if 'mid' in analysis_results or 'late' in analysis_results:
        # 智能选择分析轮次（选择初期、中期、后期各1个代表性时间点）
        selected_stages = []
        if 'mid' in analysis_results:
            mid_len = len(analysis_results['mid']['times'])
            selected_stages.append(('mid', 0))  # 中期开始
            if mid_len > 1:
                selected_stages.append(('mid', mid_len // 2))  # 中期中间
        if 'late' in analysis_results:
            late_len = len(analysis_results['late']['times'])
            selected_stages.append(('late', late_len // 2))  # 后期中间
            if late_len > 1:
                selected_stages.append(('late', -1))  # 后期结束

        # 准备绘图
        plt.figure(figsize=(14, 8))

        # 计算全局最大面积用于统一x轴范围
        max_area = max(
            max([d['area'] for d in analysis_results[stage]['size_topology_correlation'][idx]]
                for stage, idx in selected_stages)
        )

        # 对每个选中的阶段进行分析
        for i, (stage, idx) in enumerate(selected_stages, 1):
            time_point = analysis_results[stage]['times'][idx]
            size_data = analysis_results[stage]['size_topology_correlation'][idx]
            areas = np.array([d['area'] for d in size_data])
            normalized_areas = areas / np.mean(areas)  # 面积/平均面积

            # 使用Seaborn的分布图（更专业的统计可视化）
            ax = plt.subplot(2, 2, i)
            sns.histplot(normalized_areas, bins=30, kde=True,
                         color=plt.cm.tab10(i - 1), edgecolor='w', linewidth=0.5)

            # 添加统计信息
            median_val = np.median(normalized_areas)
            std_val = np.std(normalized_areas)
            ax.axvline(x=1, color='k', linestyle='--', label='平均值')
            ax.axvline(x=median_val, color='r', linestyle=':', label=f'中位数: {median_val:.2f}')

            # 图表装饰
            stage_name = '中期' if stage == 'mid' else '后期'
            ax.set_title(f'{stage_name}阶段 (t={time_point})\nμ=1.0, σ={std_val:.2f}', fontsize=11)
            ax.set_xlabel('面积/平均面积')
            ax.set_ylabel('频数')
            ax.set_xlim(0, min(3, max_area / np.mean(areas)))  # 限制x轴范围
            ax.legend(fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)

        # 整体标题和布局调整
        plt.suptitle('晶粒面积分布演变（标准化处理）', fontsize=14, y=1.02)
        plt.tight_layout()

        # 保存图像
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, '晶粒面积频数分布.svg')
        plt.savefig(image_path, dpi=300, format='svg', bbox_inches='tight')

        plt.show()


def plot_lewis_law_relationship(size_topology_data, image_dir='images/'):
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
    plt.title('晶粒面积与边数关系')
    plt.legend()
    plt.grid(True)

    # 保存
    image_path = os.path.join(image_dir, '晶粒面积与边数关系.svg')
    plt.savefig(image_path, dpi=300, format='svg')

    plt.show()

def save_results(all_results, filename='grain_analysis_results.pkl'):
    """
    保存分析结果到pickle文件
    :param all_results: 要保存的分析结果字典
    :param filename: 保存文件名，默认'grain_analysis_results.pkl'
    """
    with open(filename, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"分析结果已保存到 {filename}")


def load_results(filename='grain_analysis_results.pkl'):
    """
    从pickle文件加载分析结果
    :param filename: 要加载的文件名，默认'grain_analysis_results.pkl'
    :return: 加载的分析结果字典
    """
    try:
        with open(filename, 'rb') as f:
            loaded_data = pickle.load(f)
        print(f"已从 {filename} 加载分析结果")
        return loaded_data
    except FileNotFoundError:
        print(f"错误：文件 {filename} 不存在")
        return None
    except Exception as e:
        print(f"加载文件时出错: {str(e)}")
        return None

def main():
    data_dir = "data/"
    # 尝试加载已保存结果
    loaded_results = load_results()
    if loaded_results is not None:
        print("使用已保存的分析结果")
        analysis_results = dynamic_analysis(loaded_results)
        plot_dynamic_results(analysis_results)
        return loaded_results

    csv_files = sorted([f for f in os.listdir(data_dir) if f.startswith('time_') and f.endswith('.csv')],
                       key=lambda x: int(x.split('_')[1].split('.')[0]))

    # 1. 根据时间步长范围确定分析间隔
    selected_intervals = get_time_intervals(csv_files)
    # print(f"Selected time intervals for analysis: {selected_intervals}")

    # 2. 加载选定时间点的数据
    all_results = load_selected_data(data_dir, selected_intervals)

    # 3. 动态分析不同阶段
    analysis_results = dynamic_analysis(all_results)

    # 4. 可视化结果
    plot_dynamic_results(analysis_results)

    # 5. 保存结果
    save_results(all_results)

    return analysis_results


if __name__ == "__main__":
    results = main()

    # 1、中/后期轮次?
    # 2、早期是否要画？
    # 3、最小面积阈值？
    # 4、晶粒边数是否遵从某个分布？