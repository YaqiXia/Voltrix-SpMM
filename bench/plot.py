import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set(style="whitegrid")
fontsize = 18
datasets = ['amazon0505', 'DD', 'ppi', 'reddit', 'amazon0601', 'com-amazon', 'ddi', 'FraudYelp-RSR', 'web-BerkStan', 'protein', 'YeastH', 'Yeast']
# 特征维度
featdims = [256, 512, 1024]
# 设置全局字体大小
plt.rc('font', size=14)

# 读取 CSV 文件
df = pd.read_csv('results.csv')


# 获取所有唯一的数据集和特征维度
# datasets = sorted(df['Dataset'].unique())
featdims = sorted(df['FeatDim'].unique())

# 定义方法列表（移除 'PT-Embedding'）
methods = ['cuSPARSE', 'Sputnik', 'GE-SPMM', 'RoDe', 
           'TC-GNN', 'DTC-SPMM', 'Voltrix']
methods_legend = ['cuSPARSE', 'Sputnik-SpMM', 'GE-SpMM', 'RoDe-SpMM', 
           'TC-GNN', 'DTC-SpMM', 'Voltrix-SpMM']
colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))

# 创建一个字典来存储数据
# 数据结构: data[dataset][featdim][method] = time
data = {dataset: {featdim: {method: np.nan for method in methods} 
                 for featdim in featdims} 
        for dataset in datasets}

# 填充数据字典
for _, row in df.iterrows():
    method = row['Method']
    dataset = row['Dataset']
    featdim = row['FeatDim']
    time = row['Time (ms)']
    
    # 只处理定义好的方法和特征维度
    if method in methods and featdim in featdims:
        # 处理 'NAN' 字符串，将其转换为 np.nan
        try:
            time = float(time)
        except ValueError:
            time = np.nan
        data[dataset][featdim][method] = time

# 创建 3x4 的子图布局（因为有12个数据集）
fig, axs = plt.subplots(3, 4, figsize=(25, 11))
axs = axs.flatten()  # 将子图数组展平，方便迭代

# 定义柱状图的宽度
width = 0.1

# 遍历每个数据集
for i, dataset in enumerate(datasets):
    ax = axs[i]
    x = np.arange(len(featdims))  # 特征维度的位置
    
    # 获取 cuSPARSE 在不同特征维度下的时间作为基准
    cusparse_times = [data[dataset][featdim]['cuSPARSE'] for featdim in featdims]
    
    # 绘制每个方法的加速比
    for j, method in enumerate(methods):
        if method == 'cuSPARSE':
            speedups = [1.0 for _ in featdims]  # cuSPARSE 的加速比为1
        else:
            speedups = []
            for k, featdim in enumerate(featdims):
                method_time = data[dataset][featdim][method]
                cusparse_time = cusparse_times[k]
                if pd.isna(method_time) or method_time == 0:
                    speedups.append(np.nan)
                else:
                    speedups.append(cusparse_time / method_time)
        
        # 绘制条形图
        bars = ax.bar(x + j * width, speedups, width, label=methods_legend[j], color = colors[j], edgecolor='black')
        
        # 遍历每个 speedup，检查是否为 NaN
        for idx, speedup in enumerate(speedups):
            if pd.isna(speedup):
                # 获取柱状图的位置
                bar_x = x[idx] + j * width
                bar_y = 0.15  # 设置文本的y位置，确保在 y=0.1 以上
                # 添加 "CUDA ERROR" 文本，竖直显示
                ax.text(bar_x, bar_y, 'CUDA ERROR', rotation=90, 
                        ha='center', va='bottom', fontsize=10, color=colors[j], fontweight='bold')
    
    # 设置标题和标签
    ax.set_title(dataset, fontsize=fontsize)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(featdims, fontsize = fontsize)
    ax.set_ylim(bottom=0.1)  # 避免 log(0) 的问题
    ax.tick_params(axis='y', labelsize=fontsize)
    # if i % 4 == 0:
    #     ax.set_ylabel('Speedup over cuSPARSE', fontsize=14)
    
    # 添加网格线
    ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
    
    # 计算 Voltrix 相对于 cuSPARSE 的加速比范围
    voltrix_speedups = []
    for k, featdim in enumerate(featdims):
        voltrix_time = data[dataset][featdim]['Voltrix']
        cusparse_time = cusparse_times[k]
        if not pd.isna(voltrix_time) and voltrix_time != 0:
            speedup = cusparse_time / voltrix_time
            voltrix_speedups.append(speedup)
    
    if voltrix_speedups:
        min_speedup = np.min(voltrix_speedups)
        max_speedup = np.max(voltrix_speedups)
        speedup_str = f"{min_speedup:.2f}x - {max_speedup:.2f}x speedup"
    else:
        speedup_str = "N/A speed"
    
    # 设置横轴标题为 'Voltrix Speedup Range'
    # 使用换行符实现多行标签
    ax.set_xlabel(f"{speedup_str}", fontsize=fontsize)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# 添加整体的纵轴标签
fig.text(0.045, 0.5, 'Speedup over cuSPARSE', va='center', rotation='vertical', fontsize=fontsize)

# 设置图例（从最后一个子图获取）
handles, labels = axs[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', 
           bbox_to_anchor=(0.52, 0.98), fontsize=fontsize, ncol=7, frameon=False,
            columnspacing=3.0,       # 调整列间距
            handletextpad=0.7,       # 调整符号与文本之间的间距
            handlelength=3.0,          # 调整符号长度)
)

# 调整子图间距
plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

# 显示图表
plt.savefig('results.png', dpi=300, bbox_inches='tight')
plt.show()