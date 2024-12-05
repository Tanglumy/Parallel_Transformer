import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
df = pd.read_csv('timings.csv')

# 排除总执行时间
df_filtered = df[df['Module'] != 'Total Execution']

# 设置图形大小
plt.figure(figsize=(10, 6))

# 生成条形图
plt.bar(df_filtered['Module'], df_filtered['Time_ms'], color='skyblue')

# 添加标题和标签
plt.title('Transformer Forward Pass Execution Timings')
plt.xlabel('Module')
plt.ylabel('Time (ms)')

# 显示数值标签
for index, value in enumerate(df_filtered['Time_ms']):
    plt.text(index, value + 0.5, f"{value:.3f}", ha='center')

# 保存图像
plt.savefig('timings.png')

# 显示图像
plt.show()