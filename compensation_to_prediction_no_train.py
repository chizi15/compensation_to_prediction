import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
import numpy as np
import random
from warnings import filterwarnings
import seaborn as sns

sns.set_style('darkgrid')
plt.rc('font', size=10)
filterwarnings("ignore")

# ###########---------------set up and plot input data-----------------######################

base_value = 10  # 设置level、trend、season项的基数
steps_day, steps_week = 7, 1
length = [steps_day * 5 + steps_day, steps_week * 5 + steps_week]  # 代表每个序列的长度，分别为周、日序列的一年及两年

weights = []
for i in range(-base_value + 1, 1):
    weights.append(0.5 ** i)  # 设置y_level项随机序列的权重呈递减指数分布，底数越小，y_level中较小值所占比例越大。
weights = np.array(weights)

##########################################################--构造加法周期性时间序列，模拟真实销售
# random.seed(0)
# np.random.seed(0)
y_level_actual, y_trend_actual, y_season_actual, y_noise_actual, y_input_add_actual = [[]] * len(length), [[]] * len(
    length), [[]] * len(length), [[]] * len(length), [[]] * len(length)
for i in range(0, len(length)):
    y_season_actual[i] = np.sqrt(base_value) * np.sin(np.linspace(np.pi / 2, 10 * np.pi, length[i]))  # 用正弦函数模拟周期性
    y_season_actual[i] = y_season_actual[i] + max(y_season_actual[i]) + 1  # 使y_season均为正
    y_level_actual[i] = np.array(random.choices(range(0, base_value), weights=weights, k=length[i])) / np.average(
        abs(y_season_actual[i])) + np.average(abs(y_season_actual[i]))  # 用指数权重分布随机数模拟水平项
    y_trend_actual[i] = (2 * max(y_season_actual[i]) + np.log(
        np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)
                         + (min(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)) + max(
                np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)))
                         / length[i] * np.linspace(1, length[i], num=length[i])) / 10 * np.average(
        y_level_actual[i])  # 用对数函数与线性函数的均值模拟趋势性
    y_noise_actual[i] = 3 * np.random.standard_t(length[i] - 1, length[
        i])  # normal(0, 1, length[i])  # 假定数据处于理想状态，并使噪音以加法方式进入模型，则可令噪音在0附近呈学生分布。
    y_noise_actual[i][abs(y_noise_actual[i]) < max(y_noise_actual[i]) * 0.9] = 0  # 只保留随机数中的离群值
    y_input_add_actual[i] = 10 * (
                y_level_actual[i] + y_trend_actual[i] + y_season_actual[i] + y_noise_actual[i])  # 假定各项以加法方式组成输入数据

    print(f'第{i}条真实序列中水平项的极差：{max(y_level_actual[i]) - min(y_level_actual[i])}，均值：{np.mean(y_level_actual[i])}')
    print(f'第{i}条真实序列中趋势项的极差：{max(y_trend_actual[i]) - min(y_trend_actual[i])}，均值：{np.mean(y_trend_actual[i])}')
    print(f'第{i}条真实序列中周期项的极差：{max(y_season_actual[i]) - min(y_season_actual[i])}，均值：{np.mean(y_season_actual[i])}')
    print(f'第{i}条真实序列中噪音项的极差：{max(y_noise_actual[i]) - min(y_noise_actual[i])}，均值：{np.mean(y_noise_actual[i])}')
    print(
        f'第{i}条真实加法性序列最终极差：{max(y_input_add_actual[i]) - min(y_input_add_actual[i])}，均值：{np.mean(y_input_add_actual[i])}',
        '\n')

    y_level_actual[i] = pd.Series(y_level_actual[i]).rename('y_level_actual')
    y_trend_actual[i] = pd.Series(y_trend_actual[i]).rename('y_trend_actual')
    y_season_actual[i] = pd.Series(y_season_actual[i]).rename('y_season_actual')
    y_noise_actual[i] = pd.Series(y_noise_actual[i]).rename('y_noise_actual')
    y_input_add_actual[i] = pd.Series(y_input_add_actual[i]).rename('y_input_add_actual')
    # y_input_add_actual[i][y_input_add_actual[i] < 0] = 0

# # 绘制加法季节性时间序列；xlim让每条折线图填充满x坐标轴
# plt.figure('add_actual_pred: 14+7', figsize=(5, 10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# y_input_add_actual[0].plot(ax=ax1, legend=True)
# y_level_actual[0].plot(ax=ax2, legend=True)
# y_trend_actual[0].plot(ax=ax3, legend=True)
# y_season_actual[0].plot(ax=ax4, legend=True)
# y_noise_actual[0].plot(ax=ax5, legend=True)
#
# plt.figure('add_actual_pred: 4+1', figsize=(5, 10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# y_input_add_actual[1].plot(ax=ax1, legend=True)
# y_level_actual[1].plot(ax=ax2, legend=True)
# y_trend_actual[1].plot(ax=ax3, legend=True)
# y_season_actual[1].plot(ax=ax4, legend=True)
# y_noise_actual[1].plot(ax=ax5, legend=True)

##########################################################--构造乘法周期性时间序列，模拟真实销售
y_level_actual, y_trend_actual, y_season_actual, y_noise_actual, y_input_mul_actual = [[]] * len(length), [[]] * len(
    length), [[]] * len(length), [[]] * len(length), [[]] * len(length)
for i in range(0, len(length)):
    y_season_actual[i] = np.sqrt(base_value) * np.sin(np.linspace(np.pi / 2, 10 * np.pi, length[i]))  # 用正弦函数模拟周期性
    y_season_actual[i] = y_season_actual[i] + max(y_season_actual[i]) + 1  # 使y_season均为正
    y_level_actual[i] = np.array(random.choices(range(0, base_value), weights=weights, k=length[i])) / np.average(
        abs(y_season_actual[i])) + np.average(abs(y_season_actual[i]))  # 用指数权重分布随机数模拟水平项
    y_trend_actual[i] = (2 * max(y_season_actual[i]) + np.log(
        np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)
                         + (min(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)) + max(
                np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)))
                         / length[i] * np.linspace(1, length[i], num=length[i])) / 10 * np.average(
        y_level_actual[i])  # 用对数函数与线性函数的均值模拟趋势性
    y_noise_actual[i] = 3 * np.random.standard_t(length[i] - 1, length[i])  # 假定数据处于理想状态，并使噪音以加法方式进入模型，则可令噪音在0附近呈学生分布。
    y_noise_actual[i][abs(y_noise_actual[i]) < max(y_noise_actual[i]) * 0.9] = 1  # 保留随机数中的离群值，将非离群值置为1
    y_input_mul_actual[i] = (y_level_actual[i] + y_trend_actual[i]) * y_season_actual[i] * abs(
        y_noise_actual[i])  # 假定周期项以乘法方式组成输入数据

    print(f'第{i}条真实序列中水平项的极差：{max(y_level_actual[i]) - min(y_level_actual[i])}，均值：{np.mean(y_level_actual[i])}')
    print(f'第{i}条真实序列中趋势项的极差：{max(y_trend_actual[i]) - min(y_trend_actual[i])}，均值：{np.mean(y_trend_actual[i])}')
    print(f'第{i}条真实序列中周期项的极差：{max(y_season_actual[i]) - min(y_season_actual[i])}，均值：{np.mean(y_season_actual[i])}')
    print(f'第{i}条真实序列中噪音项的极差：{max(y_noise_actual[i]) - min(y_noise_actual[i])}，均值：{np.mean(y_noise_actual[i])}')
    print(
        f'第{i}条真实乘法性序列最终极差：{max(y_input_mul_actual[i]) - min(y_input_mul_actual[i])}，均值：{np.mean(y_input_mul_actual[i])}',
        '\n')

    y_level_actual[i] = pd.Series(y_level_actual[i]).rename('y_level_actual')
    y_trend_actual[i] = pd.Series(y_trend_actual[i]).rename('y_trend_actual')
    y_season_actual[i] = pd.Series(y_season_actual[i]).rename('y_season_actual')
    y_noise_actual[i] = pd.Series(y_noise_actual[i]).rename('y_noise_actual')
    y_input_mul_actual[i] = pd.Series(y_input_mul_actual[i]).rename('y_input_mul_actual')
    # y_input_mul_actual[i][y_input_mul_actual[i] < 0] = 0

# # 绘制四条乘法季节性时间序列；xlim让每条折线图填充满x坐标轴
# plt.figure('mul_actual_pred: 14+7', figsize=(5,10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# y_input_mul_actual[0].plot(ax=ax1, legend=True)
# y_level_actual[0].plot(ax=ax2, legend=True)
# y_trend_actual[0].plot(ax=ax3, legend=True)
# y_season_actual[0].plot(ax=ax4, legend=True)
# y_noise_actual[0].plot(ax=ax5, legend=True)
#
# plt.figure('mul_actual_pred: 4+1', figsize=(5,10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# y_input_mul_actual[1].plot(ax=ax1, legend=True)
# y_level_actual[1].plot(ax=ax2, legend=True)
# y_trend_actual[1].plot(ax=ax3, legend=True)
# y_season_actual[1].plot(ax=ax4, legend=True)
# y_noise_actual[1].plot(ax=ax5, legend=True)


##########################################################--构造加法周期性时间序列，模拟预测销售
# random.seed(0)
# np.random.seed(0)
y_level_pred, y_trend_pred, y_season_pred, y_noise_pred, y_input_add_pred = [[]] * len(length), [[]] * len(length), [
    []] * len(length), [[]] * len(length), [[]] * len(length)
for i in range(0, len(length)):
    y_season_pred[i] = 1 / 2 * np.sqrt(base_value) * np.sin(
        np.linspace(np.pi / 2, 10 * np.pi, length[i]))  # 用正弦函数模拟周期性，使预测销售的波动振幅比真实销售小
    y_season_pred[i] = y_season_pred[i] + max(y_season_pred[i]) + 1  # 使y_season_pred均为正
    y_level_pred[i] = np.array(random.choices(range(0, base_value), weights=weights, k=length[i])) / np.average(
        abs(y_season_pred[i])) + np.average(abs(y_season_pred[i])) + np.random.randint(
        -np.average(abs(y_season_pred[i])), np.average(abs(y_season_pred[i])))  # 用指数权重分布随机数模拟水平项，使其相对于真实销售有所偏移
    y_trend_pred[i] = (2 * max(y_season_pred[i]) + np.log(
        np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)
                       + (min(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)) + max(
                np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)))
                       / length[i] * np.linspace(1, length[i], num=length[i])) / 10 * np.average(
        y_level_pred[i])  # 用对数函数与线性函数的均值模拟趋势性
    y_noise_pred[i] = np.random.standard_t(length[i] - 1, length[
        i])  # normal(0, 1, length[i])  # 假定数据处于理想状态，并使噪音以加法方式进入模型，则可令噪音在0附近呈学生分布。使其比真实销售的噪音小。
    y_noise_pred[i][abs(y_noise_pred[i]) < max(y_noise_pred[i]) * 0.9] = 0  # 只保留随机数中的离群值
    y_input_add_pred[i] = 10 * (
                y_level_pred[i] + y_trend_pred[i] + y_season_pred[i] + y_noise_pred[i])  # 假定各项以加法方式组成输入数据

    print(f'第{i}条预测序列中水平项的极差：{max(y_level_pred[i]) - min(y_level_pred[i])}，均值：{np.mean(y_level_pred[i])}')
    print(f'第{i}条预测序列中趋势项的极差：{max(y_trend_pred[i]) - min(y_trend_pred[i])}，均值：{np.mean(y_trend_pred[i])}')
    print(f'第{i}条预测序列中周期项的极差：{max(y_season_pred[i]) - min(y_season_pred[i])}，均值：{np.mean(y_season_pred[i])}')
    print(f'第{i}条预测序列中噪音项的极差：{max(y_noise_pred[i]) - min(y_noise_pred[i])}，均值：{np.mean(y_noise_pred[i])}')
    print(f'第{i}条预测加法性序列最终极差：{max(y_input_add_pred[i]) - min(y_input_add_pred[i])}，均值：{np.mean(y_input_add_pred[i])}',
          '\n')

    y_level_pred[i] = pd.Series(y_level_pred[i]).rename('y_level_pred')
    y_trend_pred[i] = pd.Series(y_trend_pred[i]).rename('y_trend_pred')
    y_season_pred[i] = pd.Series(y_season_pred[i]).rename('y_season_pred')
    y_noise_pred[i] = pd.Series(y_noise_pred[i]).rename('y_noise_pred')
    y_input_add_pred[i] = pd.Series(y_input_add_pred[i]).rename('y_input_add_pred')
    # y_input_add_pred[i][y_input_add_pred[i] < 0] = 0

# # 绘制加法季节性时间序列；xlim让每条折线图填充满x坐标轴
# plt.figure('add_actual_pred: 14+7', figsize=(5, 10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# y_input_add_pred[0].plot(ax=ax1, legend=True)
# y_level_pred[0].plot(ax=ax2, legend=True)
# y_trend_pred[0].plot(ax=ax3, legend=True)
# y_season_pred[0].plot(ax=ax4, legend=True)
# y_noise_pred[0].plot(ax=ax5, legend=True)
#
# plt.figure('add_actual_pred: 4+1', figsize=(5, 10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# y_input_add_pred[1].plot(ax=ax1, legend=True)
# y_level_pred[1].plot(ax=ax2, legend=True)
# y_trend_pred[1].plot(ax=ax3, legend=True)
# y_season_pred[1].plot(ax=ax4, legend=True)
# y_noise_pred[1].plot(ax=ax5, legend=True)

##########################################################--构造乘法周期性时间序列，模拟预测销售
y_level_pred, y_trend_pred, y_season_pred, y_noise_pred, y_input_mul_pred = [[]] * len(length), [[]] * len(length), [
    []] * len(length), [[]] * len(length), [[]] * len(length)
for i in range(0, len(length)):
    y_season_pred[i] = 1 / 2 * np.sqrt(base_value) * np.sin(
        np.linspace(np.pi / 2, 10 * np.pi, length[i]))  # 用正弦函数模拟周期性，使预测销售的波动振幅比真实销售小
    y_season_pred[i] = y_season_pred[i] + max(y_season_pred[i]) + 1  # 使y_season_pred均为正
    y_level_pred[i] = np.array(random.choices(range(0, base_value), weights=weights, k=length[i])) / np.average(
        abs(y_season_pred[i])) + np.average(abs(y_season_pred[i])) + np.random.randint(
        -np.average(abs(y_season_pred[i])), np.average(abs(y_season_pred[i])))  # 用指数权重分布随机数模拟水平项，使其相对于真实销售有所偏移
    y_trend_pred[i] = (2 * max(y_season_pred[i]) + np.log(
        np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)
                       + (min(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)) + max(
                np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)))
                       / length[i] * np.linspace(1, length[i], num=length[i])) / 10 * np.average(
        y_level_pred[i])  # 用对数函数与线性函数的均值模拟趋势性
    y_noise_pred[i] = np.random.standard_t(length[i] - 1,
                                           length[i])  # 假定数据处于理想状态，并使噪音以加法方式进入模型，则可令噪音在0附近呈学生分布；使其比真实销售的噪音小。
    y_noise_pred[i][abs(y_noise_pred[i]) < max(y_noise_pred[i]) * 0.9] = 1  # 保留随机数中的离群值，将非离群值置为1
    y_input_mul_pred[i] = (y_level_pred[i] + y_trend_pred[i]) * y_season_pred[i] * abs(
        y_noise_pred[i])  # 假定周期项以乘法方式组成输入数据

    print(f'第{i}条预测序列中水平项的极差：{max(y_level_pred[i]) - min(y_level_pred[i])}，均值：{np.mean(y_level_pred[i])}')
    print(f'第{i}条预测序列中趋势项的极差：{max(y_trend_pred[i]) - min(y_trend_pred[i])}，均值：{np.mean(y_trend_pred[i])}')
    print(f'第{i}条预测序列中周期项的极差：{max(y_season_pred[i]) - min(y_season_pred[i])}，均值：{np.mean(y_season_pred[i])}')
    print(f'第{i}条预测序列中噪音项的极差：{max(y_noise_pred[i]) - min(y_noise_pred[i])}，均值：{np.mean(y_noise_pred[i])}')
    print(f'第{i}条预测乘法性序列最终极差：{max(y_input_mul_pred[i]) - min(y_input_mul_pred[i])}，均值：{np.mean(y_input_mul_pred[i])}',
          '\n')

    y_level_pred[i] = pd.Series(y_level_pred[i]).rename('y_level_pred')
    y_trend_pred[i] = pd.Series(y_trend_pred[i]).rename('y_trend_pred')
    y_season_pred[i] = pd.Series(y_season_pred[i]).rename('y_season_pred')
    y_noise_pred[i] = pd.Series(y_noise_pred[i]).rename('y_noise_pred')
    y_input_mul_pred[i] = pd.Series(y_input_mul_pred[i]).rename('y_input_mul_pred')
    # y_input_mul_pred[i][y_input_mul_pred[i] < 0] = 0


# # 绘制四条乘法季节性时间序列；xlim让每条折线图填充满x坐标轴
# plt.figure('mul_actual_pred: 14+7', figsize=(5,10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[0]-1)
# y_input_mul_pred[0].plot(ax=ax1, legend=True)
# y_level_pred[0].plot(ax=ax2, legend=True)
# y_trend_pred[0].plot(ax=ax3, legend=True)
# y_season_pred[0].plot(ax=ax4, legend=True)
# y_noise_pred[0].plot(ax=ax5, legend=True)
#
# plt.figure('mul_actual_pred: 4+1', figsize=(5,10))
# ax1 = plt.subplot(5,1,1)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax2 = plt.subplot(5,1,2)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax3 = plt.subplot(5,1,3)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax4 = plt.subplot(5,1,4)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# ax5 = plt.subplot(5,1,5)
# xlim = plt.gca().set_xlim(0, length[1]-1)
# y_input_mul_pred[1].plot(ax=ax1, legend=True)
# y_level_pred[1].plot(ax=ax2, legend=True)
# y_trend_pred[1].plot(ax=ax3, legend=True)
# y_season_pred[1].plot(ax=ax4, legend=True)
# y_noise_pred[1].plot(ax=ax5, legend=True)


def compensation(hist_actual, hist_pred, pred, j, period=7, steps_day=7):
    """
    :param hist_actual: 近期理论销售历史值，任何一维数据，转化为pd.Series，为了作图观察
    :param hist_pred: 近期倒数第二次预测值，加载老r、效应系数、新r，但不加载补偿量，任何一维数据，转化为pd.Series，为了作图观察
    :param pred: 未来7天的倒数第二次预测值，加载老r、效应系数、新r，但不加载补偿量，任何一维数据，转化为pd.Series，为了作图观察
    :param j: 输入的测试数据的索引号，是第几组
    :param period: HoltWinter方法的周期
    :param steps_day: 预测天数
    :return: 加载补偿量的最终预测值，np.array，使其无索引
    """
    hist_actual = pd.Series(hist_actual[:29])
    hist_pred = pd.Series(hist_pred[:29])
    pred = pd.Series(pred[:steps_day])
    deviation = hist_pred - hist_actual
    weights = []
    for i in range(1, len(deviation) + 1):
        weights.append(i / len(deviation))
    weights = np.array(weights) / sum(weights)
    if len(deviation) <= 3:
        print('历史数据过少，不进行偏差量补偿')
        final_pred = pred
    elif 4 <= len(deviation) <= 14:
        # SES
        # 拟合历史数据
        fit_SES = SimpleExpSmoothing(deviation, initialization_method='known', initial_level=np.average(deviation,
            weights=weights)).fit(smoothing_level=1 / 10, optimized=False, start_params=1/10, use_brute=False, use_boxcox=False, remove_bias=False)
        fit_SES_train = SimpleExpSmoothing(deviation, initialization_method='known', initial_level=np.average(deviation,
            weights=weights)).fit(smoothing_level=1 / 5, optimized=False, start_params=1/5, use_brute=False, use_boxcox=False, remove_bias=False)
        # 打印模型参数
        print('SimpleExpSmoothing_[4,14]:')
        results = pd.DataFrame(index=[r"$\alpha$", r"$l_0$", "SSE"])
        params = ['smoothing_level', 'initial_level']
        results["fit_SES"] = [fit_SES.params[p] for p in params] + [fit_SES.sse]
        results["fit_SES_train"] = [fit_SES_train.params[p] for p in params] + [fit_SES_train.sse]
        print(results, '\n')
        # 对各模型拟合值及预测值绘图，并作比较
        plt.figure('SES_[4,14]', figsize=(20, 10))
        ax_SES = deviation.rename('deviation').plot(color='k', legend=True)
        ax_SES.set_ylabel("amount")
        ax_SES.set_xlabel("day")
        xlim = plt.gca().set_xlim(0, length[j] - 1)
        fit_SES.fittedvalues.plot(ax=ax_SES, color='b')
        fit_SES_train.fittedvalues.plot(ax=ax_SES, color='r')
        fit_SES.forecast(steps_day).rename(r'$\alpha=%s$' % fit_SES.model.params['smoothing_level']).plot(
            ax=ax_SES, color='b', legend=True)
        fit_SES_train.forecast(steps_day).rename(r'$\alpha=%s$' % fit_SES_train.model.params['smoothing_level']).plot(
            ax=ax_SES, color='r', legend=True)

        # Holt
        # fit models
        Holt_add_dam = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 10,
            smoothing_trend=1 / 10, damping_trend=0.90, optimized=False, start_params=[1/10, 1/10], use_brute=False,
            use_boxcox=False, remove_bias=False)
        Holt_add_dam_train = Holt(deviation, exponential=False, damped=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 5,
                                                                                 smoothing_trend=1 / 5,
                                                                                 damping_trend=0.85, optimized=False,
        start_params = [1 / 5, 1 / 5], use_brute = False, use_boxcox = False, remove_bias = False)
        # print parameters
        print('Hlot_[4,14]:')
        results = pd.DataFrame(index=[r"$\alpha$", r"$\beta$", r"$\phi$", r"$l_0$", "$b_0$", "SSE"])
        params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
        results["Holt_add_dam"] = [Holt_add_dam.params[p] for p in params] + [Holt_add_dam.sse]
        results["Holt_add_dam_train"] = [Holt_add_dam_train.params[p] for p in params] + [Holt_add_dam_train.sse]
        print(results, '\n')
        # print figures
        plt.figure('Hlot_[4,14]', figsize=(20, 10))
        ax_Holt = deviation.rename('deviation').plot(color='black', legend=True)
        ax_Holt.set_ylabel("amount")
        ax_Holt.set_xlabel("day")
        xlim = plt.gca().set_xlim(0, length[j] - 1)
        Holt_add_dam.fittedvalues.plot(ax=ax_Holt, color='blue')
        Holt_add_dam_train.fittedvalues.plot(ax=ax_Holt, color='red')
        Holt_add_dam.forecast(steps_day).rename('Holt_add_dam').plot(ax=ax_Holt, color='b', legend=True)
        Holt_add_dam_train.forecast(steps_day).rename('Holt_add_dam_train').plot(ax=ax_Holt, color='red', legend=True)

        # 总补偿量为正，则按预测值的正比重新分配每步补偿量，总补偿量为负，则按预测值的反比重新分配每步补偿量；使补偿后的预测值更加稳定，并在多步补偿的情况下与实际值更接近。
        pred = np.array(pred)  # pd.Series是按对应索引进行运算，np.array是按对应顺序进行运算
        ratio_posi = pred / sum(pred)
        ratio_posi[ratio_posi < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_posi = ratio_posi / sum(ratio_posi)
        ratio_neg = 1 / 2 * (1 - ratio_posi) / sum(1 - ratio_posi) + 1 / 2 * (1 / ratio_posi) / sum(1 / ratio_posi)
        ratio_neg[ratio_neg < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_neg = ratio_neg / sum(ratio_neg)
        total_compensation = sum(4 / 5 * np.average(
            np.column_stack([np.array(Holt_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                Holt_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1) + 1 / 5 * np.average(
            np.column_stack([np.array(fit_SES.forecast(steps_day)).reshape(-1, 1), np.array(
                fit_SES_train.forecast(steps_day)).reshape(-1, 1)]), axis=1))
        if total_compensation >= 0:
            every_compensation = total_compensation * ratio_posi
        else:
            every_compensation = total_compensation * ratio_neg
        final_pred = pred - every_compensation

        # 检查最终预测值
        print('sum of ratio_posi and ratio_neg:', sum(ratio_posi), sum(ratio_neg), '\n')
        print('Holt_add_dam.forecast(steps_day):', '\n', Holt_add_dam.forecast(steps_day), '\n')
        print('Holt_add_dam_train.forecast(steps_day):', '\n', Holt_add_dam_train.forecast(steps_day), '\n')
        print('fit_SES.forecast(steps_day):', '\n', fit_SES.forecast(steps_day), '\n')
        print('fit_SES_train.forecast(steps_day):', '\n', fit_SES_train.forecast(steps_day), '\n')
        print('average compensation of every step:', total_compensation / 7, '\n')
        print('every_compensation:', '\n', every_compensation, '\n')

        # 对最终预测值作图
        plt.figure('[4, 14]_final_pred', figsize=(20, 10))
        ax_SESHolt = pd.Series(final_pred, index=range(len(deviation), len(deviation) + steps_day)).rename(
            'final_pred').plot(color='k', legend=True)
        ax_SESHolt.set_ylabel("amount")
        ax_SESHolt.set_xlabel("day")
        xlim = plt.gca().set_xlim(0, length[j] - 1)
        pd.Series(pred, index=range(len(deviation), len(deviation) + steps_day)).rename('pred').plot(ax=ax_SESHolt,
                                                                                                     color='r',
                                                                                                     legend=True)
        hist_pred.rename('hist_pred').plot(ax=ax_SESHolt, color='r', legend=True)
        hist_actual.rename('hist_actual').plot(ax=ax_SESHolt, color='b', legend=True)
        pd.Series(every_compensation, index=range(len(deviation), len(deviation) + steps_day)).rename(
            'every_compensation').plot(ax=ax_SESHolt,
                                       color='g', legend=True)

    else:
        # Holt
        # fit models
        Holt_add_dam = Holt(deviation, exponential=False, damped_trend=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 10,
            smoothing_trend=1 / 10, damping_trend=0.90, optimized=False, start_params=[1/10, 1/10], use_brute=False,
            use_boxcox=False, remove_bias=False)
        Holt_add_dam_train = Holt(deviation, exponential=False, damped=True, initialization_method='known',
                            initial_level=np.average(deviation,
                                                     weights=weights),
                            initial_trend=np.array((sum(deviation[
                                                        int(
                                                            np.ceil(
                                                                len(
                                                                    deviation) / 2)):]) - sum(
                                deviation[:int(np.floor(
                                    len(deviation) / 2))])) / (
                                                       np.floor(
                                                           len(
                                                               deviation) / 2)) ** 2)).fit(smoothing_level=1 / 5,
                                                                                 smoothing_trend=1 / 5,
                                                                                 damping_trend=0.85, optimized=False,
        start_params = [1 / 5, 1 / 5], use_brute = False, use_boxcox = False, remove_bias = False)
        # print parameters
        print('Hlot_[15, 29]:')
        results = pd.DataFrame(index=[r"$\alpha$", r"$\beta$", r"$\phi$", r"$l_0$", "$b_0$", "SSE"])
        params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
        results["Holt_add_dam"] = [Holt_add_dam.params[p] for p in params] + [Holt_add_dam.sse]
        results["Holt_add_dam_train"] = [Holt_add_dam_train.params[p] for p in params] + [Holt_add_dam_train.sse]
        print(results, '\n')
        # print figures
        plt.figure('Hlot_[15, 29]', figsize=(20, 10))
        ax_Holt = deviation.rename('deviation').plot(color='black', legend=True)
        ax_Holt.set_ylabel("amount")
        ax_Holt.set_xlabel("day")
        xlim = plt.gca().set_xlim(0, length[j] - 1)
        Holt_add_dam.fittedvalues.plot(ax=ax_Holt, color='blue')
        Holt_add_dam_train.fittedvalues.plot(ax=ax_Holt, color='red')
        Holt_add_dam.forecast(steps_day).rename('Holt_add_dam').plot(ax=ax_Holt, color='b', legend=True)
        Holt_add_dam_train.forecast(steps_day).rename('Holt_add_dam_train').plot(ax=ax_Holt, color='r', legend=True)

        # HoltWinters
        # fit models
        HW_add_add_dam = ExponentialSmoothing(deviation, trend='add', damped_trend=True,
                                              seasonal='add', seasonal_periods=period, initialization_method='known',
                                              initial_level=np.average(deviation, weights=weights),
                                              initial_trend=np.array((sum(
                                                  deviation[int(np.ceil(len(deviation) / 2)):]) - sum(
                                                  deviation[:int(np.floor(len(deviation) / 2))])) / (
                                                                         np.floor(len(deviation) / 2)) ** 2),
                                              initial_seasonal=np.array(
                                                  deviation[:len(deviation) // period * period]).reshape(-1,
                                                                                                         period).mean(
                                                  axis=0) - np.average(deviation, weights=weights), use_boxcox=False, missing='raise'). \
            fit(smoothing_level=1 / 10, smoothing_trend=1 / 10, smoothing_seasonal=1 / 2,
                damping_trend=0.90, optimized=False, remove_bias=False, start_params=[1/10, 1/10, 1/2], use_brute=False)
        HW_add_add_dam_train = ExponentialSmoothing(deviation, trend='add', damped_trend=True, seasonal='add',
                                                    seasonal_periods=period, initialization_method='known',
                                                    initial_level=np.average(deviation, weights=weights),
                                                    initial_trend=np.array((sum(
                                                        deviation[int(np.ceil(len(deviation) / 2)):]) - sum(
                                                        deviation[:int(np.floor(len(deviation) / 2))])) / (
                                                                               np.floor(len(deviation) / 2)) ** 2),
                                                    initial_seasonal=np.array(
                                                        deviation[:len(deviation) // period * period]).reshape(-1,
                                                                                                               period).mean(
                                                        axis=0) - np.average(deviation, weights=weights), use_boxcox=False, missing='raise'). \
            fit(smoothing_level=1 / 5, smoothing_trend=1 / 10, smoothing_seasonal=1 / 10,
                damping_trend=0.85, optimized=False, remove_bias=False, start_params=[1/5, 1/10, 1/10], use_brute=False)

        # print figures
        plt.figure('HoltWinters_[15, 29]', figsize=(20, 10))
        ax_HoltWinters = deviation.rename('deviation').plot(color='k', legend='True')
        ax_HoltWinters.set_ylabel("amount")
        ax_HoltWinters.set_xlabel("day")
        xlim = plt.gca().set_xlim(0, length[j] - 1)
        HW_add_add_dam.fittedvalues.plot(ax=ax_HoltWinters, color='b')
        HW_add_add_dam_train.fittedvalues.plot(ax=ax_HoltWinters, color='r')
        HW_add_add_dam.forecast(steps_day).rename('HW_add_add_dam').plot(ax=ax_HoltWinters, color='b', legend=True)
        HW_add_add_dam_train.forecast(steps_day).rename('HW_add_add_dam_train').plot(ax=ax_HoltWinters, color='r',
                                                                                     legend=True)

        # print parameters
        print('HoltWinters_[15, 29]')
        results = pd.DataFrame(index=["alpha", "beta", "phi", "gamma", "l_0", "b_0", "SSE"])
        params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'smoothing_seasonal', 'initial_level',
                  'initial_trend']
        results["HW_add_add_dam"] = [HW_add_add_dam.params[p] for p in params] + [HW_add_add_dam.sse]
        results["HW_add_add_dam_train"] = [HW_add_add_dam_train.params[p] for p in params] + [HW_add_add_dam_train.sse]
        print(results, '\n')

        # print internals
        df_HW_add_add_dam = pd.DataFrame(np.c_[
                                             deviation, HW_add_add_dam.level, HW_add_add_dam.slope, HW_add_add_dam.season, HW_add_add_dam.fittedvalues],
                                         columns=['y_t', 'l_t', 'b_t', 's_t', 'yhat_t'],
                                         index=deviation.index)
        print('internal items of HoltWinters_[15, 29] HW_add_add_dam are:')
        print(df_HW_add_add_dam, '\n')
        HW_add_add_dam_residual = (np.array(HW_add_add_dam.forecast(steps_day)) - np.array(pred)) / np.array(
            pred) * 100  # pd.Series是按对应索引进行运算，np.array是按对应顺序进行运算
        print('forecast and actual deviation ratio(%) of HoltWinters_[15, 29] HW_add_add_dam is:')
        print(HW_add_add_dam_residual, '\n')

        df_HW_add_add_dam_train = pd.DataFrame(np.c_[
                                                   deviation, HW_add_add_dam_train.level, HW_add_add_dam_train.slope, HW_add_add_dam_train.season, HW_add_add_dam_train.fittedvalues],
                                               columns=['y_t', 'l_t', 'b_t', 's_t', 'yhat_t'],
                                               index=deviation.index)
        print('internal items of HoltWinters_[15, 29] HW_add_add_dam_train are:')
        print(df_HW_add_add_dam_train, '\n')
        HW_add_add_dam_train_residual = (np.array(HW_add_add_dam_train.forecast(steps_day)) - np.array(
            pred)) / np.array(pred) * 100  # pd.Series是按对应索引进行运算，np.array是按对应顺序进行运算
        print('forecast and actual deviation ratio(%) of HoltWinters_[15, 29] HW_add_add_dam_train is:')
        print(HW_add_add_dam_train_residual, '\n')

        # 总补偿量为正，则按预测值的正比重新分配每步补偿量，总补偿量为负，则按预测值的反比重新分配每步补偿量；使补偿后的预测值更加稳定，并在多步补偿的情况下与实际值更接近。
        pred = np.array(pred)  # pd.Series是按对应索引进行运算，np.array是按对应顺序进行运算
        ratio_posi = pred / sum(pred)
        ratio_posi[ratio_posi < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_posi = ratio_posi / sum(ratio_posi)
        ratio_neg = 1 / 2 * (1 - ratio_posi) / sum(1 - ratio_posi) + 1 / 2 * (1 / ratio_posi) / sum(1 / ratio_posi)
        ratio_neg[ratio_neg < 1 / (7 * 3)] = 1 / (7 * 3)
        ratio_neg = ratio_neg / sum(ratio_neg)
        total_compensation = sum(1 / 2 * np.average(
            np.column_stack([np.array(Holt_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                Holt_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1) + 1 / 2 * np.average(
            np.column_stack([np.array(HW_add_add_dam.forecast(steps_day)).reshape(-1, 1), np.array(
                HW_add_add_dam_train.forecast(steps_day)).reshape(-1, 1)]), axis=1))
        if total_compensation >= 0:
            every_compensation = total_compensation * ratio_posi
        else:
            every_compensation = total_compensation * ratio_neg
        final_pred = pred - every_compensation

        print('sum of ratio_posi and ratio_neg:', sum(ratio_posi), sum(ratio_neg), '\n')
        print('Holt_add_dam.forecast(steps_day):', '\n', Holt_add_dam.forecast(steps_day), '\n')
        print('Holt_add_dam_train.forecast(steps_day):', '\n', Holt_add_dam_train.forecast(steps_day), '\n')
        print('HW_add_add_dam.forecast(steps_day):', '\n', HW_add_add_dam.forecast(steps_day), '\n')
        print('HW_add_add_dam_train.forecast(steps_day):', '\n', HW_add_add_dam_train.forecast(steps_day), '\n')
        print('average compensation of every step:', total_compensation / 7, '\n')
        print('every_compensation:', '\n', every_compensation, '\n')

        plt.figure('[15, 29]_final_pred', figsize=(20, 10))
        ax_HoltWinters = pd.Series(final_pred, index=range(len(deviation), len(deviation) + steps_day)).rename(
            'final_pred').plot(color='k', legend=True)
        ax_HoltWinters.set_ylabel("amount")
        ax_HoltWinters.set_xlabel("day")
        xlim = plt.gca().set_xlim(0, length[j] - 1)
        pd.Series(pred, index=range(len(deviation), len(deviation) + steps_day)).rename('pred').plot(ax=ax_HoltWinters,
                                                                                                     color='r',
                                                                                                     legend=True)
        hist_pred.rename('hist_pred').plot(ax=ax_HoltWinters, color='r', legend=True)
        hist_actual.rename('hist_actual').plot(ax=ax_HoltWinters, color='b', legend=True)
        pd.Series(every_compensation, index=range(len(deviation), len(deviation) + steps_day)).rename(
            'every_compensation').plot(ax=ax_HoltWinters,
                                       color='g', legend=True)

    return final_pred


if __name__ == "__main__":
    for i in range(len(length) - 1):
        final_pred_add = compensation(hist_actual=list(y_input_add_actual[i][:3]),
                                      hist_pred=np.array(y_input_add_pred[i][:3]),
                                      pred=tuple(y_input_add_pred[i][3:3 + steps_day]), j=i)
        print('final prediction:')
        print(final_pred_add, '\n')
        final_pred_add = compensation(hist_actual=y_input_add_actual[i][:14], hist_pred=y_input_add_pred[i][:14],
                                      pred=y_input_add_pred[i][14:14 + steps_day], j=i)
        print('final prediction:')
        print(final_pred_add, '\n')
        # final_pred_add = compensation(hist_actual=y_input_add_actual[i][:29], hist_pred=y_input_add_pred[i][:29],
        #                               pred=y_input_add_pred[i][29:29 + steps_day], j=i)
        # print('final prediction:')
        # print(final_pred_add, '\n')
        final_pred_add = compensation(hist_actual=y_input_add_actual[i][:-steps_day],
                                      hist_pred=y_input_add_pred[i][:-steps_day],
                                      pred=y_input_add_pred[i][-steps_day:], j=i)
        print('final prediction:')
        print(final_pred_add, '\n')

        # final_pred_mul = compensation(hist_actual=y_input_mul_actual[i][:3], hist_pred=y_input_mul_pred[i][:3],
        #                               pred=y_input_mul_pred[i][3:3 + steps_day], j=i)
        # print('final prediction:')
        # print(final_pred_mul, '\n')
        # final_pred_mul = compensation(hist_actual=y_input_mul_actual[i][:14], hist_pred=y_input_mul_pred[i][:14],
        #                               pred=y_input_mul_pred[i][14:14 + steps_day], j=i)
        # print('final prediction:')
        # print(final_pred_mul, '\n')
        # final_pred_mul = compensation(hist_actual=y_input_mul_actual[i][:29], hist_pred=y_input_mul_pred[i][:29],
        #                               pred=y_input_mul_pred[i][29:29 + steps_day], j=i)
        # print('final prediction:')
        # print(final_pred_mul, '\n')
        # final_pred_mul = compensation(hist_actual=y_input_mul_actual[i][:30], hist_pred=y_input_mul_pred[i][:30],
        #                               pred=y_input_mul_pred[i][30:30 + steps_day], j=i)
        # print('final prediction:')
        # print(final_pred_mul, '\n')
