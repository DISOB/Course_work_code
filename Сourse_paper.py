import xlrd
import math
import pandas as pd
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline


# Функция вывода DataFrame без индексов
def pprint_df(dframe):
    print(tabulate(dframe, headers='keys', tablefmt='psql', showindex=False))


# функция вывода DataFrame с корреляциями между активами
def pprint_df_corr(dframe):
    print(tabulate(dframe, headers='keys', tablefmt='psql', showindex=True))


# Импортируйте файл Excel и назовите его xls_file
excel_file = pd.ExcelFile('Data_coursework.xlsx')

# Просмотр имен листов excel_file
# print(excel_file.sheet_names)

# Загрузите лист1 excel_file в качестве фрейма данных
df = excel_file.parse('Лист1')

# only read specific columns from an excel file
# require_cols = [0, 1, 2, 3, 4, 5, 6]
# required_df = pd.read_excel('Data_coursework.xlsx', usecols=require_cols)
# required_df = required_df.head(8)
# print(required_df.to_string(index=False))

file = pd.ExcelFile('Data_coursework.xlsx')
new_df = file.parse('Лист1')
new_df = new_df.iloc[1:-1, :]
print(new_df.shape)


# Нахождение лог-доходности акций за каждый день периода
for i in range(0, len(new_df)):
    for j in range(1, len(new_df.columns)):
        new_df.iat[i, j] = float(math.log(float(df.iat[i+1, j]) / float(df.iat[i, j])))


# Вывод DateFrame с лог-доходностями
df_for_print = new_df.iloc[:8, :6]
pprint_df(df_for_print)

print("\n\n\n")

# Используется для установки максимального количества строк, которое будет отображаться при печати dataframe
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# Метод для увеличения выводимой ширины фрейма
pd.set_option('display.width', 200)

# Находим корреляции между активами
corr_df = new_df.drop('Date', axis=1)
correlation_matrix = corr_df.corr()
Names = list(correlation_matrix.columns)

pprint_df_corr(correlation_matrix.iloc[:5, :5])

# ------------------------------------------------------------------------------------

# Нахождение максимальных клик графа

# import networkx as nx
# G = nx.Graph(correlation_matrix)
# print(len(max(nx.algorithms.clique.find_cliques(G), key = len)))

point = 0.85
res_name = set()

for i in range(0, len(correlation_matrix)):
    for j in range(0, len(correlation_matrix.columns)):
        if i != j and correlation_matrix.iat[i, j] >= point:
            res_name.add(Names[i])
            res_name.add(Names[j])


print('\n', res_name)
print('\n', len(res_name))


# ------------------------------------------------------------------------------------
# Построение графика плотности распределения коэффициентов корреляции

# point = -1
# step = 2 / 80
# point += step
# x, y = [-1], [0]
# for k in range(1, 80):
#     count = 0
#     for i in range(0, len(correlation_matrix)):
#         for j in range(0, len(correlation_matrix.columns)):
#             if i != j and ((point - step) <= correlation_matrix.iat[i, j] <= point):
#                 count += 1
#     x.append(point)
#     y.append(count/2)
#     point += step
#
#
# #create data
# x_p = np.array(x)
# y_p = np.array(y)
#
# #define x as 200 equally spaced values between the min and max of original x
# xnew = np.linspace(x[0], x[-1], 200)
#
# #define spline with degree k=7
# spl = make_interp_spline(x, y, k=2)
# y_smooth = spl(xnew)
#
# #create smooth line chart
# plt.plot(xnew, y_smooth)
# plt.xlabel("Коэффициент корреляции")
# plt.ylabel("Плотность распределения")
# plt.grid()
# plt.show()

# ------------------------------------------------------------------------------------

# Построение графика плотности ребёр рыночного графа

# point = 1
# all_values = 148 * 147 / 2
# step = 2 / 80
# point -= step
# x_2, y_2 = [1], [0]
# for k in range(1, 80):
#     count = 0
#     for i in range(0, len(correlation_matrix)):
#         for j in range(0, len(correlation_matrix.columns)):
#             if i != j and correlation_matrix.iat[i, j] >= point:
#                 count += 1
#     x_2.append(point)
#     y_2.append(count/2 / all_values)
#     point -= step
#
# x_2 = x_2[::-1]
# y_2 = y_2[::-1]
#
# x_p_2 = np.array(x_2)
# y_p_2 = np.array(y_2)
#
# #define x as 200 equally spaced values between the min and max of original x
# xnew = np.linspace(x_2[0], x_2[-1], 200)
#
# #define spline with degree k=7
# spl = make_interp_spline(x_2, y_2, k=2)
# y_smooth = spl(xnew)
#
# #create smooth line chart
# plt.plot(xnew, y_smooth)
#
# # plt.plot(x_p_2, y_p_2)
# plt.xlabel("Порог")
# plt.ylabel("Плотность ребёр графа")




# ------------------------------------------------------------------------------------



