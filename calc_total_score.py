# -*- coding: utf-8 -*-
#run in py27
from __future__ import division
from decimal import Decimal
import os
import time
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import imageio,glob,re

#输出全部数组元素，不省略
np.set_printoptions(threshold=np.inf)
#输出不以科学计数法显示
np.set_printoptions(suppress=False)

def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.5)
    return

def find_all_png(path):
    png_filenames = sorted(glob.glob(path), key=os.path.getmtime)
    buf=[]
    for png_file in png_filenames:
        buf.append(png_file)
    return buf

if os.path.exists('../data/source/filted_anchor_stat_1.csv') == False or os.path.getsize('../data/source/filted_anchor_stat_1.csv') == 0:
    exit(1)
if os.path.exists('../data/source/filted_anchor_stat_2.csv') == False or os.path.getsize('../data/source/filted_anchor_stat_2.csv') == 0:
    exit(1)

arr1 = np.loadtxt(open('../data/source/filted_anchor_stat_1.csv','rb'),delimiter=',',skiprows=0)
arr2 = np.loadtxt(open('../data/source/filted_anchor_stat_2.csv','rb'),delimiter=',',skiprows=0)

arr2_column_min = np.min(arr2, axis=0)
arr2_column_max = np.max(arr2, axis=0)

inc_watch_live_time_arr = []
inc_watch_num_arr = []
inc_follow_num_arr = []
inc_gift_num_arr = []
inc_comment_num_arr = []
inc_gold_coin_arr = []
inc_true_love_coin_arr = []
for i, val in enumerate(arr2):
    inc_watch_live_time = arr2[i][15] - arr1[i][15]
    inc_watch_live_time_arr.append(inc_watch_live_time)

    inc_watch_num = arr2[i][23] - arr1[i][23]
    inc_watch_num_arr.append(inc_watch_num)

    inc_follow_num = arr2[i][27] - arr1[i][27]
    inc_follow_num_arr.append(inc_follow_num)

    inc_gift_num = arr2[i][17] - arr1[i][17]
    inc_gift_num_arr.append(inc_gift_num)

    inc_comment_num = arr2[i][25] - arr1[i][25]
    inc_comment_num_arr.append(inc_comment_num)

    inc_gold_coin_num = arr2[i][19] - arr1[i][19]
    inc_gold_coin_arr.append(inc_gold_coin_num)

    inc_true_love_coin_num = arr2[i][21] - arr1[i][21]
    inc_true_love_coin_arr.append(inc_true_love_coin_num)

inc_watch_live_time_min = 0 if np.min(inc_watch_live_time_arr, axis=0) < 0 else np.min(inc_watch_live_time_arr, axis=0)
inc_watch_num_min = 0 if np.min(inc_watch_num_arr, axis=0) < 0 else np.min(inc_watch_num_arr, axis=0)
inc_follow_num_min = 0 if np.min(inc_follow_num_arr, axis=0) < 0 else np.min(inc_follow_num_arr, axis=0)
inc_gift_num_min = 0 if np.min(inc_gift_num_arr, axis=0) < 0 else np.min(inc_gift_num_arr, axis=0)
inc_comment_num_min = 0 if np.min(inc_comment_num_arr, axis=0) < 0 else np.min(inc_comment_num_arr, axis=0)
inc_gold_coin_min = 0 if np.min(inc_gold_coin_arr, axis=0) < 0 else np.min(inc_gold_coin_arr, axis=0)
inc_true_love_coin_min = 0 if np.min(inc_true_love_coin_arr, axis=0) < 0 else np.min(inc_true_love_coin_arr, axis=0)

inc_watch_live_time_max = np.max(inc_watch_live_time_arr, axis=0)
inc_watch_num_max = np.max(inc_watch_num_arr, axis=0)
inc_follow_num_max = np.max(inc_follow_num_arr, axis=0)
inc_gift_num_max = np.max(inc_gift_num_arr, axis=0)
inc_comment_num_max = np.max(inc_comment_num_arr, axis=0)
inc_gold_coin_max = np.max(inc_gold_coin_arr, axis=0)
inc_true_love_coin_max = np.max(inc_true_love_coin_arr, axis=0)

def nor(x, min, max):
    res = 0

    if max > min and x > 0 and min >= 0:
        res = Decimal((x - min)/(max - min)).quantize(Decimal('0.000000000'))
    else:
        res = 0

    return float(res)

score_arr = []
basic_arr = []
interact_arr = []
consume_arr = []
memberid_arr = []
for i, val in enumerate(arr2):
    basic = \
        (
            nor(val[29], arr2_column_min[29], arr2_column_max[29])*0.3 \
            + \
            (
                nor(val[2], arr2_column_min[2], arr2_column_max[2])*0.15 + \
                nor(val[5], arr2_column_min[5], arr2_column_max[5])*0.25 + \
                nor(val[6], arr2_column_min[6], arr2_column_max[6])*0.2 + \
                nor(val[3], arr2_column_min[3], arr2_column_max[3])*0.1 + \
                (nor(val[9], arr2_column_min[9], arr2_column_max[9])*(1/3) + nor(val[10], arr2_column_min[10], arr2_column_max[10])*(1/3) + nor(val[11], arr2_column_min[11], arr2_column_max[11])*(1/3))*0.15 + \
                (nor(val[12], arr2_column_min[12], arr2_column_max[12])*(1/3) + nor(val[13], arr2_column_min[13], arr2_column_max[13])*(1/3) + nor(val[14], arr2_column_min[14], arr2_column_max[14])*(1/3))*0.15 \
            )*0.2 \
            + \
            (
                nor(val[15], arr2_column_min[15], arr2_column_max[15])*0.5 + \
                nor(inc_watch_live_time_arr[i], inc_watch_live_time_min, inc_watch_live_time_max)*0.5 \
            )*0.3 \
            + \
            (
                nor(val[23], arr2_column_min[23], arr2_column_max[23])*0.5 + \
                nor(inc_watch_num_arr[i], inc_watch_num_min, inc_watch_num_max)*0.5 \
            )*0.2 \
        )*0.25
    basic_arr.append(basic)

    interact = \
        (
            (
                nor(val[27], arr2_column_min[27], arr2_column_max[27])*0.5 + \
                nor(inc_follow_num_arr[i], inc_follow_num_min, inc_follow_num_max)*0.5 \
            )*0.2 \
            + \
            (
                nor(val[17], arr2_column_min[17], arr2_column_max[17])*0.5 + \
                nor(inc_gift_num_arr[i], inc_gift_num_min, inc_gift_num_max)*0.5 \
            )*0.6 \
            + \
            (
                nor(val[25], arr2_column_min[25], arr2_column_max[25])*0.5 + \
                nor(inc_comment_num_arr[i], inc_comment_num_min, inc_comment_num_max)*0.5 \
            )*0.2 \
        )*0.3
    interact_arr.append(interact)

    consume = \
        (
            (
                nor(val[19], arr2_column_min[19], arr2_column_max[19])*0.5 + \
                nor(inc_gold_coin_arr[i], inc_gold_coin_min, inc_gold_coin_max)*0.5 \
            )*0.8 \
            + \
            (
                nor(val[21], arr2_column_min[21], arr2_column_max[21])*0.5 + \
                nor(inc_true_love_coin_arr[i], inc_true_love_coin_min, inc_true_love_coin_max)*0.5
            )*0.2
        )*0.35
    consume_arr.append(consume)

    tmp = \
        (
            nor(val[29], arr2_column_min[29], arr2_column_max[29])*0.3 \
            + \
            (
                nor(val[2], arr2_column_min[2], arr2_column_max[2])*0.15 + \
                nor(val[5], arr2_column_min[5], arr2_column_max[5])*0.25 + \
                nor(val[6], arr2_column_min[6], arr2_column_max[6])*0.2 + \
                nor(val[3], arr2_column_min[3], arr2_column_max[3])*0.1 + \
                (nor(val[9], arr2_column_min[9], arr2_column_max[9])*(1/3) + nor(val[10], arr2_column_min[10], arr2_column_max[10])*(1/3) + nor(val[11], arr2_column_min[11], arr2_column_max[11])*(1/3))*0.15 + \
                (nor(val[12], arr2_column_min[12], arr2_column_max[12])*(1/3) + nor(val[13], arr2_column_min[13], arr2_column_max[13])*(1/3) + nor(val[14], arr2_column_min[14], arr2_column_max[14])*(1/3))*0.15 \
            )*0.2 \
            + \
            (
                nor(val[15], arr2_column_min[15], arr2_column_max[15])*0.5 + \
                nor(inc_watch_live_time_arr[i], inc_watch_live_time_min, inc_watch_live_time_max)*0.5 \
            )*0.3 \
            + \
            (
                nor(val[23], arr2_column_min[23], arr2_column_max[23])*0.5 + \
                nor(inc_watch_num_arr[i], inc_watch_num_min, inc_watch_num_max)*0.5 \
            )*0.2 \
        )*0.25 \
        + \
        (
            (
                nor(val[27], arr2_column_min[27], arr2_column_max[27])*0.5 + \
                nor(inc_follow_num_arr[i], inc_follow_num_min, inc_follow_num_max)*0.5 \
            )*0.2 \
            + \
            (
                nor(val[17], arr2_column_min[17], arr2_column_max[17])*0.5 + \
                nor(inc_gift_num_arr[i], inc_gift_num_min, inc_gift_num_max)*0.5 \
            )*0.6 \
            + \
            (
                nor(val[25], arr2_column_min[25], arr2_column_max[25])*0.5 + \
                nor(inc_comment_num_arr[i], inc_comment_num_min, inc_comment_num_max)*0.5 \
            )*0.2 \
        )*0.3 \
        + \
        (
            (
                nor(val[19], arr2_column_min[19], arr2_column_max[19])*0.5 + \
                nor(inc_gold_coin_arr[i], inc_gold_coin_min, inc_gold_coin_max)*0.5 \
            )*0.8 \
            + \
            (
                nor(val[21], arr2_column_min[21], arr2_column_max[21])*0.5 + \
                nor(inc_true_love_coin_arr[i], inc_true_love_coin_min, inc_true_love_coin_max)*0.5
            )*0.2
        )*0.35
    score_arr.append(tmp)
    memberid_arr.append(val[1])

now = time.localtime()
date_Ymd = time.strftime('%Y%m%d', now)
date_HMS = time.strftime('%H%M%S', now)

dir_score_all = '../data/score/' + date_Ymd + '/'
if not os.path.exists(dir_score_all):
    os.makedirs(dir_score_all)
res_file = dir_score_all + 'score_live_all_' + date_HMS + '.csv'

basic_score_arr = []
interact_score_arr = []
consume_score_arr = []
memberid_score_arr = []
a = np.argsort(score_arr)[::-1][:200]
for i, val in enumerate(a):
    basic_score_arr.append(basic_arr[val])
    interact_score_arr.append(interact_arr[val])
    consume_score_arr.append(consume_arr[val])
    memberid_score_arr.append(memberid_arr[val])

    f = open(res_file, 'a+')
    row = str(int(memberid_arr[val])) + ',' + str(round(basic_arr[val] + interact_arr[val] + consume_arr[val], 6)) + ',' + str(basic_arr[val]) + ',' + str(interact_arr[val]) + ',' + str(consume_arr[val]) +'\n'
    f.write(row)
    f.close()

#总分绘制堆叠柱状图
plt.rcParams['figure.figsize'] = (12.0, 7.0) # 设置figure_size尺

x = range(len(basic_score_arr))
plt.bar(x, basic_score_arr, label='baisc_score', fc='r')
plt.bar(x, interact_score_arr, bottom=basic_score_arr, label='interact_score', fc='b')
plt.bar(x, consume_score_arr, bottom=(np.array(basic_score_arr)+np.array(interact_score_arr)).tolist(), label='consume_score', fc='y')

plt.title('Total_Score_' + date_Ymd + ' ' + date_HMS)

plt.xticks(np.arange(0, 201, 20)) #设置x轴初始值、最大值、刻度步长
plt.yticks(np.arange(0, 1.0, 0.05)) #设置y轴初始值、最大值、刻度步长

plt.grid(True, color='k', linestyle=':', linewidth=0.4) #显示参考线
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
plt.legend() #显示图例

dir_png = '../imgs/png/' + date_Ymd + '/'
if not os.path.exists(dir_png):
    os.makedirs(dir_png)
plt.savefig(dir_png + date_HMS + '.png')
#plt.show()

create_gif(find_all_png(dir_png + '*.png'), '../imgs/gif/' + date_Ymd + '.gif')
