# -*- coding: utf-8 -*-
#tested in pyton27
import imageio
import glob
import re

def create_gif(image_list, gif_name):

    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.2)

    return

def find_all_png(path):

    png_filenames = glob.glob(path)
    buf=[]
    for png_file in png_filenames:
        buf.append(png_file)
    return buf

create_gif(find_all_png(r"./imgs/all/*.png"),'total_score.gif')
