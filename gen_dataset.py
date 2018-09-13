"""
 -*- coding: utf-8 -*-
 @author: Kaijian Liu
 @email: kaijianliu@qq.com
 @file: gen_dataset.py
 @time: 2018/09/06
"""
import cv2
import numpy as np
import pickle
import random
from PIL import Image,ImageDraw,ImageFont
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class FontColor(object):
    def __init__(self, col_file):
        with open(col_file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            self.colorsRGB = u.load()
        self.ncol = self.colorsRGB.shape[0]

        # convert color-means from RGB to LAB for better nearest neighbour
        # computations:
        self.colorsRGB = np.r_[self.colorsRGB[:, 0:3], self.colorsRGB[:, 6:9]].astype('uint8')
        self.colorsLAB = np.squeeze(cv2.cvtColor(self.colorsRGB[None, :, :], cv2.COLOR_RGB2Lab))


def Lab2RGB(c):
    if type(c) == list:
        return cv2.cvtColor(np.array([c], dtype=np.uint8)[None,:],cv2.COLOR_Lab2RGB)
    else:
        return cv2.cvtColor(c[None, :, :],cv2.COLOR_Lab2RGB)


def RGB2Lab(rgb):
    import numpy as np
    if type(rgb) == list:
        return(cv2.cvtColor(np.asarray([rgb],dtype=np.uint8)[None,:],cv2.COLOR_RGB2Lab))
    else:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)


def get_fonts(font_path,font_sizes):
    fonts = {}
    font_files = os.listdir(font_path)

    for size in font_sizes:
        tmp = []
        for font_file in font_files:
            tmp.append(ImageFont.truetype(os.path.join(font_path,font_file), size))
        fonts[size] = tmp

    return fonts


def get_char_lines(txt_root_path):
    txt_files = random.sample(os.listdir(txt_root_path),10)  # 选十篇
    char_lines = []
    for txt in txt_files:
        f = open(os.path.join(txt_root_path,txt))
        lines = f.readlines()
        f.close()
        for line in lines:
            char_lines.append(line.strip().replace('\xef\xbb\xbf', '').replace('\ufeff', ''))
        return char_lines


# 获取chars
def get_chars(char_lines):
    while True:
        char_line = random.choice(char_lines)
        if len(char_line)>0:
            break
    line_len = len(char_line)
    char_len = random.randint(1,20)  # 最多20个字
    if line_len<=char_len:
        return char_line
    char_start = random.randint(0,line_len-char_len)
    chars = char_line[char_start:(char_start+char_len)]
    return chars


# 选择字体
def chose_font(fonts,font_sizes):
    f_size = random.choice(font_sizes)  # 不满就取最大字号吧
    font = random.choice(fonts[f_size])
    return font


# 分析图片，获取最适宜的字体颜色
def get_bestcolor(color_lib, crop_lab):
    if crop_lab.size > 4800:
        crop_lab = cv2.resize(crop_lab,(100,16))
    labs = np.reshape(np.asarray(crop_lab), (-1, 3))

    clf = KMeans(n_clusters=8)
    clf.fit(labs)

    total = [0] * 8
    for i in clf.labels_:
        total[i] = total[i] + 1

    clus_result = [[i, j] for i, j in zip(clf.cluster_centers_, total)]
    clus_result.sort(key=lambda x: x[1], reverse=True)

    color_sample = random.sample(range(color_lib.colorsLAB.shape[0]), 500)

    def caculate_distance(color_lab, clus_result):
        weight = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01]
        d = 0
        for c, w in zip(clus_result, weight):
            d = d + np.linalg.norm(c[0] - color_lab)
        return d

    color_dis = list(map(lambda x: [caculate_distance(color_lib.colorsLAB[x], clus_result), x], color_sample))
    color_dis.sort(key=lambda x: x[0], reverse=True)

    return tuple(color_lib.colorsRGB[color_dis[0][1]])


# 获得水平文本图片
def get_horizontal_text_picture(image_file,color_lib,char_lines,fonts,font_sizes):
    retry = 0
    img = Image.open(image_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w, h = img.size
    while True:
        chars = get_chars(char_lines)
        font = chose_font(fonts,font_sizes)
        f_w, f_h = font.getsize(chars)
        if f_w < w:
            # 完美分割时应该取的
            x1 = random.randint(0, w - f_w)
            y1 = random.randint(0, h - f_h)
            x2 = x1 + f_w
            y2 = y1 + f_h
            # 随机加一点偏移
            rd = random.random()
            if rd < 0.3:  # 设定偏移的概率
                crop_y1 = y1 - random.random() / 4 * f_h
                crop_x1 = x1 - random.random() / 2 * f_h
                crop_y2 = y2 + random.random() / 4 * f_h
                crop_x2 = x2 + random.random() / 2 * f_h
                crop_y1 = int(max(0, crop_y1))
                crop_x1 = int(max(0, crop_x1))
                crop_y2 = int(min(h, crop_y2))
                crop_x2 = int(min(w, crop_x2))
            else:
                crop_y1 = y1
                crop_x1 = x1
                crop_y2 = y2
                crop_x2 = x2

            crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            crop_lab = cv2.cvtColor(np.asarray(crop_img), cv2.COLOR_RGB2Lab)
            if np.linalg.norm(np.reshape(np.asarray(crop_lab),(-1,3)).std(axis=0))>35 and retry<30:  # 颜色标准差阈值，颜色太丰富就不要了
                retry = retry+1
                continue
            best_color = get_bestcolor(color_lib, crop_lab)
            break
        else:
            pass

    draw = ImageDraw.Draw(img)
    draw.text((x1, y1), chars, best_color, font=font)
    crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    return crop_img,chars


def get_vertical_text_picture(image_file,color_lib,char_lines,fonts,font_sizes):
    img = Image.open(image_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w, h = img.size
    retry = 0
    while True:
        chars = get_chars(char_lines)
        font = chose_font(fonts,font_sizes)
        ch_w = []
        ch_h = []
        for ch in chars:
            wt, ht = font.getsize(ch)
            ch_w.append(wt)
            ch_h.append(ht)
        f_w = max(ch_w)
        f_h = sum(ch_h)
        # 完美分割时应该取的,也即文本位置
        if h>f_h:
            x1 = random.randint(0, w - f_w)
            y1 = random.randint(0, h - f_h)
            x2 = x1 + f_w
            y2 = y1 + f_h
            # 随机加一点偏移
            rd = random.random()
            if rd < 0.2:  # 设定偏移的概率
                crop_x1 = x1 - random.random() / 4 * f_w
                crop_y1 = y1 - random.random() / 2 * f_w
                crop_x2 = x2 + random.random() / 4 * f_w
                crop_y2 = y2 + random.random() / 2 * f_w
                crop_y1 = int(max(0, crop_y1))
                crop_x1 = int(max(0, crop_x1))
                crop_y2 = int(min(h, crop_y2))
                crop_x2 = int(min(w, crop_x2))
            else:
                crop_y1 = y1
                crop_x1 = x1
                crop_y2 = y2
                crop_x2 = x2
            crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            crop_lab = cv2.cvtColor(np.asarray(crop_img), cv2.COLOR_RGB2Lab)
            if np.linalg.norm(
                    np.reshape(np.asarray(crop_lab), (-1, 3)).std(axis=0)) > 35 and retry < 30:  # 颜色标准差阈值，颜色太丰富就不要了
                retry = retry + 1
                continue
            best_color = get_bestcolor(color_lib, crop_lab)
            break
        else:
            pass

    draw = ImageDraw.Draw(img)
    i = 0
    for ch in chars:
        draw.text((x1, y1), ch, best_color, font=font)
        y1 = y1 + ch_h[i]
        i = i + 1

    crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    crop_img = crop_img.transpose(Image.ROTATE_270)
    return crop_img,chars

if __name__ == '__main__':
    # 读入字体色彩库
    color_lib = FontColor('./models/colors_new.cp')
    # 读入字体
    font_path = './fonts/more_font/'
    font_sizes = list(range(16,40))
    fonts = get_fonts(font_path,font_sizes)
    # 读入newsgroup
    txt_root_path = './newsgroup'
    char_lines = get_char_lines(txt_root_path=txt_root_path)


    img_root_path = './bg_img/'
    imnames_path = './imnames.cp'
    with open(imnames_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        imnames = u.load()
    imnames.remove('hubble_44.jpg')  # 这张图片太大，运行的时候会说超过限制报错，移除算了

    # import matplotlib.pyplot as plt
    labels_path = 'labels.txt'
    gs = 0
    if os.path.exists(labels_path):  # 支持中断程序后，在生成的图片基础上继续
        f = open(labels_path,'r',encoding='utf-8')
        lines = list(f.readlines())
        f.close()
        gs = int(lines[-1].strip().split('.')[0].split('_')[1])
        print('Resume generating from step %d'%gs)

    f = open(labels_path,'a',encoding='utf-8')
    print('start generating...')
    for i in range(gs+1,2000000):
        imname = random.choice(imnames)
        img_path = os.path.join(img_root_path,imname)
        rnd = random.random()
        if rnd<0.8: # 设定产生水平文本的概率
            gen_img, chars = get_horizontal_text_picture(img_path,color_lib,char_lines,fonts,font_sizes)
        else:
            gen_img, chars = get_vertical_text_picture(img_path, color_lib, char_lines, fonts, font_sizes)
        save_img_name = 'img_' + str(i).zfill(7) + '.jpg'
        if gen_img.mode != 'RGB':
            gen_img= gen_img.convert('RGB')
        gen_img.save('./images/'+save_img_name)
        f.write(save_img_name+ ' '+chars+'\n')
        print('gennerating:-------'+save_img_name)
        # plt.figure()
        # plt.imshow(np.asanyarray(gen_img))
        # plt.show()
    f.close()