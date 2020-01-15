# -*- coding: utf-8 -*-
"""
Transfrom the pdf to png.
Jan. 2020, CISSDATA,  https://www.cissdata.com/
@author: Zihao Li <poderlee@outlook.com>
"""

import os
from pdf2image import convert_from_path, convert_from_bytes
import shutil
import pylab
import numpy as np
from PIL import Image


def pdf2img(file_path):
    """
    Transforming the pdf files to images.
    :param file_path: the path saving the raw material
    :return:
    """
    formats = ['.pdf', '.PDF']  ## do not consider the situation like '.pDF', '.Pdf', etc.
    formats_recognize = ['.pdf', '.PDF', '.jpg', '.png']
    if not os.path.exists(file_path + 'img'):
        os.makedirs(file_path + 'img')
    pathDir = os.listdir(file_path)
    for allDir in pathDir:
        if allDir[len(allDir) - 4: len(allDir)] in formats:
            ## the default value of dpi is 200, turn up this value the processing time will increased dramatically.
            convert_from_path(os.path.join(file_path, allDir), output_folder=file_path + 'img', fmt='.png', dpi=200,
                              output_file=allDir.split('.', 1)[0])
        elif allDir[len(allDir) - 4: len(allDir)] in formats_recognize:
            shutil.copyfile(os.path.join(file_path, allDir), os.path.join(file_path + 'img', allDir))
        else:
            print('The {0} is not image or PDF file.'.format(allDir))


def adjustion(path_img):
    """
    Processing the images by interaction, include rotation, clip,
    :param path_img: the path of image files, eg. r'../input_img/'img/.
    :return:
    """
    fig, ax = pylab.subplots(figsize=(10, 7))  # setting the window size.
    if not os.path.exists(path_img + 'adjust'):
        os.makedirs(path_img + 'adjust')  ## creating the folder to save the image.
    for root, dirs, files in os.walk(path_img, topdown=False):
        for file in files:
            img_ori = Image.open(os.path.join(root, file))
            img = np.array(img_ori)

            ## rotation.
            # """
            # have to rewrite this parts.
            # """
            # pylab.imshow(img)
            # pylab.ion()
            # pylab.pause(1)
            # pylab.close()
            # print('Please ensure whether to rotate the img:')
            # rotation_flag = input()
            # if rotation_flag == 'Yes':
            #     print('Please input the rotation angle of img:')
            #     angle = input()
            #     for i in range(int(int(angle)/90)):
            #         img = np.rot90(img)
            #     pylab.imshow(img)
            #     pylab.ion()
            #     pylab.pause(1)
            #     pylab.close()
            #     print('Please ensure whether to rotate the img:')
            #     rotation_flag_again = input()
            #     if rotation_flag_again != 'No':
            #         print('Error')

            ax.imshow(img)
            pylab.tight_layout()
            x = pylab.ginput(4)  ## capture the corner points of images by  human-computer interaction.
            img_adj = clipping_img(img_ori, x)  ## calling the function to clip the image.
            img_adj.save(path_img + 'adjust/' + file)  ## saving the processed images.
    print('Processing is over.')


def clipping_img(img, points):
    """
    Baesed on the appoint points to clip the image, thus we are able to acquire the part to recognize.
    :param img: the processed image
    :param points: the corner points of the table we wanna to recognize in the image.
    :return:
    """
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = points
    xmin = min(x0, x1, x2, x3)
    xmax = max(x0, x1, x2, x3)
    ymin = min(y0, y1, y2, y3)
    ymax = max(y0, y1, y2, y3)
    cropped = img.crop((xmin, ymin, xmax, ymax))
    return cropped


def pdf_img(path_input):
    ## preprocessing includes format transformation, determine the recognize region, skew adjustment.
    # transform the pdf files to img.
    print('Reading the raw images.')
    pdf2img(path_input)
    print('pdf2img is over.')
    # process the images, include rotation, clip.
    adjustion(path_input + 'img/')
    print('Preprocessing...')
