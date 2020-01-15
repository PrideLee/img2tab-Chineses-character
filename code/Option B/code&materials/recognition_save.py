# -*- coding: utf-8 -*-
"""
Recognize the fragment images based on tesseract-ocr/tesseract, https://github.com/tesseract-ocr/tesseract.
Thanks to Google.
Jan. 2020, CISSDATA,  https://www.cissdata.com/
@author: Zihao Li <poderlee@outlook.com>
"""
import cv2
import os
import pytesseract
import numpy as np
import pandas as pd


def recognition(img_lists, tesseract_path, path_results, img_names):
    """
    recognize the image grids
    :param img_lists: lists, to save the image grids
    :param tesseract_path: the folder path to save tesseract.exe.
    :param path_results: the folder path to save the csv files.
    :param img_names: lists, the image names we want to recognize.
    :return:
    """
    count = 1
    for img_list, name_img in zip(img_lists, img_names):
        rec_result = recognition_character(img_list, path_tesseract=tesseract_path)  # recognize
        save_csv(rec_result, path_results, name_img)  # save to the csv
        print("Image {} is over".format(count))
        count += 1
    print('Processing is over.')


def recognition_character(img_buffer, path_tesseract):
    """
    recognize
    :param img_buffer:list
    :param path_tesseract: the dir path you save the tesseract.exe
    :return:
    """
    recognition_results = []
    for img_list in img_buffer:
        recognition_list = []
        for img in img_list:
            process_img = preprocessing_img(img)
            text = tesseract_ocr(path_tesseract, process_img)
            text = text.strip('丨')  ## delete the specific symbol which be recognized as the box edge
            recognition_list.append(text)
            # cv2.imshow('temp', process_img)
            # cv2.waitKey()
        recognition_results.append(recognition_list)
    return recognition_results


def preprocessing_img(img_raw):
    """
    image enhancement.
    :param img_raw: object, original image
    :return: enhanced image
    """
    (height, width, num) = img_raw.shape
    img_resize = cv2.resize(img_raw, (width * 10, height * 10), cv2.INTER_LINEAR)
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(img_gray, (5, 5), 0)
    return blur_img


def tesseract_ocr(path_tesseract, img):
    """
    recognize each image
    :param path_tesseract: the dir path you save the tesseract.exe
    :param img: array
    :return: str, recognize results
    """
    pytesseract.pytesseract.tesseract_cmd = path_tesseract
    ## chi_sim; chienes_simple, psm -7: Treat the image as a single text line. the more detail you can see -help of tesseract.exe
    text1 = pytesseract.image_to_string(img, lang='chi_sim', config='psm -7')
    return text1


def save_csv(rec_reults, path_save, img_name):
    """
    save the recognize results as csv files
    :param rec_reults:
    :param path_save:
    :param img_name:
    :return:
    """
    # the max length in lists
    us_lens = [len(upois) for upois in rec_reults]
    len_max = max(us_lens)
    ## padding as np.nan
    us_pois = np.array([upois + [np.nan] * (len_max - le) for upois, le in zip(rec_reults, us_lens)])
    data_frame = pd.DataFrame(data=us_pois)
    ## creating the folder to save the csv
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    data_frame.to_csv(path_save+img_name+'.csv', encoding='gbk', index=False)


