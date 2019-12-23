"""Image to Excel, created by Li Zihao, 2019.12.19"""

import time
import shutil
import requests
import json
import base64
import pylab
from pdf2image import convert_from_path, convert_from_bytes
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import shutil


def input_processing(path_input, ali_api):
    """
    Processing the original input images, includes format transformation, rotation, clipp.
    :param path_input: the path of original images saving, eg. r'../input_img/'.
    :param ali_api: your ali code, you have to apply it from [here](https://market.aliyun.com/products/57124001/cmapi024968.html?spm=a2c4g.11186623.2.30.22de28c2aQrFQG#sku=yuncode1896800000).
    :return:
    """
    # transform the pdf files to img.
    print('Reading the raw images.')
    pdf2img(path_input)
    print('pdf2img is over.')
    # process the images, include rotation, clip.
    adjustion(path_input + 'img/')
    print('Preprocessing...')
    ## Calling the Ali api to recognize the table.
    img2excel(path_input, path_input + 'img/adjust/', appcode=ali_api)
    print('Recognition is over.')
    ## delete the temp file
    shutil.rmtree(path_input + 'img')
    print('Congratulations!')





def pdf2img(file_path):
    """
    Transforming the pdf files to images.
    :param file_path: the path saving the raw material
    :return:
    """
    formats = ['.pdf', 'PDF']  ## do not consider the situation like '.pDF', '.Pdf', etc.
    os.makedirs(file_path + 'img')
    for root, dirs, files in os.walk(file_path, topdown=False):
        for file in files:
            if file[len(file) - 4: len(file)] in formats:
                convert_from_path(os.path.join(root, file), output_folder=root + 'img', fmt='.png')
            else:
                shutil.copyfile(os.path.join(root, file), os.path.join(root + 'img', file))


def adjustion(path_img):
    """
    Processing the images by interaction, include rotation, clip,
    :param path_img: the path of image files, eg. r'../input_img/'img/.
    :return:
    """
    fig, ax = pylab.subplots(figsize=(10, 7))  # setting the window size.
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


def img2excel(ori_path, path_img_processed, appcode=None):
    """
    Calling Alicloudapi to transform the image to excel.
    :param path_img_processed: the folder you want to save the excel files, eg. r'../input_img/img/adjust/'.
    :param ori_path: the folder you save the processed images, eg. r'../input_img/'.
    :param appcode: your client code of ali.
    :return:
    """

    os.makedirs(ori_path + 'excel')
    for root, dirs, files in os.walk(path_img_processed, topdown=False):
        for file in files:
            path_img = os.path.join(root, file)
            time.sleep(3)
            table = api_calling(appcode, img_file=path_img)
            data = table[1:-1].replace("\\n", "").replace("\\", "")  # replace the insignificant character to the standard html file.
            df = pd.read_html(table, encoding='utf-8')[0]
            print(df)
            save_path = ori_path + 'excel/' + file[:-4]
            df.to_csv(save_path + '.csv', encoding='gbk', index=False)  # saving as csv
            # saving as html file
            with open(save_path + '.html', "w", encoding="utf-8") as f_temp:
                f_temp.write(data)


def get_img_base64(img_file):
    with open(img_file, 'rb') as infile:
        s = infile.read()
        return base64.b64encode(s).decode('ascii')


def predict(url, appcode, img_base64, kv_config, old_format):
    if not old_format:
        param = {}
        param['image'] = img_base64
        if kv_config is not None:
            param['configure'] = json.dumps(kv_config)
        body = json.dumps(param)
    else:
        param = {}
        pic = {}
        pic['dataType'] = 50
        pic['dataValue'] = img_base64
        param['image'] = pic

        if kv_config is not None:
            conf = {}
            conf['dataType'] = 50
            conf['dataValue'] = json.dumps(kv_config)
            param['configure'] = conf

        inputs = {"inputs": [param]}
        body = json.dumps(inputs)

    headers = {'Authorization': 'APPCODE %s' % appcode}
    try:
        response = requests.post(url=url, headers=headers, data=body)
        return response.status_code, response.headers, response.text
    except:
        return 'ERROR', 'ERROR', 'ERROR'


def api_calling(appcode, img_file=None):
    url = 'https://form.market.alicloudapi.com/api/predict/ocr_table_parse'
    # 如果输入带有inputs, 设置为True，否则设为False
    is_old_format = False
    config = {'format': 'html', 'finance': False, 'dir_assure': False}
    # 如果没有configure字段，config设为None
    # config = None

    img_base64data = get_img_base64(img_file)
    stat, header, content = predict(url, appcode, img_base64data, config, is_old_format)
    if stat != 200:
        print('Http status code: ', stat)
        print('Error msg in header: ', header['x-ca-error-message'] if 'x-ca-error-message' in header else '')
        print('Error msg in body: ', content)
        exit()
    if is_old_format:
        result_str = json.loads(content)['outputs'][0]['outputValue']['dataValue']
    else:
        result_str = content
    return result_str.partition("\"tables\":")[2][:-1]


if __name__ == '__main__':
    input_processing(path_input=r'..\..\user_input\test\\', ali_api=None)
