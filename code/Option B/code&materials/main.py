# -*- coding: utf-8 -*-
"""
An example script that shows how to transform the img, PDF table to EXCEL.
Jan. 2020, CISSDATA,  https://www.cissdata.com/
@author: Zihao Li <poderlee@outlook.com>
"""
import pdf2img
import table_analysis
import recognition_save


def main(path_raw_material):
    """
    main function to realize the transformation between image ant table, only support .pdf, .jpg and .png files.
    :param path_raw_material: the path to save the material wanna to process and recognition.
    :return:
    """
    # convert and uniform the format, preprocessing the img, the results will be saved in the folder 'img/adjust'.
    pdf2img.pdf_img(path_raw_material)
    # analysis the table.
    path_img = r'data_test/img/adjust'
    recognize_fragments, img_names = table_analysis.table_parse(path_img)
    # recognize and save
    tesseract_path = r'G:\tesseract_ocr\Tesseract-OCR\tesseract.exe'
    path_results = r'generated_output/results/'
    recognition_save.recognition(recognize_fragments, tesseract_path, path_results, img_names)


if __name__ == '__main__':
    main(r'data_test/')

