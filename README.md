# img2tab-Chineses-character

## Introduction
Extract the table from the image (.pdf, .png, .jpg), and transform it to excel.

Here, we try utilize two strategies:

- Option A: Preprocessing the input images, and calling the API to recognize the table;
- Option B: Do not rely on the API or SDK, realize the function by open source packages;

### Option A

The detail steps includes:

- Step1. pdf2img. Transform the pdf files to img.
- Step2. preprocessing. Processing the images, include rotation, clip.
- Step3. Recognition. Calling the Ali api to recognize the table.
- Step4. Saving. Saving the excel files, and delete the intermediate results as well as temp files. 

### Option B

Based on the [Tessaract](https://github.com/tesseract-ocr/tesseract) and [pdftabextract](https://github.com/WZBSocialScienceCenter/pdftabextract) to recognize the table. To be specific, through pdftablextract package to dispose the table structure and apply tessaract to recognize the chinese characters.

**pdateing !!!!!!!!!!!**

## Usage and Example

For option.

