# -*- coding: utf-8 -*-
"""
Image processing functions.
Jan. 2020, CISSDATA,  https://www.cissdata.com/
@author: Zihao Li <poderlee@outlook.com>
"""

import numpy as np
import cv2
import math
from imutils.perspective import four_point_transform


DIRECTION_HORIZONTAL = 'h'
DIRECTION_VERTICAL = 'v'


PIHLF = np.pi / 2
PI4TH = np.pi / 4


def pt_to_tuple(p):
    return (int(round(p[0])), int(round(p[1])))


def pt(x, y, dtype=np.float):
    """Create a point in 2D space at <x>, <y>"""
    return np.array((x, y), dtype=dtype)


def normalize_angle(theta):
    """Normalize an angle theta to theta_norm so that: 0 <= theta_norm < 2 * np.pi"""
    twopi = 2 * np.pi

    if theta >= twopi:
        m = math.floor(theta / twopi)
        if theta / twopi - m > 0.99999:  # account for rounding errors
            m += 1
        theta_norm = theta - m * twopi
    elif theta < 0:
        m = math.ceil(theta / twopi)
        if theta / twopi - m < -0.99999:  # account for rounding errors
            m -= 1
        theta_norm = abs(theta - m * twopi)
    else:
        theta_norm = theta

    return theta_norm


def project_polarcoord_lines(lines, img_w, img_h):
    """
    Project lines in polar coordinate space <lines> (e.g. from hough transform) onto a canvas of size
    <img_w> by <img_h>.
    """

    if img_w <= 0:
        raise ValueError('img_w must be > 0')
    if img_h <= 0:
        raise ValueError('img_h must be > 0')

    lines_ab = []
    for i, (rho, theta) in enumerate(lines):
        # calculate intersections with canvas dimension minima/maxima

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_miny = rho / cos_theta if cos_theta != 0 else float("inf")  # x for a minimal y (y=0)
        y_minx = rho / sin_theta if sin_theta != 0 else float("inf")  # y for a minimal x (x=0)
        x_maxy = (rho - img_w * sin_theta) / cos_theta if cos_theta != 0 else float("inf")  # x for maximal y (y=img_h)
        y_maxx = (rho - img_h * cos_theta) / sin_theta if sin_theta != 0 else float("inf")  # y for maximal x (y=img_w)

        # because rounding errors happen, sometimes a point is counted as invalid because it
        # is slightly out of the bounding box
        # this is why we have to correct it like this

        def border_dist(v, border):
            return v if v <= 0 else v - border

        # set the possible points
        # some of them will be out of canvas
        possible_pts = [
            ([x_miny, 0], (border_dist(x_miny, img_w), 0)),
            ([0, y_minx], (border_dist(y_minx, img_h), 1)),
            ([x_maxy, img_h], (border_dist(x_maxy, img_w), 0)),
            ([img_w, y_maxx], (border_dist(y_maxx, img_h), 1)),
        ]

        # get the valid and the dismissed (out of canvas) points
        valid_pts = []
        dismissed_pts = []
        for p, dist in possible_pts:
            if 0 <= p[0] <= img_w and 0 <= p[1] <= img_h:
                valid_pts.append(p)
            else:
                dismissed_pts.append((p, dist))

        # from the dismissed points, get the needed ones that are closed to the canvas
        n_needed_pts = 2 - len(valid_pts)
        if n_needed_pts > 0:
            dismissed_pts_sorted = sorted(dismissed_pts, key=lambda x: abs(x[1][0]), reverse=True)

            for _ in range(n_needed_pts):
                p, (dist, coord_idx) = dismissed_pts_sorted.pop()
                p[coord_idx] -= dist  # correct
                valid_pts.append(p)

        p1 = pt(*valid_pts[0])
        p2 = pt(*valid_pts[1])

        lines_ab.append((p1, p2))

    return lines_ab


class ImageProc:
    """
    Class for image processing. Methods for detecting lines in an image and clustering them. Helper methods for
    drawing.
    """
    DRAW_LINE_WIDTH = 2

    def __init__(self, imgfile):
        """
        Create a new image processing object for <imgfile>.
        """
        if not imgfile:
            raise ValueError("parameter 'imgfile' must be a non-empty, non-None string")

        self.imgfile = imgfile
        self.input_img = None
        self.img_w = None
        self.img_h = None

        self.gray_img = None  # grayscale version of the input image
        self.blur_img = None  # gaussian blur
        self.edges = None  # edges detected by Canny algorithm
        self.skew_correction_img = None

        self.lines_hough = []  # contains tuples (rho, theta, theta_norm, DIRECTION_HORIZONTAL or DIRECTION_VERTICAL)

        self._load_imgfile()

    def detect_lines(self, canny_low_thresh, canny_high_thresh, canny_kernel_size,
                     hough_rho_res, hough_theta_res, hough_votes_thresh,
                     gray_conversion=cv2.COLOR_BGR2GRAY):
        """
        Detect lines in input image using hough transform.
        Return detected lines as list with tuples:
        (rho, theta, normalized theta with 0 <= theta_norm < np.pi, DIRECTION_VERTICAL or DIRECTION_HORIZONTAL)
        """

        self.gray_img = cv2.cvtColor(self.input_img, gray_conversion)
        self.blur_img = cv2.GaussianBlur(self.img_gray, (5, 5), 0)
        self.edges = cv2.Canny(self.blur_img, canny_low_thresh, canny_high_thresh, apertureSize=canny_kernel_size)

        # detect lines with hough transform
        lines = cv2.HoughLines(self.edges, hough_rho_res, hough_theta_res, hough_votes_thresh)


        if lines is None:
            lines = []

        self.lines_hough = self._generate_hough_lines(lines)

        return self.lines_hough

    def detect_lines_houghP(self, canny_low_thresh, canny_high_thresh, canny_kernel_size,
                     hough_rho_res, hough_theta_res, hough_votes_thresh, hough_minLineLength, hough_maxLineGap,
                     gray_conversion=cv2.COLOR_BGR2GRAY):
        """
        Detect lines in input image using hough probability transform.
        Return detected lines as list with tuples:
        (rho, theta, normalized theta with 0 <= theta_norm < np.pi, DIRECTION_VERTICAL or DIRECTION_HORIZONTAL)
        """

        self.gray_img = cv2.cvtColor(self.input_img, gray_conversion)
        self.blur_img = cv2.GaussianBlur(self.gray_img, (5, 5), 0)
        self.edges = cv2.Canny(self.blur_img, canny_low_thresh, canny_high_thresh, apertureSize=canny_kernel_size)

        # detect lines with hough transform

        lines = cv2.HoughLinesP(self.edges, hough_rho_res, hough_theta_res, hough_votes_thresh,
                                  minLineLength=hough_minLineLength, maxLineGap=hough_maxLineGap)

        if lines is None:
            lines = []

        self.lines_hough = self._generate_hough_lines_houghP(lines)

        return self.lines_hough


    def segmentation_site_img(self, column_list_gray_scale, boundary=None):
        seg_bin_sign = max(column_list_gray_scale) * 0.4
        site_roughly_list = []
        for i in range(len(column_list_gray_scale) - 1):
            if column_list_gray_scale[i] > seg_bin_sign:
                # if abs(column_list_gray_scale[i]-column_list_gray_scale[i+1]) > seg_bin_sign:
                site_roughly_list.append(i)
        site_precisely_list = []

        for z in range(len(site_roughly_list)):
            site_sign = 0.02 * len(column_list_gray_scale)
            try:
                if site_roughly_list[z + 1] - site_roughly_list[z] > site_sign:
                    site_precisely_list.append(site_roughly_list[z])
            except:
                site_precisely_list.append(site_roughly_list[z])
        if boundary is not None:
            if site_precisely_list[0] - 0 > site_sign:
                site_precisely_list.insert(0, 0)
            if boundary - site_precisely_list[-1] > site_sign:
                site_precisely_list.append(boundary)
        return site_precisely_list

    def scan_perpixl(self):
        img_raw = self.input_img
        (height, width, num) = img_raw.shape

        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(img_gray, (5, 5), 0)
        binary_img = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        filter_img = cv2.medianBlur(binary_img, 3)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(filter_img, kernel, iterations=1)
        thresh1 = erosion

        (h, w) = thresh1.shape
        column_list = [0 for k in range(w)]
        #   Traversing through the  columns.
        for j in range(0, w):
            for m in range(0, h):
                #   Traversing through the rows.
                if thresh1[m, j] == 0:
                    column_list[j] += 1
        column_list_sum = column_list
        # ax1 = plt.subplot(1, 2, 1)
        # plt.sca(ax1)
        # plt.bar(range(len(column_list_sum)), column_list_sum, 0.4, color="red")

        row_list = [0 for k in range(h)]
        for j in range(0, h):
            for m in range(0, w):
                if thresh1[j, m] == 0:
                    row_list[j] += 1
        row_list_sum = row_list
        # ax2 = plt.subplot(1, 2, 2)
        # plt.sca(ax2)
        # plt.bar(range(len(row_list_sum)), row_list_sum, 0.4, color="green")
        # plt.show()

        column_site = self.segmentation_site_img(column_list_sum, w)
        row_site = self.segmentation_site_img(row_list_sum, h)
        lines = []
        for row in row_site:
            lines.append([(0, row), (width, row), 'h'])
        for column in column_site:
            lines.append([(column, 0), (column, height), 'v'])
        self.lines_hough = lines
        return self.lines_hough
        # draw_lines(lines, img_obj)

    def img_adjust(self):
        """
        This function is used to correct the skew image.

        Paraneters:
        -----------
        path_img: str
        The image path we want to correct
        img_name: num
        A name representing the sequence of corrected image.
        img_name_se: str
        A name represent the secondary sequence of corrected image, the default value is 0.

        Returns:
        --------
        skew_correction_img :the image after skew correction

        """
        # Reading the image we want to correct.
        # Calling the scanning_location function to find the image border.
        img_or = self.input_img
        left, right, upper, under = self.scanning_location(img_or)
        # According the image border to clip the image.
        img_clipping = img_or[upper:under, left:right]
        # Getting the image size after clipped.
        (height, width, num) = img_clipping.shape
        # Scaling the image to reduce the calculation.
        scaling_factor = np.maximum(height / 1500, width / 1500)
        re_height = int(np.ceil(height / scaling_factor))
        re_width = int(np.ceil(width / scaling_factor))
        img_resize = cv2.resize(img_clipping, (re_width, re_height), cv2.INTER_LINEAR)
        # Clipped RGB image to gray image.
        img_gray_sca = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        # Gaussian blur to smoothing image.
        blur_img = cv2.GaussianBlur(img_gray_sca, (5, 5), 0)
        # Gray image to binary image adaptively.
        th3 = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
        # Calling the function to find the table image corners.
        l_u_x, l_u_y, r_u_x, r_u_y, r_d_x, r_d_y, l_d_x, l_d_y = self.find_point(th3)

        # Correcting the skew and torsion image.
        skew_correction_img = four_point_transform(img_resize,
                                                   np.array([[l_u_x, l_u_y], [r_u_x, r_u_y], [r_d_x, r_d_y],
                                                             [l_d_x, l_d_y]]))

        # cv2.imshow("skew correction img", skew_correction_img)
        # # cv2.imwrite(r'E:\ciss_cool\image2table\skew correction image.png',skew_correction_img)
        # cv2.waitKey(0)
        self.skew_correction_img = skew_correction_img
        return self.skew_correction_img

    def find_point(self, bin_img):
        """
        This function is used to find the four corners of table.

        Parameters:
        -----------
        bin_img； Array
        The binary image after clipped.

        Return:
        -------
        x1: int
        The horizontal axis of left-upper corner.
        y1: int
        The vertical axis of left-upper corner.
        x2: int
        The horizontal axis of right-upper corner.
        y2: int
        The vertical axis of right-upper corner.
        x3: int
        The horizontal axis of right-under corner.
        y3: int
        The vertical axis of right-under corner.
        x4: int
        The horizontal axis of left-under corner.
        y4: int
        The vertical axis of left-under corner.
        """
        # To get the size of image.
        (hei, wid) = bin_img.shape
        # Initialize the distance list between corners and borders.
        distance1 = [([10000000000] * wid) for x in range(hei)]
        distance2 = [([10000000000] * wid) for x in range(hei)]
        distance3 = [([10000000000] * wid) for x in range(hei)]
        distance4 = [([10000000000] * wid) for x in range(hei)]
        # Calculate the distance between the each pixel and the image corners.
        for i in range(hei):
            for j in range(wid):
                if bin_img[i, j] == 0:
                    distance1[i][j] = (j - 0) ** 2 + (i - 0) ** 2
                    distance2[i][j] = (j - (wid - 1)) ** 2 + (i - 0) ** 2
                    distance3[i][j] = (j - (wid - 1)) ** 2 + (i - (hei - 1)) ** 2
                    distance4[i][j] = (j - 0) ** 2 + (i - (hei - 1)) ** 2
        # Find the points which have the minimal distance to the image corners and regarded them as table points.
        min_x_1 = np.amin(distance1, axis=0).tolist()
        min_1 = min(min_x_1)
        x1 = min_x_1.index(min_1)
        min_y_1 = np.amin(distance1, axis=1).tolist()
        y1 = min_y_1.index(min_1)
        min_x_2 = np.amin(distance2, axis=0).tolist()
        min_2 = min(min_x_2)
        x2 = min_x_2.index(min_2)
        min_y_2 = np.amin(distance2, axis=1).tolist()
        y2 = min_y_2.index(min_2)
        min_x_3 = np.amin(distance3, axis=0).tolist()
        min_3 = min(min_x_3)
        x3 = min_x_3.index(min_3)
        min_y_3 = np.amin(distance3, axis=1).tolist()
        y3 = min_y_3.index(min_3)
        min_x_4 = np.amin(distance4, axis=0).tolist()
        min_4 = min(min_x_4)
        x4 = min_x_4.index(min_4)
        min_y_4 = np.amin(distance4, axis=1).tolist()
        y4 = min_y_4.index(min_4)
        return x1, y1, x2, y2, x3, y3, x4, y4

    def scanning_location(self, img_original):
        """
        This function is used to convert the gray image to binary image, and find the image boundary.

        Parameters
        ----------
        img_original: Array
        The gray image.
        Returns
        -------
        x0: int
        The image left border.
        x1: int
        The image right border.
        y0: int
        The image upper border.
        y1: int
        The image under border.
        """
        # RGB image to gray image.
        img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        # Gray image to binary image, 100 and 255 decide the gray threshold.
        ret, bin_img = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        # Get the size of image.
        (h, w) = bin_img.shape
        column_list = []
        row_list = []
        # Scanning the columns to find the image left and right borders.
        for j in range(0, w):
            for m in range(0, h):
                if bin_img[m, j] == 0:
                    column_list.append(j)
                    break
        # Scanning the raws to find the image upper and under borders.
        for j in range(0, h):
            for m in range(0, w):
                if bin_img[j, m] == 0:
                    row_list.append(j)
                    break
        x1 = max(column_list)
        x0 = min(column_list)
        y0 = min(row_list)
        y1 = max(row_list)
        return x0, x1, y0, y1

    def ab_lines_from_hough_lines(self, lines_hough):
        """
        From a list of lines <lines_hough> in polar coordinate space, generate lines in cartesian coordinate space
        from points A to B in image dimension space. A and B are at the respective opposite borders
        of the line projected into the image.
        Will return a list with tuples (A, B, DIRECTION_HORIZONTAL or DIRECTION_VERTICAL).
        """

        projected = project_polarcoord_lines([l[:2] for l in lines_hough], self.img_w, self.img_h)
        return [(p1, p2, line_dir) for (p1, p2), (_, _, _, line_dir) in zip(projected, lines_hough)]

    def draw_lines(self, orig_img_as_background=True, draw_line_num=False, houghP=False):
        """
        Draw detected lines and return the rendered image.
        <orig_img_as_background>: if True, draw on top of input image
        <draw_line_num>: if True, draw line number
        """
        if houghP == False:
            lines_ab = self.ab_lines_from_hough_lines(self.lines_hough)
        else:
            lines_ab = self.lines_hough

        baseimg = self._baseimg_for_drawing(orig_img_as_background)

        for i, (p1, p2, line_dir) in enumerate(lines_ab):
            line_color = (0, 255, 0) if line_dir == DIRECTION_HORIZONTAL else (0, 0, 255)

            cv2.line(baseimg, pt_to_tuple(p1), pt_to_tuple(p2), line_color, self.DRAW_LINE_WIDTH)

            if draw_line_num:
                p_text = pt_to_tuple(p1 + (p2 - p1) * 0.5)
                cv2.putText(baseimg, str(i), p_text, cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, 3)

        return baseimg

    def _baseimg_for_drawing(self, use_orig):
        """
        Get a base image for drawing: Either the input image if <use_orig> is True or an empty (black) image.
        """
        if use_orig:
            return np.copy(self.input_img)
        else:
            return np.zeros((self.img_h, self.img_w, 3), np.uint8)

    def _load_imgfile(self):
        """Load the image file self.imgfile to self.input_img. Additionally set the image width and height (self.img_w
        and self.img_h)"""
        self.input_img = cv2.imread(self.imgfile)
        if self.input_img is None:
            raise IOError("could not load file '%s'" % self.imgfile)

        self.img_h, self.img_w = self.input_img.shape[:2]

    def _generate_hough_lines(self, lines):
        """
        From a list of lines in <lines> detected by cv2.HoughLines, create a list with a tuple per line
        containing:
        (rho, theta, normalized theta with 0 <= theta_norm < np.pi, DIRECTION_VERTICAL or DIRECTION_HORIZONTAL)
        """
        lines_hough = []
        for l in lines:

            rho, theta = l[0]  # they come like this from OpenCV's hough transform
            theta_norm = normalize_angle(theta)

            if abs(PIHLF - theta_norm) > PI4TH:  # vertical
                line_dir = DIRECTION_VERTICAL
            else:
                line_dir = DIRECTION_HORIZONTAL

            lines_hough.append((rho, theta, theta_norm, line_dir))

        return lines_hough

    def _generate_hough_lines_houghP(self, lines):
        """
        From a list of lines in <lines> detected by cv2.HoughLines, create a list with a tuple per line
        containing:
        (rho, theta, normalized theta with 0 <= theta_norm < np.pi, DIRECTION_VERTICAL or DIRECTION_HORIZONTAL)
        """
        lines_hough = []
        for l in lines:

            x1, y1, x2, y2 = l[0]  # they come like this from OpenCV's hough transform
            if abs(x1-x2) > abs(y1-y2):
                line_dir = DIRECTION_HORIZONTAL
            else:
                line_dir = DIRECTION_VERTICAL

            lines_hough.append(((x1, y1), (x2, y2), line_dir))

        return lines_hough

