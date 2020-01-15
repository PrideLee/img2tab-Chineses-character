# -*- coding: utf-8 -*-
"""
Analysis and recognize the table in the specific image.
Jan. 2020, CISSDATA,  https://www.cissdata.com/
@author: Zihao Li <poderlee@outlook.com>
"""
import os
import img_obj
import cv2
import numpy as np


OUTPUTPATH = 'generated_output/'


DIRECTION_HORIZONTAL = 'h'
DIRECTION_VERTICAL = 'v'


def table_parse(path_img):
    """
    Parsing the img and extract the table grids.
    :param path_img: the path saved the img
    :return: img_recognize: lists, the fragment images to recognize
    :return: files_name: list, the images name wanna to recognize
    """
    format_list = ['jpg', 'png']  # the format of image, we defined
    count = 0
    img_recognize = []
    files_name = []
    for root, dirs, files in os.walk(path_img, topdown=False):
        for file in files:
            suffix_file = file.split('.', 1)
            if str.lower(suffix_file[1]) in format_list:
                print("num %d: detecting lines in image file '%s'..." % (count, file))
                count += 1
                img_fragment = table_analysis(root, file)
                img_recognize.append(img_fragment)
                files_name.append(suffix_file[0])
    return img_recognize, files_name


def table_analysis(root, file):
    # create an image processing object with the scanned page
    iproc_obj = img_obj.ImageProc(os.path.join(root, file))
    imgfilebasename = file[:file.rindex('.')]
    ## HOUGH transform do not work, so replace it as pixel scan
    flag_line = 1
    ## recongize the lines of the images
    if (iproc_obj.img_h / iproc_obj.img_w < 0.25 or iproc_obj.img_h / iproc_obj.img_w > 4) or flag_line == 1:
        ## if the aspect ratio of image  over 4, choose pre-pixel scan.
        ## skew adjustment, based on four_point_transform
        skew_img = iproc_obj.img_adjust()
        iproc_obj.input_img = skew_img
        lines_hough = iproc_obj.scan_perpixl()
    else:
        # # detect the lines, Canny Sobel(edge detection) + hough transform(line detection).
        ## adjust the hough threshold to detect the line
        # lines_hough = iproc_obj.detect_lines(canny_low_thresh=50, canny_high_thresh=150, canny_kernel_size=3,
        #                                      hough_rho_res=1, hough_theta_res=np.pi / 500,
        #                                      hough_votes_thresh=round(0.2 * iproc_obj.img_w))
        ## adjust the hough threshold to detect the line
        lines_hough = iproc_obj.detect_lines_houghP(canny_low_thresh=30, canny_high_thresh=150,
                                                    canny_kernel_size=3,
                                                    hough_rho_res=1,
                                                    hough_theta_res=np.pi / 1000,
                                                    hough_votes_thresh=round(0.2 * iproc_obj.img_w),
                                                    hough_minLineLength=round(
                                                        min(iproc_obj.img_w, iproc_obj.img_h) * 0.7),
                                                    hough_maxLineGap=round(min(iproc_obj.img_w,
                                                                               iproc_obj.img_h) * 0.2))

    print("> found %d lines" % len(lines_hough))

    ## creating the folder to save the image.
    if not os.path.exists(OUTPUTPATH):
        os.makedirs(OUTPUTPATH)

    ## If choose hough transforme, argument hough_P must be False, otherwise it should be True.
    save_image_w_lines(iproc_obj, imgfilebasename, orig_img_as_background=True, hough_P=True)
    save_image_w_lines(iproc_obj, imgfilebasename, orig_img_as_background=False, hough_P=True)

    ## recognize the grids in the table
    img_grid = box_detection(iproc_obj)

    return img_grid


def save_image_w_lines(iproc_obj, imgfilebasename, orig_img_as_background, hough_P=False):
    file_suffix = 'lines-orig' if orig_img_as_background else 'lines'
    img_lines = iproc_obj.draw_lines(orig_img_as_background=orig_img_as_background, houghP=hough_P)  ## draw lines
    # write the drawing lines image.
    img_lines_file = os.path.join(OUTPUTPATH, '%s-%s.png' % (imgfilebasename, file_suffix))
    print("> saving image with detected lines to '%s'" % img_lines_file)
    cv2.imwrite(img_lines_file, img_lines)


def box_detection(img_class):
    """
    creat the table, based the lines we decateded
    :param img_class: the object of img
    :return:
    """
    vertical_list = []
    horizon_list = []
    for line in img_class.lines_hough:
        if line[2] == 'v':
            vertical_list.append([line[0], line[1]])
        else:
            horizon_list.append([line[0], line[1]])

    ## if do not find any lines, add the edges of image.
    if len(vertical_list) == 0:
        vertical_list.append(((0, 0), (0, img_class.img_h), 'v'))
        vertical_list.append(((img_class.img_w, 0), (img_class.img_w, img_class.img_h), 'v'))
    if len(horizon_list) == 0:
        horizon_list.append(((0, 0), (img_class.img_w, 0), 'h'))
        horizon_list.append(((0, img_class.img_h), (img_class.img_w, img_class.img_h), 'h'))

    ## delete the noise lines (the close line)
    vertical_list = sorted(vertical_list, key=lambda vertical_list: vertical_list[0][0])  # sorted the lines by the value of x coordinates.
    points_cluster = [line[0][0] for line in vertical_list]
    vertical_list = noise_eliminate(vertical_list, points_cluster, 'v')
    # another endpoint of lines to delete the tilt lines
    points_cluster = [line[1][0] for line in vertical_list]
    vertical_list = noise_eliminate(vertical_list, points_cluster, 'v')

    horizon_list = sorted(horizon_list, key=lambda horizon_list: horizon_list[0][1])
    points_cluster = [line[0][1] for line in horizon_list]
    horizon_list = noise_eliminate(horizon_list, points_cluster, 'h')

    points_cluster = [line[1][1] for line in horizon_list]
    horizon_list = noise_eliminate(horizon_list, points_cluster, 'h')

    ## figure the line
    # lines = []
    # for i in vertical_list:
    #     lines.append(i + ['v'])
    # for i in horizon_list:
    #     lines.append(i + ['h'])
    # draw_lines(lines, img_class)

    ##########################################################################
    ## find or determined the crossover points
    crossover_points = []
    for v_line in vertical_list:
        vertical_list_point = []
        x1, y1 = v_line[0]
        x2, y2 = v_line[1]
        for h_line in horizon_list:
            x3, y3 = h_line[0]
            x4, y4 = h_line[1]
            # calculate the formula of lines
            A1, B1, C1 = GeneralEquation(x1, y1, x2, y2)
            A2, B2, C2 = GeneralEquation(x3, y3, x4, y4)
            m = A1 * B2 - A2 * B1
            if m == 0:
                x = None
                y = None
                print("no crossover point")
            else:
                # in order to overfill, plus 0.001
                x = round(((C2 * B1 - C1 * B2)+0.001) / (m + 0.001), 4)
                y = round(((C1 * A2 - C2 * A1)+0.001) / (m + 0.001), 4)
            vertical_list_point.append((x, y))
        # sort_list = sorted(vertical_list_point, key=lambda vertical_list_point: vertical_list_point[1])
        crossover_points.append(vertical_list_point)

    # transform the vertical to horizon,
    # eg:vertical [[(x1, y1), (x2, y2), (x3, y3), (x4, y4)], [(x5, y5), (x6, y6), (x7, y7)], [(x8, y8), (x9, y9)]]
    # horizon [[(x1, y1), (x5, y5), (x8, y8)], [(x2, y2), (x6, y6), (x9, y9)], [(x3, y3), (x7, y7)]]
    crossover_points_horizon = []
    for i in range(len(crossover_points[0])):
        temp = []
        for j in range(len(crossover_points)):
            temp.append(crossover_points[j][i])
        crossover_points_horizon.append(temp)
    crossover_points_vertical = crossover_points

    ## delete the false crosspoints (the points over the table edges too much are the false or fake points)
    ## vertical
    vertical_list_filter_point = []
    for vertical_list_endpoint, crossover_point in zip(vertical_list, crossover_points_vertical):
        ## determine the margin or threshold
        y_0 = vertical_list_endpoint[0][1]
        y_1 = vertical_list_endpoint[1][1]
        margin = max(round(abs(y_0 - y_1) * 0.08), 25)
        vertical_point_delete = []
        ## if the crosspoints are far away(external) from the edges and the distances over the margin, then delete them
        for point_temp in crossover_point:
            if (min(abs(point_temp[1] - y_0), abs(point_temp[1] - y_1)) > margin) and (
              (min(y_0, y_1) > point_temp[1]) or (point_temp[1] > max(y_0, y_1))):
                vertical_point_delete.append(point_temp)
        vertical_list_filter_point.append(vertical_point_delete)

    ## delete the false points
    ## for horizon
    horizon_list_filter_point = []
    for horizon_list_endpoint, crossover_point in zip(horizon_list, crossover_points_horizon):
        x_0 = horizon_list_endpoint[0][0]
        x_1 = horizon_list_endpoint[1][0]
        margin = max(round(abs(x_0 - x_1) * 0.08), 25)
        horizon_point_delete = []
        for point_temp in crossover_point:
            if (min(abs(point_temp[0] - x_0), abs(point_temp[0] - x_1)) > margin) and ((min(x_0, x_1) > point_temp[1]) or (point_temp[1] > max(x_0, x_1))):
                horizon_point_delete.append(point_temp)
        horizon_list_filter_point.append(horizon_point_delete)

    ## combine the delete points (vertical and horizon direction)
    delete_points = []
    for temp_point in vertical_list_filter_point:
        delete_points += temp_point
    for temp_point in horizon_list_filter_point:
        delete_points += temp_point

    ## filte the crosspoints based on the delete points.
    crossover_points_vertical_delete = []
    for point_list in crossover_points_vertical:
        point_list_delete = []
        for point_temp in point_list:
            if point_temp not in delete_points:
                point_list_delete.append(point_temp)
        crossover_points_vertical_delete.append(point_list_delete)

    ## transform the points from vertical to horizon
    crossover_points_horizon_delete = []
    for point_list in crossover_points_horizon:
        point_list_delete = []
        for point_temp in point_list:
            if point_temp not in delete_points:
                point_list_delete.append(point_temp)

        if len(point_list_delete) > 1:
            crossover_points_horizon_delete.append(point_list_delete)
        # if the number of crosspoints for each line is one, than add another point to form a line
        elif len(point_list_delete) == 1:
            if abs(point_list_delete[0][0] - 0) < abs(point_list_delete[0][0] - img_class.img_w):
                point_list_delete.append((img_class.img_w, point_list_delete[0][1]))
            # else:
            #     point_list_delete = [(0, point_list_delete[0][1])] + point_list_delete
            crossover_points_horizon_delete.append(point_list_delete)

    ## adjust crossover_points_horizon_delete includes the top and down edges of table, if not add them
    margin_h = int(img_class.img_h/20)
    flag_top = 0
    flag_down = 0
    for point_cross_list in crossover_points_horizon_delete:
        for point_cross in point_cross_list:
            if abs(point_cross[1]-0) < margin_h:
                flag_top = 1
            elif abs(point_cross[1]-img_class.img_h) < margin_h:
                flag_down = 1
    if flag_top == 0:
        crossover_points_horizon_delete.append([(0, 0), (img_class.img_w, 0)])
    if flag_down == 0:
        crossover_points_horizon_delete.append([(0, img_class.img_h), (img_class.img_w, img_class.img_h)])

    # draw_points(crossover_points_horizon_delete, img_class)
    ## plot the table box(spreadsheet), insert
    img_grid = plot_box(crossover_points_horizon_delete, img_class)
    return img_grid


def noise_eliminate(lines, coordinates, flag):
    """
    # delete the noise lines which are defined as the close lines.
    :param lines: list. lines saved as endpoints coordinates, eg:[[(x_0, y_0), (x_1, y_1), 'v'],[(x_2, y_2), (x_3,y_3), 'h'],...]
    :param coordinates: list. [x1, x2, x3,...] or [y1, y2, y3,..] is the coordinates of lines endpoints. Based on these points to judge whether the lines are close.
    :param flag: 'v' or 'h' noted as horizon or vertical represented the lines direction.
    :return: lists, the filtered lines
    """

    # default the noise lines will have the similar length
    threshold = max(abs((coordinates[0] - coordinates[-1]) / 25), 10)
    filter_list = []
    ## as for vertical lines
    ## save the longest line and filte whose surrounded lines close to it
    if flag == 'v':
        # initialize
        longest_line = lines[0]
        max_length = abs(longest_line[0][1] - longest_line[1][1])
        for i in range(len(coordinates) - 1):
            if abs(coordinates[i] - coordinates[i + 1]) < threshold:  # whether it is the close lines
                # update the longest line
                if max_length < abs(lines[i + 1][0][1] - lines[i + 1][1][1]):
                    longest_line = lines[i + 1]
                    max_length = abs(lines[i + 1][0][1] - lines[i + 1][1][1])
            else:
                # save the filtered lines
                filter_list.append(longest_line)
                # initialized
                longest_line = lines[i + 1]
                max_length = abs(lines[i + 1][0][1] - lines[i + 1][1][1])
        # the last line
        filter_list.append(longest_line)
    # for horizon lines
    elif flag == 'h':
        longest_line = lines[0]
        max_length = abs(longest_line[0][0] - longest_line[1][0])
        for i in range(len(coordinates) - 1):
            if abs(coordinates[i] - coordinates[i + 1]) < threshold:
                if max_length < abs(lines[i + 1][0][0] - lines[i + 1][1][0]):
                    longest_line = lines[i + 1]
                    max_length = abs(lines[i + 1][0][0] - lines[i + 1][1][0])
            else:
                filter_list.append(longest_line)
                longest_line = lines[i + 1]
                max_length = abs(lines[i + 1][0][0] - lines[i + 1][1][0])
        filter_list.append(longest_line)
    else:
        print('Error for the key argument')
    return filter_list


def GeneralEquation(first_x, first_y, second_x, second_y):
    ## based on the coordinates to determin the formula of line
    #  Ax+By+C=0
    A = second_y - first_y
    B = first_x - second_x
    C = second_x * first_y - first_x * second_y
    return A, B, C


def plot_box(points_horizon, img_obj):
    """
    Based on the crosspoints to extract the grids
    :param points_horizon: horizon crosspoints
    :param img_obj: the class of img
    :return:
    """

    ####################################################################################################################
    ## determine the crosspoints
    ## merge all the crosspoints for each line, and delete the close points, inorder to complete the crosspoints
    ## eg. points_horizon = [[(1, 4), (2, 4), (3, 4), (6, 4)], [(2, 4), (7, 4), (8, 4)]]
    ## combination_x_cluster = [(1, 4), (2, 4), (3, 4), (6, 4), (7, 4), (8, 4)]
    combination_x = []
    for point_list in points_horizon:
        combination_x += [i[0] for i in point_list]
    combination_sort_x = sorted(combination_x)
    combination_x_cluster = [combination_sort_x[0]]
    margin = round(abs(combination_sort_x[0] - combination_sort_x[-1]) / 25)
    for i in range(len(combination_sort_x) - 1):
        if abs(combination_sort_x[i] - combination_sort_x[i + 1]) > margin:
            combination_x_cluster.append(combination_sort_x[i + 1])

    ## complete the crosspoints as the same length for each list by None
    ## eg: [[(1, 4), (2, 4), (3, 4), (6, 4)], [(2, 4), (7, 4), (8, 4)]] to
    # [[(1, 4), (2, 4), (3, 4), (6, 4), None, None], [None, (2, 4), Noen, None, None, (7, 4), (8, 4)]].
    points_horizon_add = []
    # complete each line
    for points in points_horizon:
        point_add = []
        for anchor_point_x in combination_x_cluster:
            gap = [abs(i[0] - anchor_point_x) for i in points]
            flag = 0
            for point_index, gap_temp in enumerate(gap):
                if gap_temp < margin:
                    point_add.append(points[point_index])
                    flag = 1
                    break
            if flag == 0:
                point_add.append(None)
        points_horizon_add.append(point_add)

    # transform horizon to vertical
    points_vertical_add = []
    for i in range(len(points_horizon_add[0])):
        vertical_temp = []
        for j in range(len(points_horizon_add)):
            vertical_temp.append(points_horizon_add[j][i])
        points_vertical_add.append(vertical_temp)

    # vertical points
    points_vertical = []
    for i in range(len(points_horizon_add[0])):
        vertical_temp = []
        for j in range(len(points_horizon_add)):
            if points_horizon_add[j][i] != None:
                vertical_temp.append(points_horizon_add[j][i])
        points_vertical.append(vertical_temp)

    ####################################################################################################################
    ## determine the grids
    windows = line_box(points_horizon_add, points_vertical_add, points_horizon, points_vertical)
    #### we can create another function to judge whether the box edges are the real endgs of tables
    # box_show(img_obj, windows)

    ## segmentation the img based on windows
    img_grid = segementation_grid(img_obj.input_img, windows)
    return img_grid


def segementation_grid(img, windows):
    """
    segment the image based on the grids
    :param img: object, image
    :param windows: list, grids coordinates
    :return: img buffer, the list to save the fragment images
    """
    img_buffer = []
    for windows_line in windows:
        img_line = []
        for window in windows_line:
            corner_point, left_point, down_point, diagonal_point = window
            img_line.append(img[int(corner_point[1]):int(diagonal_point[1]), int(corner_point[0]):int(diagonal_point[0])])
        img_buffer.append(img_line)
    return img_buffer


def line_box(points_horizon_all, points_vertical_all, points_horizon, points_vertical):
    """
    determine the grids
    :param points_horizon_all:list, horizon lines completed
    :param points_vertical_all:list, vertical lines completed
    :param points_horizon:list, horizon lines not completed
    :param points_vertical:list, vertical lines not completed
    :return: list, the four corner point of each windows, [corner_point, left_point, down_point, opposite_point]
    """
    margin_horizon = 0
    for temp in points_horizon:
        if abs(temp[0][0] - temp[-1][0]) > margin_horizon:
            margin_horizon = round((temp[0][0] - temp[-1][0]) / 25)
    margin_vertical = 0
    for temp in points_vertical:
        if abs(temp[0][1] - temp[-1][1]) > margin_vertical:
            margin_vertical = round((temp[0][1] - temp[-1][1]) / 25)

    windows = []
    # line-by-line (horizon direction) to extract the windows
    for line_num in range(len(points_horizon) - 1):
        points_h_temp = points_horizon[line_num]
        points_h_all_temp = points_horizon_all[line_num]
        windows_temp = []
        for point_num in range(len(points_h_temp) - 1):
            corner_point = points_h_temp[point_num]
            left_point = points_h_temp[point_num + 1]
            try:
                left_vertical_index = points_h_all_temp.index(left_point)
                left_vertical_temp = points_vertical[left_vertical_index]
                diagonal_a_index = left_vertical_temp.index(left_point) + 1
                diagonal_a_point = left_vertical_temp[diagonal_a_index]

                corner_vertical_index = points_h_all_temp.index(corner_point)
                corner_vertical_temp = points_vertical[corner_vertical_index]
                down_index = corner_vertical_temp.index(corner_point) + 1
                down_point = corner_vertical_temp[down_index]

                down_index_h = points_vertical_all[corner_vertical_index].index(down_point)
                down_horizon_point = points_horizon[down_index_h]
                diagonal_b_index = down_horizon_point.index(down_point) + 1
                diagonal_b_point = down_horizon_point[diagonal_b_index]

                diagonal_real_x = max(diagonal_a_point[0], diagonal_b_point[0])
                diagonal_real_y = max(diagonal_a_point[1], diagonal_b_point[1])

                ## select the points based on the width and length of windows (maximaze).
                if abs(diagonal_real_x - diagonal_a_point[0]) < margin_horizon and abs(
                  diagonal_real_y - diagonal_a_point[1]) < margin_vertical:
                    diagonal_point_real = diagonal_a_point
                elif abs(diagonal_real_x - diagonal_b_point[0]) < margin_horizon and abs(
                  diagonal_real_y - diagonal_b_point[1]) < margin_vertical:
                    diagonal_point_real = diagonal_b_point
                else:
                    diagonal_point_real = (diagonal_real_x, diagonal_real_y)

                if abs(left_point[0] - diagonal_point_real[0]) < margin_horizon:
                    left_point_real = left_point
                else:
                    left_point_real = (diagonal_point_real[0], left_point[1])

                if abs(down_point[1] - diagonal_point_real[1]) < margin_vertical:
                    down_point_real = down_point
                else:
                    down_point_real = (down_point[0], diagonal_point_real[1])

                windows_temp.append([corner_point, left_point_real, down_point_real, diagonal_point_real])
            except:
                # print('out of range')
                pass
        windows.append(windows_temp)
    return windows


def draw_lines(lines, img_object, orig_img_as_background=True, draw_line_num=False):
    """
    plot the lines in the specific image
    :param lines: [(x_head,y_head), (x_tail,y_tail), 'h'] or [(x_head,y_head), (x_tail,y_tail), 'v'], the coordinates are the lines' endpoints, 'v'(vertical) or 'h' (horizon)
    :param img_object: img objection (class)
    :param orig_img_as_background: Bool
    :param draw_line_num:Bool
    :return:
    """
    # Get a base image for drawing
    baseimg = _baseimg_for_drawing(img_object, orig_img_as_background)

    for i, (p1, p2, line_dir) in enumerate(lines):
        line_color = (0, 255, 0) if line_dir == DIRECTION_HORIZONTAL else (0, 0, 255)
        #
        cv2.line(baseimg, pt_to_tuple(p1), pt_to_tuple(p2), line_color, img_object.DRAW_LINE_WIDTH)
        #
        if draw_line_num:
            p_text = pt_to_tuple(p1 + (p2 - p1) * 0.5)
            cv2.putText(baseimg, str(i), p_text, cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, 3)
    cv2.imshow('line filter', baseimg)
    cv2.waitKey()


def _baseimg_for_drawing(img_object, use_orig):
    """
    Get a base image for drawing: Either the input image if <use_orig> is True or an empty (black) image.
    """
    if use_orig:
        return np.copy(img_object.input_img)
    else:
        return np.zeros((img_object.img_h, img_object.img_w, 3), np.uint8)


def pt_to_tuple(p):
    return (int(round(p[0])), int(round(p[1])))


def box_show(obj_img, points_extremity_line):
    """
    plot the grids or table
    :param obj_img: object
    :param points_extremity_line:list, the corner points of each window or grid
    :return:
    """
    lines = []
    for windows_line in points_extremity_line:
        for window in windows_line:
            corner_point, left_point, down_point, diagonal_point = window
            lines.append([corner_point, left_point, 'h'])
            lines.append([down_point, diagonal_point, 'h'])
            lines.append([corner_point, down_point, 'v'])
            lines.append([left_point, diagonal_point, 'v'])
    draw_lines(lines, obj_img)


def draw_points(points, img_object, orig_img_as_background=True):
    """
    show the crosspoints in image
    :param points: list, coordinates of each point
    :param img_object: object, image
    :param orig_img_as_background: bool, whether to show the background
    :return:
    """
    baseimg = _baseimg_for_drawing(img_object, orig_img_as_background)
    for line_points in points:
        for point in line_points:
            cv2.circle(baseimg, (int(point[0]), int(point[1])), 3, (0, 0, 255))
    cv2.imshow('point', baseimg)
    cv2.waitKey()




