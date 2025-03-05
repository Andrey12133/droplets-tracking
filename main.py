# Standard libraries
import os
import json
import math
import random
from collections import Counter

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Local imports
from image_processing import *
from droplets_tracking import *
from droplets_analysis import *


# Configuration Constants
TRACK_DISTANCE = 25  # pixels
COEFFICIENT = 1.5  # px/um
TRACK_TIME = 3 # frames (for deleting old droplets)
CLUSTER_COEFF = 0.05
SCALE = 0.5  # Rescale factor for frames
FPS = 100  # frames per second


def photo_analysis(image_path, frame_num, drops_dict, total_drops_dict):
    global TRACK_TIME, SCALE, TRACK_DISTANCE, COEFFICIENT

    frame = cv.imread(image_path)

    if frame is None:
        print("Invalid image file:", image_path)
        return None, None

    if TRACK_TIME == 0:
        drops_dict, total_drops_dict = delete_drops(drops_dict, total_drops_dict, frame_num, TRACK_TIME)
    elif frame_num % TRACK_TIME == 0:
        drops_dict, total_drops_dict = delete_drops(drops_dict, total_drops_dict, frame_num, TRACK_TIME)

    blur_frame, frame_resized = modify_frame(frame, SCALE)
    circles = find_circles(blur_frame)

    radius_list, drops_dict, frame_resized = track_drops(circles, frame_num, TRACK_DISTANCE, frame_resized, drops_dict)

    kappa, fill_rate, d_in_um, frame_with_overlay = cluster_measure(frame_resized, radius_list, COEFFICIENT, SCALE, CLUSTER_COEFF, drops_dict)

    cv.imshow("Clustered Frame", frame_with_overlay)
    cv.waitKey(10)
    return kappa, fill_rate, d_in_um, drops_dict, total_drops_dict


def plot_graph(data):
    x = np.linspace(0, len(data['kappa'])/FPS, len(data['kappa']))
    plt.plot(x, data['kappa'], label='$\kappa$')
    plt.plot(x, data['fill_rate'], label=r'$\rho_{area}$')
    plt.plot(x, data['norm_amount'], label='Normalized Amount')
    plt.plot(np.full((2,), (CHARGE_TIME+5)/FPS), np.array([0, max(data['kappa'])]), color='r', linestyle = 'dotted')
    plt.legend()
    plt.show()


def save_data_to_scv(my_dict, file_name, charge_time=0):
    read_me_name = 'descriptions_' + file_name + '.txt'
    file_name_1 = file_name + '.csv'

    with open(read_me_name, 'w') as output_1:
        output_1.write(f'Used parameters: \n'
                       f'Track distance: {TRACK_DISTANCE}\n'
                       f'Coefficient: {COEFFICIENT} px/um\n'
                       f'Track Time: {TRACK_TIME}\n'
                       f'Charge Time: {charge_time}\n'
                       f'Fill rate: {round(np.mean(my_dict["fill_rate"][0:charge_time+5]), 3)}\n'
                       f'Kappa_1 average value: {round(np.mean(my_dict["kappa"][0:charge_time+5]), 3)}\n'
                       f'Source DIR: {DIR}\n'
                       f'Image Scale: {SCALE}\n'
                       f'FPS: {FPS}')
        output_1.close()

    if data_cluster['amount'] == 0:
        data_cluster['norm_amount'] = [round(e / np.mean(data_cluster['amount'][charge_time:charge_time+5]), 3) for e in data_cluster['amount']]
    data_cluster['norm_amount'] = [round(e / np.mean(data_cluster['amount'][charge_time:charge_time+5]), 3) for e in data_cluster['amount']]

    with open(file_name_1, 'w') as output:
        output.write(f"image,kappa,fill_rate,diameter,Nt/N0,l_distance\n")
        for i in range(len(my_dict['kappa'])):
            output.write(
                f"{my_dict['image'][i]},"
                f"{round(my_dict['kappa'][i], 3)},"
                f"{round(my_dict['fill_rate'][i], 3)},"
                f"{round(my_dict['d'][i], 3)},"
                f"{round(my_dict['norm_amount'][i], 3)}\n")
        output.close()


def convert(o):
    if not isinstance(o, np.int64):
        raise TypeError
    return int(o)


def choose_images(ref):
    list_DIR = os.listdir(ref + '\\')
    # List_DIR_split = [int(os.path.splitext(x)[0]) for x in list_DIR] # if int
    List_DIR_split = [os.path.splitext(x)[0] for x in list_DIR]
    Sorted_DIR = sorted(List_DIR_split)
    return Sorted_DIR

def process_list(sorted_list, data_cluster, ref):
    relevant_drops_dict = {}
    total_drops_dict = {}
    frame_number = 0
    l_total = 37
    if 'folder' not in globals():
        folder = '1'
    for image_file in sorted_list:
        image_str = str(image_file) + r'.tiff'
        image_path = os.path.join(ref, image_str)
        print(folder, len(os.listdir(DIR+'\\')), image_str)
        kappa, fill_rate, d_in_um, relevant_drops_dict, total_drops_dict = photo_analysis(image_path, frame_number, relevant_drops_dict, total_drops_dict)
        if kappa is not None and fill_rate is not None:
            data_cluster['image'].append(image_file)
            data_cluster['kappa'].append(kappa)
            data_cluster['fill_rate'].append(fill_rate)
            data_cluster['d'].append(d_in_um)
            data_cluster['amount'].append(len(relevant_drops_dict))
        frame_number += 1
    return data_cluster


if __name__ == "__main__":
    # Directory setup and initialization

    DIR = r'D:\Tiushkevich Andrei\Electrocoalescence\September_2024\Water droplets\C-Phase_Temperature\6-12kPa_T=38_V15\2'

    SAVE_DIR = r'\data\test'
    FILE_NAME = r'test_1'

    # specify the number of the starting electrocoalescence frame and subtract 5 from it
    CHARGE_TIME = 90

    # dict for kappa, pho (l_distance)
    data_cluster = {'image': [], 'kappa': [], 'fill_rate': [], 'amount': [], 'd': [], 'l_distance': []}

    save_velocity_data = True
    test = 'velocity_8,5-12kPa.json'

    # go to view_gen_params.py to see general params of emulsion flow
    # go to compare_curves.py to compare received data
    # go to estimate flow rate to know flow rate

    # set save path
    path_to_save = os.getcwd() + '\\' + SAVE_DIR
    if not os.path.isdir(path_to_save):
        os.mkdir(path_to_save)
    os.chdir(path_to_save)

    #do iterations

    # for folder in os.listdir(DIR+'\\'):
    #     reff = os.path.join(DIR, folder)
    #     # print('reff = ', reff)
    #     sorted_image_list = choose_images(reff)
    #     # print(sorted_image_list)
    #     new_dict = process_list(sorted_image_list, data_cluster, reff)
    #     data_cluster = data_cluster | new_dict
    #     new_dict.clear()

    # choose files
    list_DIR = os.listdir(DIR + '\\')
    List_DIR_split = [int(os.path.splitext(x)[0]) for x in list_DIR] # if int
    # List_DIR_split = [os.path.splitext(x)[0] for x in list_DIR]
    Sorted_DIR = sorted(List_DIR_split)

    # Iterate through images
    frame_number = 0
    relevant_drops_dict = {}
    total_drops_dict = {}
    # for image_file in os.listdir(DIR + '\\'):
    for image_file in Sorted_DIR:
        image_str = str(image_file) + r'.tiff'
        # image_path = os.path.join(DIR, image_file)
        image_path = os.path.join(DIR, image_str)
        print(image_str)
        kappa, fill_rate, d_in_um, relevant_drops_dict, total_drops_dict = photo_analysis(image_path, frame_number, relevant_drops_dict, total_drops_dict)
        if kappa is not None and fill_rate is not None:
            data_cluster['image'].append(image_file)
            data_cluster['kappa'].append(kappa)
            data_cluster['fill_rate'].append(fill_rate)
            data_cluster['d'].append(d_in_um)
            data_cluster['amount'].append(len(relevant_drops_dict))
        frame_number += 1

    # Save data and plot

    full_dict = total_drops_dict | relevant_drops_dict
    data = {key: val[0] for key, val in full_dict.items()}

    if save_velocity_data:
        with open(test, 'w') as f:
            json.dump(data, f, indent=4, sort_keys=True, default=convert)

    save_data_to_scv(data_cluster, FILE_NAME, CHARGE_TIME)
    plot_graph(data_cluster)

    cv.destroyAllWindows()
