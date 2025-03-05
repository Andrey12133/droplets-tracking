import numpy as np
import cv2 as cv
from collections import Counter


def minimal_distance(l):
    return l['length']


def find_P_cluster(image, length, drops_dict):

    list_distribution = []
    overlay = image.copy()
    alpha = 0.3

    for drop in drops_dict.keys():
        i = -1
        for neighbor in drops_dict.keys():
            if (drops_dict[drop][-2][-1][0] - drops_dict[neighbor][-2][-1][0]) ** 2 + (
                    drops_dict[drop][-2][-1][1] - drops_dict[neighbor][-2][-1][1]) ** 2 <= length ** 2:
                i += 1
                if i > 0:
                    colour = (255, 0, 95)
                    cv.circle(overlay, drops_dict[drop][-2][-1][:2], int(length / 2), colour, thickness=-1)
        list_distribution.append(i)

    new_image = cv.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    c = Counter(list_distribution)

    percent_of_cluster = round((len(drops_dict) - c[0]) / len(drops_dict), 3)

    # print(f'cluster = {percent_of_cluster} %')

    return new_image, percent_of_cluster


def cluster_measure(frame_resized, radius, coefficient, scale, cluster_coeff, local_drops_dict):

    if radius is None:
        print("There aren't any circles in the image")
        return 0, 0, 0, 0

    mean_diameter = 2 * np.mean(radius)
    mean_diameter_in_um = mean_diameter / coefficient / scale
    # print(f'mean diameter is {round(mean_diameter_in_um, 2)} um')
    data = {'dist_to_1': []}

    for drop in local_drops_dict.keys():
        distances = []
        for neighbor in local_drops_dict.keys():
            length = round((np.sqrt((local_drops_dict[drop][-2][-1][0] - local_drops_dict[neighbor][-2][-1][0]) ** 2 +
                                    (local_drops_dict[drop][-2][-1][1] - local_drops_dict[neighbor][-2][-1][1]) ** 2)))
            distances.append({"droplet_number": neighbor, 'length': length})
        distances.sort(key=minimal_distance)

        if len(distances)>1:
            data['dist_to_1'].append(distances[1]['length'])
            start_point = local_drops_dict[drop][-2][-1][:2]
            end_point = local_drops_dict[distances[1]['droplet_number']][-2][-1][:2]
            cv.line(frame_resized, start_point, end_point, (0, 255, 0), thickness=2)
        else:
            data['dist_to_1'].append(0)

    mean_to_1 = np.mean(data['dist_to_1'])

    S_image = frame_resized.shape[0] * frame_resized.shape[1]
    # print(S_image)
    fill_rate = (np.pi * mean_diameter ** 2 * len(local_drops_dict)) / (S_image * 4)
    # print('s_droplet = ', int(np.pi * mean_diameter ** 2/4), 'droplet_amount = ', len(local_drops_dict))
    # len_cont = np.sqrt(S_image / len(local_drops_dict))
    # leng = int(cluster_coeff * (len_cont - mean_diameter) + mean_diameter)
    #
    # new_image, Pc = find_P_cluster(frame_resized, leng, local_drops_dict)

    eta = mean_to_1 / mean_diameter

    # return eta, Pc, mean_diameter_in_um, new_image
    return eta, fill_rate, mean_diameter_in_um, frame_resized
