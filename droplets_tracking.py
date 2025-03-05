import cv2 as cv
import random
import numpy as np
import math


# Configuration Constants
MIN_DROPLET_SIZE = 18
# MIN_DROPLET_SIZE = 20
DROPLET_MULTIPLIER = 1.2


def random_colour():
    r = random.randint(0, 256)
    b = random.randint(0, 256)
    g = random.randint(0, 256)
    return r, b, g


def find_circles(local_frame):
    return cv.HoughCircles(local_frame, cv.HOUGH_GRADIENT,
                           dp=1,
                           minDist=int(MIN_DROPLET_SIZE/DROPLET_MULTIPLIER),
                           param1=100,
                           # param2=25,
                           param2=15,
                           minRadius=int(MIN_DROPLET_SIZE/DROPLET_MULTIPLIER),
                           maxRadius=int(MIN_DROPLET_SIZE*DROPLET_MULTIPLIER))


def delete_drops(local_dict, total_dd, num, track_time):
    new_drops = local_dict.copy()
    for items in local_dict:
        if local_dict[items][0][-1][-1] + track_time < num:
            total_dd[items] = local_dict[items]
            del new_drops[items]
    return new_drops.copy(), total_dd


def droplet_check(center_point, drops_dict):
    minimum_distance = np.inf
    minimum_distance_droplet_item = None

    for item, droplet_data in drops_dict.items():
        last_position = droplet_data[0][-1]
        distance = math.sqrt((center_point[0] - last_position[0]) ** 2 + (center_point[1] - last_position[1]) ** 2)
        if distance < minimum_distance:
            minimum_distance = distance
            minimum_distance_droplet_item = item
    return minimum_distance_droplet_item, minimum_distance


def track_drops(infunc_circle_coordinates, num, track_distance, frame_resized, drops_dict):
    if infunc_circle_coordinates is not None:
        int_circles = np.round(infunc_circle_coordinates[0, :]).astype("int")
        radii_list = []
        for droplet in int_circles:
            # checking if there are any matches and make new if yes
            drop_id, drop_distance = droplet_check(droplet, drops_dict)
            # print(not drops_dict and drop_distance < track_distance)
            if drops_dict and drop_distance < track_distance:
                cv.circle(frame_resized, droplet[:2], droplet[2], drops_dict[drop_id][-1].colors, thickness=2)
                cv.putText(frame_resized, 'ID#{}'.format(drop_id), droplet[:2], cv.FONT_HERSHEY_TRIPLEX, 0.5,
                           drops_dict[drop_id][-1].colors, 1)
                drops_dict[drop_id][-1].add((droplet[0], droplet[1], num))
                # print(type(droplet[0]), droplet[0])
            else:
                name = Drops.name
                drop = Drops()
                drops_dict[name] = (drop.track, drop)
                drops_dict[name][-1].add((droplet[0], droplet[1], num))
                cv.circle(frame_resized, droplet[:2], droplet[2], (100, 0, 0), thickness=2)
            radii_list.append(droplet[2])
    else:
        print("There aren't any circles in the image")

    return radii_list, drops_dict, frame_resized


class Drops:
    name = 0

    def __init__(self):
        # super().__init__()
        self.name = Drops.name
        self.colors = (random_colour())
        Drops.name += 1
        self.track = []

    def add(self, v):
        self.track.append(v)


# if __name__ == '__main__':
#     # test1.py executed as script
#     # do something