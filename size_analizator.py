import cv2 as cv
import numpy
import numpy as np
import os

print("Your OpenCV version is: " + cv.__version__)


def rescaleFrame(frame):
    print(frame.shape[:])
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    print(frame.shape[:])

    dimentions = (width, height)
    return cv.resize(frame, dimentions, interpolation=cv.INTER_AREA)


def frame_processing(frame):
    frame_resized = rescaleFrame(frame)
    gray = cv.cvtColor(frame_resized, cv.COLOR_RGB2GRAY)
    blur_gray = cv.GaussianBlur(gray, (1, 1), cv.BORDER_DEFAULT)

    return blur_gray, frame_resized


def file_txt(name, s_l):
    k = [len(s_l), np.mean(s_l), np.median(s_l), np.min(s_l), np.max(s_l), np.var(s_l), np.sqrt(np.var(s_l))]
    lk = ['amount', 'mean', 'median', 'min', 'max']
    f = open(name + '.txt', "w+")
    for h in range(len(lk)):
        f.write('{} = {}\n'.format(lk[h], k[h]))
    f.write(f'variance = {k[5]:.3f}\n')
    f.write(f'standard deviation = {k[6]:.3f}\n')
    f.write(f'in percentages = {k[6]*100/k[2]:.1f}%\n')
    print(k[2])
    f.close()


scale = 0.5
coefficient = 1.5 / 1000  # px/mm
fps = 100

vel_x_n.append(np.mean(vel_x_nn[it]) * fps * coefficient / scale)


def photo_analysis(DIR, name):
    global koef, scale
    # 10x 0.837 um/px
    # 10x 1.054 um/px lieca
    # 5x 2.108 um/px leica
    # scale = 0.7
    # 5x 1.66 um/px
    koef = 0.91
    scale = 0.7
    size_list = []
    frame = cv.imread(DIR)
    blur_gray, frame_rebuild = frame_processing(frame)

    # circles = cv.HoughCircles(blur_gray, cv.HOUGH_GRADIENT, dp=1.5, minDist=20, param1=35, param2=35,
    #                           minRadius=16, maxRadius=25)
    circles = cv.HoughCircles(blur_gray, cv.HOUGH_GRADIENT, dp=1, minDist=15, param1=40, param2=40,
                              minRadius=15, maxRadius=35)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            lenght = int(r*2/scale*koef)
            size_list.append(lenght)
            cv.circle(frame_rebuild, (x, y), r, (255, 0, 0), thickness=1)
            cv.line(frame_rebuild, (x - r, y), (x + r, y), (0, 255, 0), thickness=1)
            cv.putText(frame_rebuild, '{}'.format(lenght), (x - 15, y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            cv.line(frame_rebuild, (50, 50), (50 + int(100/koef*scale), 50), (80, 80, 200), thickness=6)
            cv.putText(frame_rebuild, '100 mkm', (50, 35), cv.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 200), 2)
    cv.imshow("Video", frame_rebuild)
    cv.waitKey(0)

    cv.imwrite(DIR_to_save + name, frame_rebuild)
    # cv.imwrite('C:/Users/Xiaomi/Documents/Laboratory/Materials/Picrures/Defenition_of_sizes/1.jpg', frame_rebuild)
    file_txt(name, size_list)


DIR_to_save = 'C:/Users/Xiaomi/Documents/Laboratory/Materials/Picrures/Defenition_of_sizes/06.07.2022/'

if not os.path.isdir(DIR_to_save):
    os.mkdir(DIR_to_save)

os.chdir(DIR_to_save)

# DIR = r'E:\new\12.11.21\UF_files\001.png'
DIR = r'E:\Andrei Tushkevich\laba_data\22.07.01\6\5.png'
photo_name = 'ret_2' + '.png'

photo_analysis(DIR, photo_name)