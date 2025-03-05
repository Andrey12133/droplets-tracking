import cv2 as cv


def rescale_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)


def modify_frame(frame, scale):
    frame_resized = rescale_frame(frame, scale)
    gray = cv.cvtColor(frame_resized, cv.COLOR_RGB2GRAY)
    blur_gray = cv.GaussianBlur(gray, (1, 1), cv.BORDER_DEFAULT)
    return blur_gray, frame_resized


# if __name__ == '__main__':
#     # test1.py executed as script
#     # do something
#     # rescaleFrame()