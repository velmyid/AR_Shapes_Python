import cv2
import numpy as np


# Compares the input image with the template provided and returns the number of ORB matches

def ORB_detector(new_image, image_template):
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    orb_detector = cv2.ORB_create(2000, 1.2)

    kp1, des1 = orb_detector.detectAndCompute(image1, None)

    kp2, des2 = orb_detector.detectAndCompute(image_template, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda val: val.distance)

    return len(matches)
