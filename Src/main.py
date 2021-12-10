import cv2
import numpy as np
from Detector.detector import ORB_detector

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
image_template = cv2.imread('images/shape41.jpg', 0)

while True:

    ret, frame = cap.read()

    # Get height and width of the frame
    height, width = frame.shape[:2]

    # Define the Box Dimensions
    top_left_x = int(width / 3)
    top_left_y = int((height / 2) + (height / 4))
    bottom_right_x = int((width / 3) * 2)
    bottom_right_y = int((height / 2) - (height / 4))

    # Draw rectangular box for our region of interest
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 3)

    # Crop window of obeservation we defined above
    cropped = frame[bottom_right_y:top_left_y, top_left_x:bottom_right_x]

    frame = cv2.flip(frame, 1)

    # Get number of ORB matches
    matches = ORB_detector(cropped, image_template)

    output_string = 'Threshold = ' + str(matches)
    cv2.putText(frame, output_string, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (250, 0, 150), 2)

    # Minimum threshold to identify the image
    threshold = 140

    if matches > threshold:
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 3)
        cv2.putText(frame, 'Shape 4', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('Object Detector ORB', frame)
    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()

