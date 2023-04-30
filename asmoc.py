#ASMOC - Another Simple Modular Object Classifier
#Author: louis-e (2023)
#Description: A simple modular object classifier for detecting multiple different objects using cascade xml files from an input stream.

import cv2, copy
import numpy as np

STOP_PERCENTAGE_THRESHOLD = 65
INPUT_SOURCE = 'data/test_cut.mp4' #(0)
STOP_XML_FILE_PATH = 'data/stop_data.xml'
SPEED_XML_FILE_PATH = 'data/speed_data.xml'
TRAFFIC_LIGHT_XML_FILE_PATH = 'data/trafficlight_data.xml'
RED_SPECTRUM = [
                [(0, 120, 70), (10, 255, 255)],
                [(170, 120, 70), (200, 255, 255)]
                ]

def process_frame(frame, gray, hsv, cascade_set, roi_coordinates, color_spectrum):
    """
    Detect objects in a frame using OpenCV cascade classifier

    Parameters:
    -----------
    frame : numpy.ndarray
        A color image (RGB or BGR) of shape (height, width, channels).
    gray : numpy.ndarray
        A grayscale image of shape (height, width).
    hsv : numpy.ndarray
        An image in the HSV color space of shape (height, width, channels).
    cascade_set : cv2.CascadeClassifier
        A cascade classifier for detecting the object of interest (e.g. stop sign).
    roi_coordinates : list
        A list of four integers representing the region of interest (ROI) in percentage values [top, bottom, left, right].

    Returns:
    --------
    processed_frame : numpy.ndarray
        A color image (RGB or BGR) of shape (height, width, channels) with the object highlighted.
    processed_gray : numpy.ndarray
        A grayscale image of shape (height, width) with the ROI cropped.
    processed_hsv : numpy.ndarray
        An image in the HSV color space of shape (height, width, channels) with the ROI cropped.
    """
    cut_x, cut_y, cut_w, cut_h = int(hsv.shape[0] * roi_coordinates[0] / 100), \
                                    int(hsv.shape[0] * roi_coordinates[1] / 100), \
                                    roi_coordinates[2], roi_coordinates[3]
    frame = copy.deepcopy(frame)
    hsv_roi = copy.deepcopy(hsv)[cut_x:cut_y, cut_w:cut_h]
    gray_roi = copy.deepcopy(gray)[cut_x:cut_y, cut_w:cut_h]

    finds = cascade_set.detectMultiScale(gray_roi, scaleFactor=1.05, minNeighbors=2, minSize=(5, 5))

    for i, (x, y, w, h) in enumerate(finds):
        # Prepare color masks
        color_mask = []
        for j, color in enumerate(color_spectrum):
            color_mask.append(cv2.inRange(hsv_roi[y:y+h, x:x+w], color[0], color[1]))

        # Combine color masks
        final_mask = color_mask[0]
        for j in range(1, len(color_mask)):
            final_mask = cv2.add(final_mask, color_mask[j])
            

        # Dilate mask
        kernel = np.ones((15, 15), np.uint8)
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)

        # Calculate percentage of masked color
        red_percentage = (cv2.countNonZero(final_mask) / (w * h)) * 100

        # Highlight object
        cv2.putText(frame, str(red_percentage), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if red_percentage > STOP_PERCENTAGE_THRESHOLD:
            #region_of_interest = frame[y:y+h, x:x+w]
            #masked_img = cv2.bitwise_and(region_of_interest, region_of_interest, mask=final_mask)
            #highlight_weight = 1
            #frame[y:y+h, x:x+w] = cv2.addWeighted(region_of_interest, 0.45, masked_img, highlight_weight, 0)
            cv2.rectangle(frame, (x, y + cut_x), (x + h, y + w + cut_x), (0, 0, 255), 2)     
            cv2.putText(frame, "STOP", (int(x + w + 5), int(y + h / 2 + cut_x)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(hsv_roi, (int(x + h / 2), int(y + w / 2)), int((h / 2 + w / 2) / 2), (255, 0, 255), -1)
        
    return frame, gray_roi, hsv_roi


if __name__ == '__main__':
    stop_cascade = cv2.CascadeClassifier(STOP_XML_FILE_PATH)
    speed_cascade = cv2.CascadeClassifier(SPEED_XML_FILE_PATH)
    trafficlight_cascade = cv2.CascadeClassifier(TRAFFIC_LIGHT_XML_FILE_PATH)
    input_capture = cv2.VideoCapture(INPUT_SOURCE)

    i = 0
    while True:
        ret, frame = input_capture.read()
        i += 1
        if i < 600: continue #DBG

        if not ret: break
        frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        processed_frame, processed_gray, processed_hsv = process_frame(frame, gray, hsv, stop_cascade, [35, 90, 0, frame.shape[1]], RED_SPECTRUM)
        cv2.imshow('frame', processed_frame)
        cv2.imshow('orig_frame', frame)
        cv2.imshow('gray', processed_gray)
        cv2.imshow('hsv', processed_hsv)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    input_capture.release()
    cv2.destroyAllWindows()