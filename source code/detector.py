# Reference:
# Author: Rosebrock, Adrian
# Date: November 12, 2018
# Date used: October 17, 2020
# Available online: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

import cv2
import numpy as np
import os
import argparse
import time
import ntpath
import pandas as pd


def convert_to_testing(cfg):
    with open(cfg, "r") as file:
        config = file.readlines()

    config[2] = config[2].replace("#", "")
    config[3] = config[3].replace("#", "")

    config[5] = "#" + config[5]
    config[6] = "#" + config[6]

    with open(cfg, "w") as file:
        file.writelines(config)


def detection():
    for images in image_path:
        img = cv2.imread(images)
        img = cv2.resize(img, None, fx = scale_factor, fy = scale_factor)
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True, crop = False)

        net.setInput(blob)

        og_path = ntpath.basename(images)
        p_ext = os.path.splitext(og_path)
        p = p_ext[0]

        detection_start = time.time()
        print("[INFO] DETECTING CELLS ON IMAGE {}".format(og_path))
        outs = net.forward(output_layers)
        detection_end = time.time()

        print('[INFO] DETECTION TIME: {:.2f}s'.format(detection_end - detection_start))

        # Showing informations on the screen
        class_ids = []
        confidences = []
        bboxes = []
        iou_coordinates = []
        for out in outs:
            for c in out:
                scores = c[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    # Object detected
                    center_x = int(c[0] * width)
                    center_y = int(c[1] * height)
                    w = int(c[2] * width)
                    h = int(c[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    x1 = int((c[0] - c[2] / 2) * width)
                    x2 = int((c[0] + c[2] / 2) * width)
                    y1 = int((c[1] - c[3] / 2) * height)
                    y2 = int((c[1] + c[3] / 2) * height)

                    bboxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    iou_coordinates.append([x1, y1, x2, y2, w, h])

        indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confidence_threshold, nms_threshold)
        font_numbers = cv2.FONT_HERSHEY_DUPLEX

        print("[INFO] TOTAL NUMBER OF OBJECTS DETECTED: " + str(len(indexes)))

        output = []
        for i in range(len(bboxes)):
            if i in indexes:
                x, y, w, h = bboxes[i]
                bbox_area = str(w) + '*' + str(h)
                x1, y1, x2, y2, w, h = iou_coordinates[i]
                confidence_score = confidences[i] * 100
                label = str(classes[class_ids[i]])
                output.append([str(i), str(x1), str(y1), str(x2), str(y2), label,
                               str(confidence_score), str(bbox_area)])

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, str(i), (x, y + 30), font_numbers, 1, (0, 0, 0), 2)

                df = pd.DataFrame.from_records(output,
                                               columns = ['ID', 'x1', 'y1', 'x2', 'y2', 'Class',
                                                          'Confidence', 'Bbox area'])

        if save_image:
            print("[INFO] SAVING DETECTED IMAGE INTO " + saving_directory)
            cv2.imwrite(saving_directory + "/" + p + "_detected.png", img)
            if os.path.exists(saving_directory + "/" + p + "_detected.png"):
                print("[INFO] DETECTED IMAGE SAVED SUCCESSFULLY")
            else:
                print("[INFO] DETECTED IMAGE COULD NOT BE SAVED")

        if export:
            print("[INFO] EXPORTING OUTPUT INTO " + saving_directory)
            df.to_csv(saving_directory + "/" + p + ".csv", index = False)
            if os.path.exists(saving_directory + "/" + p + ".csv"):
                print("[INFO] OUTPUT EXPORTED SUCCESSFULLY")
            else:
                print("[INFO] OUTPUT COULD NOT BE SAVED")

        if show:
            cv2.imshow("Image: " + p, img)
            cv2.waitKey(0)

        if print_all:
            print(df)

        if print_coor:
            print(df[['ID', 'x1', 'y1', 'x2', 'y2']])

        if print_conf:
            print(df[['ID', 'Confidence']])

        if print_area:
            print(df[['ID', 'Bbox area']])

        print("--------------------------------------------------------------------------")

if __name__ == "__main__":
    cd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', "--weights", type = str, required = True)
    parser.add_argument("-cfg", "--config_file", type = str, required = True)
    parser.add_argument("-cp", "--class_path", type = str, default = cd + "obj.names",
                        help = "path to the class names")

    # path args and variables used in detection
    parser.add_argument('-i', '--image', type = str, nargs = '*', default = '',
                        help = "Path to the image to be detected.")
    parser.add_argument("-f", "--folder", type = str, nargs = '*', default = '',
                        help = "Path to the folder on which the detection is to be performed.")
    parser.add_argument("-c", "--confidence", type = float, default = 0.5,
                        help = "Minimum probability to filter weak detections. (Default value: 0.5)")
    parser.add_argument("-t", "--threshold", type = float, default = 0.3,
                        help = "Threshold when applying non-maxima suppression to eliminate duplicate boxes."
                               "(Default value: 0.3)")
    parser.add_argument("-sf", "--scale_factor", type = float, default = 0.6,
                        help = "Scale factor to resize the horizontal and vertical "
                               "axis of the image. (Default value: 0.6)")

    # args for showing and exporting output
    parser.add_argument("-show_image", action = "store_true",
                        help = "Shows the image after detection.")
    parser.add_argument("-save_image", action = "store_true",
                        help = "Saves the image after detection with its detected bounding boxes.")
    parser.add_argument("-sd", "--saving_directory", type = str, default = cd,
                        help = "Path to the directory where the detected images will be saved. "
                               "(Default value is the current directory.)")
    parser.add_argument("-e", "--export", action = "store_true",
                        help = "Exports the output into the saving directory")

    # args for printing to the console
    parser.add_argument("-PRINTALL", action = "store_true",
                        help = "Prints all information (coordinates, confidence score and the total are of the detected"
                               "bounding boxes) to the console.")
    parser.add_argument("-COORDINATES", action = "store_true",
                        help = "Print the x1, y1, x2 and y2 coordinates of the detected bounding boxes to the console.")
    parser.add_argument("-CONFIDENCE", action = "store_true",
                        help = "Print the confidence scores of the detected bounding boxes to the console.")
    parser.add_argument("-BBOX_AREA", action = "store_true",
                        help = "Print the total area (width * height) of the detected bounding boxes to the console.")
    args = parser.parse_args()

    image = args.image
    confidence_threshold = args.confidence
    nms_threshold = args.threshold
    scale_factor = args.scale_factor
    show = args.show_image
    folder = args.folder
    save_image = args.save_image
    saving_directory = args.saving_directory
    export = args.export

    # keywords for printing to the console
    print_all = args.PRINTALL
    print_coor = args.COORDINATES
    print_conf = args.CONFIDENCE
    print_area = args.BBOX_AREA

    modelWeights = args.weights
    modelConfig = args.config_file

    net = cv2.dnn.readNet(modelWeights, modelConfig)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    class_path = args.class_path
    file = open(class_path, 'r').read().split('\n')
    classes = list(filter(None, file))

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    image_path = []

    if image != '':
        image_path.append(image[0])

    if folder != '':
        for root, dirs, files in os.walk(folder[0]):
            for f in files:
                if f.endswith(("png", "jpg", "PNG", "JPG")):
                    path = os.path.join(root, f)
                    image_path.append(path)

    convert_to_testing(modelConfig)

    detection()
