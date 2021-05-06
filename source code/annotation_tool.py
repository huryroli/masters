import cv2
import os
import ntpath
import argparse
import sys


def check_if_dir_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def convert_to_yolo(width, height, bbox):
    dw = 1. / width
    dh = 1. / height
    x = abs(((bbox[0] + bbox[1]) / 2.0) * dw)
    y = abs(((bbox[2] + bbox[3]) / 2.0) * dw)
    w = abs((bbox[1] - bbox[0]) * dh)
    h = abs((bbox[3] - bbox[2]) * dh)
    return x, y, w, h


class Labeling:
    def __init__(self):
        self.ref_points = []
        self.yolo = []
        self.ix, self.iy = -1, -1
        self.x_, self.y_ = 0, 0
        self.drawing = False
        self.c = 0

    def draw_label(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_points = [(x, y)]
            self.drawing = True
            self.ix, self.iy = x, y
            self.x_, self.y_ = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                copy = img.copy()
                self.x_, self.y_ = x, y
                cv2.rectangle(copy, (self.ix, self.iy), (self.x_, self.y_), (0, 255, 0), 2)
                cv2.imshow("Annotation tool for training YOLO", copy)

        elif event == cv2.EVENT_LBUTTONUP:
            self.ref_points.append((x, y))
            self.drawing = False
            height, width, channels = img.shape

            x1 = self.ref_points[0][0]
            x2 = self.ref_points[1][0]
            y1 = self.ref_points[0][1]
            y2 = self.ref_points[1][1]

            box = (x1, x2, y1, y2)
            yolo_format = convert_to_yolo(width, height, box)

            self.yolo.append((self.c, yolo_format))

            cv2.rectangle(img, self.ref_points[0], self.ref_points[1], (0, 255, 0), 2)
            cv2.putText(img, str(self.c), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Annotation tool for training YOLO", img)


if __name__ == "__main__":
    labeling = Labeling()

    cd = os.getcwd()
    check_if_dir_exists(cd + "/custom_data/")
    check_if_dir_exists(cd + "/custom_data/obj")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type = str, default = "",
                        help = "image path")
    parser.add_argument('-f', "--folder", type = str, default = "",
                        help = "folder path")
    parser.add_argument("-sd", "--save_dir", type = str, default = cd + "/custom_data/obj",
                        help = "saving directory")
    args = parser.parse_args()

    images_list = []
    if args.image != '':
        images_list.append(args.image)

    if args.folder != '':
        for root, dirs, files in os.walk(args.folder):
            for f in files:
                if f.endswith(("png", "jpg", "PNG", "JPG")):
                    path = os.path.join(root, f)
                    images_list.append(path)

    for i in images_list:
        og_path = ntpath.basename(i)
        p_ext = os.path.splitext(og_path)
        p = p_ext[0]

        img = cv2.imread(i)
        cv2.setMouseCallback("Annotation tool for training YOLO", labeling.draw_label)
        clone = img.copy()

        height, width, channels = img.shape

        saving_directory = args.save_dir

        while True:
            cv2.imshow("Annotation tool for training YOLO", img)
            k = cv2.waitKey(0)
            if k == ord("r"):
                img = clone.copy()
                labeling.yolo.clear()
            if k == ord("u"):
                labeling.c += 1
            if k == ord("d"):
                labeling.c -= 1
            if k == ord("s"):
                with open(saving_directory + "/" + p + ".txt", "w") as f:
                    for output in labeling.yolo:
                        class_num = output[0]
                        x_yolo = output[1][0]
                        y_yolo = output[1][1]
                        w_yolo = output[1][2]
                        h_yolo = output[1][3]
                        f.writelines("{} {} {} {} {}\n".format(class_num, x_yolo, y_yolo, w_yolo, h_yolo))
                labeling.yolo.clear()
                print("[INFO] File saved successfully into " + saving_directory)
                print("----------------------------------------------------------------------------------")
            if k == ord("n"):
                break
            if k == 27:
                sys.exit()

    cv2.destroyAllWindows()
