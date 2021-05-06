import os
import argparse
import shutil


def check_if_dir_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def create_train_paths(folder):
    dest = cd + "/custom_data/obj"
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(("png", "jpg", "PNG", "JPG")):
                path = os.path.join(root, f)
                shutil.copy(path, dest)
    images_list = []
    for root, dirs, files in os.walk(dest):
        for f in files:
            if f.endswith(("png", "jpg", "PNG", "JPG")):
                path = os.path.join(root, f)
                images_list.append(path)

    with open(cd + "/custom_data/train.txt", "w") as fp:
        for img in images_list:
            fp.writelines("%s\n" % img)


def create_test_paths(folder):
    dest = cd + "/custom_data/obj"
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(("png", "jpg", "PNG", "JPG")):
                path = os.path.join(root, f)
                shutil.copy(path, dest)
    images_list = []
    for root, dirs, files in os.walk(dest):
        for f in files:
            if f.endswith(("png", "jpg", "PNG", "JPG")):
                path = os.path.join(root, f)
                images_list.append(path)

    with open(cd + "/custom_data/test.txt", "w") as fp:
        for img in images_list:
            fp.writelines("%s\n" % img)


if __name__ == "__main__":
    cd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type = str,
                        help = "folder of images")
    parser.add_argument("--train", action = "store_true",
                        help = "If flag --train is set, the train.txt file is created.")
    parser.add_argument("--test", action = "store_true",
                        help = "If flag --test is set, the test.txt file is created.")

    args = parser.parse_args()

    check_if_dir_exists(cd + "/custom_data")
    check_if_dir_exists(cd + "/custom_data/obj")

    if args.train:
        print("[INFO] Copying image files to custom_data/obj")
        print("[INFO] Creating train.txt")
        create_train_paths(args.folder)
        if os.path.exists(cd + "/custom_data/train.txt"):
            print("[INFO] File train.txt created successfully.")
        else:
            print("[INFO] File was not created.")
    elif args.test:
        print("[INFO] Copying image files to custom_data/obj")
        print("[INFO] Creating test.txt")
        create_test_paths(args.folder)
        if os.path.exists(cd + "/custom_data/test.txt"):
            print("[INFO] File test.txt created successfully.")
        else:
            print("[INFO] File was not created.")
