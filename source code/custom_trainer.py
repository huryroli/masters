import os
import argparse
import shutil


def check_if_dir_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def write_file(file, list):
    with open(file, "w") as f:
        for n in list:
            f.writelines("%s\n" % n)


if __name__ == "__main__":
    cd = os.getcwd()

    check_if_dir_exists(cd + "/custom_data")
    check_if_dir_exists(cd + "/custom_cfg")
    check_if_dir_exists(cd + "/custom_weights")

    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', '--class_name', type = str, nargs = '*', required = True,
                        help = "Path to the image to be detected.")
    parser.add_argument("-bs", "--batch_size", type = str, default = "64",
                        help = "batch_size")
    parser.add_argument("-sd", "--subdivisions", type = str, default = "16",
                        help = "if Out of memory error occurs, increase to 32 or 64.")
    parser.add_argument("--input_size", type = str, default = "416")
    parser.add_argument("-ln", "--learning_rate", type = str, default = "0.001")
    parser.add_argument("--cfg_name", type = str, default = "custom",
                        help = "name of the config file")
    parser.add_argument("--train_img", type = str, default = cd + "/custom_data/train.txt",
                        help = "path to training images")
    parser.add_argument("--test_img", type = str, default = cd + "/custom_data/test.txt",
                        help = "path to testing images")

    # to run detection
    parser.add_argument("--no_gpu", action = "store_true",
                        help = "if the no_gpu flag is added, the training will run on cpu")
    parser.add_argument("--data", type = str, default = "custom_data/obj.data",
                        help = "path to training data")
    parser.add_argument("--weights_file", type = str, default = "darknet53.conv.74",
                        help = "path to pre trained weights file")
    parser.add_argument("--show_loss", action = "store_true")

    args = parser.parse_args()
    cn = args.class_name
    bs = args.batch_size
    sd = args.subdivisions
    c = len(cn)
    cfg_name = args.cfg_name
    train_img = args.train_img
    test_img = args.test_img

    f = (c + 5) * 3
    max_batches = c * 2000
    steps1 = int(max_batches/100 * 80)
    steps2 = int(max_batches/100 * 90)

    original_cfg = cd + r"/cfg/yolov3.cfg"
    target_cfg = cd + r"/custom_cfg/yolov3_" + cfg_name + ".cfg"
    shutil.copyfile(original_cfg, target_cfg)

    with open(target_cfg, "r") as file:
        cfg = file.readlines()

    cfg[2] = "#batch=1\n"
    cfg[3] = "#subdivisions=1\n"
    cfg[5] = "batch=" + bs + "\n"
    cfg[6] = "subdivisions=" + sd + "\n"
    cfg[7] = "width=" + args.input_size + "\n"
    cfg[8] = "height=" + args.input_size + "\n"
    cfg[17] = "learning_rate=" + args.learning_rate + "\n"
    cfg[19] = "max_batches=" + str(max_batches) + "\n"
    cfg[21] = "steps=" + str(steps1) + "," + str(steps2) + "\n"
    cfg[609] = 'classes=' + str(c) + "\n"
    cfg[695] = 'classes=' + str(c) + "\n"
    cfg[782] = 'classes=' + str(c) + "\n"
    cfg[602] = 'filters=' + str(f) + "\n"
    cfg[688] = 'filters=' + str(f) + "\n"
    cfg[775] = 'filters=' + str(f) + "\n"

    with open(target_cfg, "w") as file:
        file.writelines(cfg)

    write_file(cd + "/custom_data/obj.names", cn)

    # creating obj.data
    data = ["classes = " + str(c),
            "train = " + train_img,
            "valid = " + test_img,
            "names = " + cd + "/custom_data/obj.names",
            "backup = " + cd + "/custom_weights"]

    write_file(cd + "/custom_data/obj.data", data)

    if args.no_gpu and args.show_loss and args.map:
        os.system(
            "darknet_no_gpu.exe detector train " + args.data + " custom_cfg/yolov3_" + cfg_name + ".cfg "
            + args.weights_file + "-map")
    elif args.no_gpu and args.map:
        os.system(
            "darknet_no_gpu.exe detector train " + args.data + " custom_cfg/yolov3_" + cfg_name + ".cfg "
            + args.weights_file + " -dont_show -map")
    elif args.no_gpu:
        os.system(
            "darknet_no_gpu.exe detector train " + args.data + " custom_cfg/yolov3_" + cfg_name + ".cfg "
            + args.weights_file + " -dont_show")
    elif args.show_loss and args.map:
        os.system(
            "darknet.exe detector train " + args.data + " custom_cfg/yolov3_" + cfg_name + ".cfg "
            + args.weights_file + " -map")
    elif args.map:
        os.system(
            "darknet_no_gpu.exe detector train " + args.data + " custom_cfg/yolov3_" + cfg_name + ".cfg "
            + args.weights_file + " -dont_show -map")
    else:
        os.system(
            "darknet.exe detector train " + args.data + " custom_cfg/yolov3_" + cfg_name + ".cfg "
            + args.weights_file + " -dont_show")
