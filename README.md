# Master thesis
## Instructions

To run create_obj.exe, follow these instructions:
1) Use the "-f" ("--folder") flag to define a path to a folder.
2) Use the "--train", "--test" or both flags to perform the required actions.

To run annotation_tool.exe, follow these instructions:
1) Use either the "-i" ("--image") or "-f" ("--folder") to define the path to the images.
2) Once the application is running, start drawing the rectangles by holding down the left mouse button, and release the button once the box is complete.
3) To perform user actions, you can use the following keys: R - reset image to default, U - increase class index, D - decrease class index, S - save text file, N - go to next image, ESC - exit application.
4) Optional argument: 
    "-sd", "--save_dir" - path to the saving directory, default: cd + /custom_data/obj

To run custom_trainer.py, follow these instructions:
1) You have to build the Darknet framework on your Windows machine. For more information, go to: https://github.com/AlexeyAB/darknet#requirements
2) One required argument to run training is the "-cn" ("--class_name") argument, which contains the names of the classes (e.g., -cn class1 class2 ... classN)
3) Optional arguments:

   To make changes in the configuration file:
   - "-bs", "--batch_size", default = 64
   - "-sd", "--subdivisions", default = 16
   - "--input_size", default = 416
   - "-ln", "--learning_rate", default = 0.001
   - "--cfg_name", default = custom (e.g., --cfg_name buttons -> name: yolov3_buttons.cfg
   
   To make changes in file paths:
   - "--train_img", default = /custom_data/train.txt
   - "--test_img", default = /custom_data/test.txt
   - "--data", default = "/custom_data/obj.data"
   - "--weights_file" = "darknet53.conv.74"
   
   To make changes in detection command:
   - "--no_gpu", if this command is used, the training will run on CPU, otherwise it runs on GPU
   - "--show_loss", if this command is used, the graph of loss will be plotted
   
To run detector.py, follow these instructions:
1) There are three files needed to run detection:
    - weights file ("-w", "--weights")
    - configuration file ("-cfg", "--config_file")
    - obj.names file ("-cp", "--class_path")
2) Add images to detect on:
    - Use "-i", "--image" for one image.
    - Use "-f", "--folder" for a folder of images.
3) Optional arguments:

    To adjust detections:
    - "-c", "--confidence", minimum probability to filter weak detections. (Default value: 0.5)
    - "-t", "--threshold", threshold when applying non-maxima suppression to eliminate duplicate boxes (default value: 0.3)
    - "-sf", "--scale_factor", scale factor to resize the horizontal and vertical axis of the image (default value: 0.6)
   
    For exporting:
    - "-show_image", to show image after detection
    - "-save_image", to save image after detection
    - "-e", "--export", to export output information
    - "-sd", "--saving_directory", change saving directory (default: cd)

    For printing to the console:
    - "-PRINTALL", prints the whole dataset to the console
    - "-COORDINATES", prints the coordinates
    - "-CONFIDENCE", prints the confidence scores
    - "-BBOX_AREA", prints bounding box area (width * height)
