# Fish Landmarks
Fish landmarks detection


## Installation

* Close this repo
* Clone the code from https://github.com/eladc4/lpn-pytorch.git and attach to project
* Create an environment with `requirements.txt` file
* Install pytorch from https://pytorch.org/


## Analyze Videos

Use the `video_analyzer/run.py` tool to analyze a pair of calibrated videos

*arguments:*
* `vid1`: video from camera #1
* `vid2`: video from camera #2
* `cfg`: Model config. A YAML file required for building the model and loading weights.
* `cal`: Cameras calibration matrix (optional, if not specified then a 3d calculation is skipped
* `output_folder`: Output folder (optional, if not specified vid1 path is used)
* `start` & `end`: Start & End indices of frames in video to analyze (optional)
* `print_images`: Save images with detected labels to output_folder (optional flag, default False)

***example:***
set `MODEL.INIT_WEIGHTS: true` and `MODEL.PRETRAINED: 'pretrained_state_dict.pth'` in the YAML config file and run the following:
```angular2html
python video_analyzer/run.py --vid1 path_to_vids/vid1.mp4 --vid2 path_to_vids/vid1.mp4 --cfg path_to_config/lpn50_384x384_gc.yaml --output_folder fish_landmarks_output --start 450 --end 460 --print_images
```

## Training

### Preparing the Dataset

The dataset is managed by the `Dataset2D` object. It expects as input (`imgs_folder`) the dataset path which contains the following structure:
```
dataset path 
├── labels.pickle
├── vid_000_cam0
│   ├── img_000000.jpg
│   └── img_000001.jpg
├── vid_000_cam1
│   ├── img_000002.jpg
│   └── img_000003.jpg
├── vid_001_cam0
│   ├── img_000004.jpg
│   └── img_000005.jpg
└── vid_001_cam1
    ├── img_000006.jpg
    └── img_000007.jpg
```

The `labels.pickle` contains a list of tuples. Each item in the list is a numpy arrays of shape (10, 2) that corresponds to the image with the same index (i.e. the item at index 55 is the label of img_000055.jpg).
The numpy array describes the (x,y) location of each landmark in the image. `np.nan` means the landmark doesn't appear in the image.

*Note:* I used scripts/gen_2d_dataset.py to auto-generate the dataset from video files and their csv file annotations.


### Training

First, set some constants in the `user_constants.py` file:
* `OUTPUT_FOLDER` (str): A folder new folder will be created under this folder with a time stamp and comment if given as argument.
* `DEFAULT_DATASET_PATH` (str): Dataset path
* `_PROJECT_NAME` (str): Neptune.ai project name (required only if using neptune)
* `_NEPTUNE_API_TOKEN` (str): Neptune.ai API token (required only if using neptune)

Run the `main.py` tool to analyze a pair of calibrated videos

*main arguments:*
* `cfg` (str): Model config. A YAML file required for building the model.
* `lr` (float, optional): Override cfg LR.
* `batch_size` (int, optional): Override cfg training batch size.
* `comment` (1 or strings, optional): One or more words describing the current training.
* `neptune` (flag, optional): Use neptune.ai to track the current training.

***example:***
```angular2html
python main.py --cfg path_to_config/lpn50_384x384_gc.yaml --lr 0.0001 --batch_size 32 --neptune --comment A few words of description
```
