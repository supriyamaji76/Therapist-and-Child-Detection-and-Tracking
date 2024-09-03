
# Adult and Child Tracker In Video

This project focuses on developing a person detection and tracking system specifically designed to identify children and adults ("Therapists") in videos. The key objectives are:

1. Assign Unique IDs: Each person detected in the video is assigned a unique identifier, which is maintained throughout the video to track their movements.

2. Track Re-entries: The system should be capable of recognizing and tracking individuals if they leave the frame and later re-enter, even when dealing with multiple children and adults.

3. Assign New IDs: For any person entering the frame for the first time, a new unique ID should be generated.









## Deployment

To deploy this project run

```bash
  pip install -r requirements.txt
```
```bash
  pip install -r ultralytics
```
```bash
  python test_video.py --source path_to_input_video.mp4 --track --count
```
Here you should add input video of .mp4 format. and give the path of that input video in place of "path_to_input_video.mp4".

After running this an output file will be created and the output video of the model will be saved there





## About detectormodel.pt
This is customly trained a yolov8n model. I train this model to detect child and Adult in video file. The data set used to train is linked below

Data set link-https://universe.roboflow.com/idan-kideckel-67kqi/children-and-adults/dataset/1
