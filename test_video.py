import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import Counter, deque
import pandas as pd
import argparse

# Load a model
model = YOLO('detectormodel.pt')  # Load the YOLOv8n model (non-segmentation)
model.overrides['conf'] = 0.3  # NMS confidence threshold
model.overrides['iou'] = 0.4  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # Maximum number of detections per image

names = model.names
names = {value: key for key, value in names.items()}
colors = np.random.randint(0, 255, size=(len(names), 3), dtype='uint8')

print(names)
tracking_trajectories = {}

def process(image, track=True):
    global input_video_name
    bboxes = []
    frameId = 0

    if not os.path.exists('output'):
        os.makedirs('output')
    labels_file_path = os.path.abspath(f'./output/{input_video_name}_labels.txt')

    # Keep track of IDs in the current frame
    current_ids = set()

    with open(labels_file_path, 'a') as file:
        if track:
            results = model.track(image, verbose=False, device='cpu', persist=True, tracker="botsort.yaml")

            # Update tracking trajectories
            for id_ in list(tracking_trajectories.keys()):
                if id_ not in [int(bbox.id) for predictions in results if predictions is not None for bbox in predictions.boxes if bbox.id is not None]:
                    del tracking_trajectories[id_]

            for predictions in results:
                if predictions is None:
                    continue

                if predictions.boxes is None or predictions.boxes.id is None:
                    continue

                for bbox in predictions.boxes:
                    for scores, classes, bbox_coords, id_ in zip(bbox.conf, bbox.cls, bbox.xyxy, bbox.id):
                        xmin = bbox_coords[0]
                        ymin = bbox_coords[1]
                        xmax = bbox_coords[2]
                        ymax = bbox_coords[3]
                        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 225), 2)
                        bboxes.append([bbox_coords, scores, classes, id_])

                        # Replace 'adult' with 'Therapist' for the label
                        class_name = 'Therapist' if predictions.names[int(classes)] == 'adult' else predictions.names[int(classes)]
                        label = f'ID: {int(id_)} {class_name} {str(round(float(scores) * 100, 1))}%'

                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] // 3) - 20, int(ymin) - dim[1] + baseline), (30, 30, 30), cv2.FILLED)
                        cv2.putText(image, label, (int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        centroid_x = (xmin + xmax) / 2
                        centroid_y = (ymin + ymax) / 2

                        if id_ is not None and int(id_) not in tracking_trajectories:
                            tracking_trajectories[int(id_)] = deque(maxlen=5)
                        if id_ is not None:
                            tracking_trajectories[int(id_)].append((centroid_x, centroid_y))
                            current_ids.add(int(id_))

                    for id_, trajectory in tracking_trajectories.items():
                        for i in range(1, len(trajectory)):
                            cv2.line(image, (int(trajectory[i-1][0]), int(trajectory[i-1][1])), (int(trajectory[i][0]), int(trajectory[i][1])), (255, 255, 255), 2)

        for item in bboxes:
            bbox_coords, scores, classes, *id_ = item if len(item) == 4 else (*item, None)
            line = f'{frameId} {int(classes)} {int(id_[0])} {round(float(scores), 3)} {int(bbox_coords[0])} {int(bbox_coords[1])} {int(bbox_coords[2])} {int(bbox_coords[3])} -1 -1 -1 -1\n'
            file.write(line)

    if not track:
        results = model.predict(image, verbose=False, device='cpu')
        for predictions in results:
            if predictions is None:
                continue
            if predictions.boxes is None:
                continue

            for bbox in predictions.boxes:
                for scores, classes, bbox_coords in zip(bbox.conf, bbox.cls, bbox.xyxy):
                    xmin = bbox_coords[0]
                    ymin = bbox_coords[1]
                    xmax = bbox_coords[2]
                    ymax = bbox_coords[3]
                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 225), 2)
                    bboxes.append([bbox_coords, scores, classes])

                    # Replace 'adult' with 'Therapist' for the label
                    class_name = 'Therapist' if predictions.names[int(classes)] == 'adult' else predictions.names[int(classes)]
                    label = f'{class_name} {str(round(float(scores) * 100, 1))}%'

                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                    dim, baseline = text_size[0], text_size[1]
                    cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] // 3) - 20, int(ymin) - dim[1] + baseline), (30, 30, 30), cv2.FILLED)
                    cv2.putText(image, label, (int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Remove IDs that are no longer present from the text file
    if track:
        with open(labels_file_path, 'r') as file:
            lines = file.readlines()

        with open(labels_file_path, 'w') as file:
            for line in lines:
                if line:
                    frameId, cls, trackId, *_ = line.split()
                    if int(trackId) in current_ids:
                        file.write(line)

    return image



def process_video(args):
    source = args['source']
    track_ = args['track']
    count_ = args['count']

    global input_video_name
    cap = cv2.VideoCapture(int(source) if source == '0' else source)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    input_video_name = os.path.splitext(os.path.basename(source))[0]
    out = cv2.VideoWriter(f'output/{input_video_name}_output.mp4', fourcc, 15, (frame_width, frame_height))

    if not cap.isOpened():
        print(f"Error: Could not open video file {source}.")
        return

    frameId = 0
    start_time = time.time()

    while True:
        frameId += 1
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Warning: Skipping frame {frameId}.")
            break  # Or use `continue` to skip to the next frame if you expect more frames

        frame1 = frame.copy()
        if not ret:
            break

        frame = process(frame1, track_)

        if not track_ and count_:
            print('[INFO] count works only when objects are tracking.. so use: --track --count')
            break

        if track_ and count_:
            itemDict = {}
            try:
                df = pd.read_csv('output/'+input_video_name+'_labels.txt', header=None, delim_whitespace=True)
                df = df.iloc[:, 0:3]
                df.columns = ["frameid", "class", "trackid"]
                df = df[['class', 'trackid']]
                df = (df.groupby('trackid')['class']
                          .apply(list)
                          .apply(lambda x: sorted(x))
                         ).reset_index()
                df['class'] = df['class'].apply(lambda x: Counter(x).most_common(1)[0][0])
                vc = df['class'].value_counts()
                vc = dict(vc)

                vc2 = {0: 'Therapist', 1: 'Child'}  # Adjusted to show 'Therapist' instead of 'adult'
                itemDict = dict((vc2.get(key, 'Unknown'), value) for key, value in vc.items())
                itemDict = dict(sorted(itemDict.items(), key=lambda item: item[0]))
            except:
                pass

            display = frame.copy()
            h, w = frame.shape[0], frame.shape[1]
            x1, y1, x2, y2 = 10, 10, 10, 70
            txt_size = cv2.getTextSize(str(itemDict), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(frame, (x1, y1 + 1), (txt_size[0] * 2, y2), (0, 0, 0), -1)
            cv2.putText(frame, '{}'.format(itemDict), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 2)
            cv2.addWeighted(frame, 0.7, display, 1 - 0.7, 0, frame)

        end_time = time.time()
        # Removed FPS display lines

        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()





def main(args):
    process_video(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="0", help="source to detect object (default: '0') ")
    parser.add_argument('--track', action="store_true", help="to track detected objects with id")
    parser.add_argument('--count', action="store_true", help="to count detected objects")
    args = vars(parser.parse_args())

    main(args)
