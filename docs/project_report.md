## Person Multi-Object Tracking Using YOLOv8 and SORT

Name: Abhimanyu Dash  
Registration Number:26BIT0083  
Branch: B.Tech Information Technology

## Objective:

The objective of this project was to build a tracking system capable of detecting people in video frames and maintaining consistent identities across frames.

The system combines YOLOv8n for person detection with a custom implementation of the SORT (Simple Online and Realtime Tracking) algorithm for tracking. The system was evaluated on the MOT17-04 sequence using standard multi-object tracking metrics including MOTA, IDF1, false positives, false negatives, and identity switches.

## Approach:

The overall pipeline is:

MOT17-04 frames -> YOLOv8n -> Person detections -> Custom SORT -> Tracked bounding boxes + IDs -> Evaluation/Visualisation

## Person detection:

Used a pretrained YOLOv8n model for object detection and only passed those in the person class to the tracking system. Each box contains a bounding box in the form of [x1,y1,x2,y2] and a confidence detection score.

## SORT Tracking:

The tracking component is implemented from scratch using:
-Kalman filtering for motion prediction
-IoU(Intersection over Union) for measuring the overlap between bounding boxes.
-Hungarian assignment for matching new detections to existing tracks

Each tracked object maintains a state:[u,v,s,r,u',v',s']
where:
-u and v represent the bounding box centre.
-s represents the area of the bounding box
-r represents the aspect ratio
-u' and v' represent the estimated velocity
-s' represents the change in area.

For every new frame, existing tracks predict their new position using the Kalman filter and then the predicted positions are compared with new YOLO detections using IoU.

The Hungarian algorithm then determines the best overall assignment between detections and existing tracks.

The matched tracks are then updated with the new detections. The unmatched detections create new tracks, while tracks that remain unmatched for longer than the configured max_age are removed.

## Key Decisions:

I chose to use YOLOv8n as it provides a pretrained object detector that is suitable for person detection while still being relatively lightweight. Using a pretrained detector allowed me to focus on the tracking problem rather than training an object detector from scratch.

Instead of using a complete tracking library, I implemented the main SORT components directly, as this provided a clearer understanding of:
- Kalman filter-based prediction.
-IoU based association
-Hungarian matching
-Track creation and deletion
-Persistent object IDs

The implementation uses filterpy for the Kalman filter and SciPy's linear_sum_assignment for the Hungarian algorithm.

The current tracker uses motion and bounding box overlap but not appearance embeddings. This keeps it relatively simple but also creates limitations when multiple people overlap, cross paths or temporarily disappear.

## Dataset and Evaluation:

The system was evaluated using the MOT17-04 training sequence.
It contains:
-1050 frames
-A 1920x1080 resolution
-A 30 fps frame rate
-Ground truth pedestrian annotations

The main metrics used:
-MOTA for overall tracking accuracy, considering false positives, false negatives and ID switches.
-IDF1 measures the consistency of predicted IDs
-ID switches were counted
-False positives were incorrectly predicted objects
-False negatives were objects that were not successfully detected/tracked.

## Baseline config:
max_age=1
min_hits=3
IoU threshold=0.3

## Results:
MOTA->32.28%
IDF1->35.51%
ID switches->192
False positives->1427
False negatives->30588
Mostly tracked->10
Mostly lost->40

These show that the system is capable of maintaining object tracks but is limited in both detection recall and ID preservation.
The large number of false negatives indicates that missed detections are the largest source of error.
The IDF1 and identity switch results also show the limitations of motion-only tracking, especially when people overlap or their trajectories become difficult to distinguish.

## Failure analysis:

The evaluation showed that missed detections were the largest numerical source of error, while identity preservation was another significant limitation.

I also examined individual identity-switch events to understand why they occurred. Several representative switches occurred even when the predicted bounding box had a relatively high IoU with the ground-truth bounding box. Some examined events had IoU values above 0.8.

This suggests that the problem in these cases was not necessarily poor localisation, but maintaining the correct identity over time. Since the current SORT implementation relies on motion and bounding-box overlap rather than appearance information, it can struggle when people overlap, cross paths, or temporarily disappear.

The main areas for improvement are therefore:

- Improving detection recall to reduce false negatives.
- Adding appearance information to improve identity association.

## Limitations
-It relies primarily on motion and bounding-box overlap
-There is no appearance embedding or re-identification component.
-Missed YOLO detections cause tracks to disappear or identities to be recreated.
-The current evaluation focuses on a single MOT17 sequence.

## Future Work

The next step is to evaluate how different `max_age` values affect tracking performance while keeping the detector and other tracker parameters fixed.

Values such as 1, 2, 3, 5, and 10 can be compared using MOTA, IDF1, false positives, false negatives, and identity switches.

Another possible extension would be to compare the current motion-only SORT implementation with an appearance-based tracker such as DeepSORT.

The pipeline could also be extended to support arbitrary user-provided video files instead of being focused primarily on the MOT17 evaluation workflow.


