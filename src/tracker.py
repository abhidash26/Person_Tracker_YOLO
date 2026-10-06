"""Simple Online and Realtime Tracking (SORT) implementation.

This module provides:
- IoU-based detection-to-track association
- Kalman-filter-based motion prediction
- Track creation and deletion
- Persistent object IDs
"""


import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou_batch(
    detections: np.ndarray,
    trackers: np.ndarray,
) -> np.ndarray:
    
    """
    Calculate pairwise IoU between detections and tracks.

    Args:
        detections: Bounding boxes in [x1, y1, x2, y2] format.
        trackers: Predicted bounding boxes in the same format.

    Returns:
        An (N, M) matrix containing the IoU for every detection-track pair.

    """
    if len(detections) == 0 or len(trackers) == 0:
        return np.zeros((len(detections), len(trackers)))
    
    # Expand dimensions to enable broadcasting
    detections_exp = np.expand_dims(detections, 1) # (N, 1, 4)
    trackers_exp = np.expand_dims(trackers, 0)     # (1, M, 4)
    
    # Calculate intersection coordinates
    xx1 = np.maximum(detections_exp[..., 0], trackers_exp[..., 0])
    yy1 = np.maximum(detections_exp[..., 1], trackers_exp[..., 1])
    xx2 = np.minimum(detections_exp[..., 2], trackers_exp[..., 2])
    yy2 = np.minimum(detections_exp[..., 3], trackers_exp[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    
    intersection = w * h
    
    area_detections = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
    area_trackers = (trackers[:, 2] - trackers[:, 0]) * (trackers[:, 3] - trackers[:, 1])
    
    union = np.expand_dims(area_detections, 1) + np.expand_dims(area_trackers, 0) - intersection
    
    return intersection / (union + 1e-6)


def associate_detections_to_trackers(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Match detections to predicted tracks using IoU.

    Returns:
        matches: Detection-track index pairs.
        unmatched_detections: Detection indices without a match.
        unmatched_trackers: Track indices without a match.

    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    
    iou_matrix = iou_batch(detections, trackers)
    
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            # Simple matching when no ambiguity
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # Hungarian algorithm for optimal assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.stack([row_ind, col_ind], axis=1)
    else:
        matched_indices = np.empty((0, 2))
        
    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
            
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
            
    # Filter out matches with low IoU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class KalmanBoxTracker:
    """
    Track one object using a constant-velocity Kalman filter.

    State:
        [center_x, center_y, area, aspect_ratio,
         velocity_x, velocity_y, area_velocity]
    """
    count = 0
    def __init__(self, bbox):
        """
        Initializes a tracker using initial bounding box.
        """
        # State:
        # [center_x, center_y, area, aspect_ratio,
        #  velocity_x, velocity_y, area_velocity]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],  
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000. # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box in the form [x1, y1, x2, y2] and returns z in the form
        [x, y, s, r] where x,y is the center of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x, score=None):
        """
        Takes a bounding box in the center form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if (score is None):
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))

    def update(self, bbox):
        """
        Updates the state vector with observed bounding box.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if (self.time_since_update > 0):
            self.hit_streak = 0
            
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.convert_x_to_bbox(self.kf.x)

class Sort:
    '''Manage multiple Kalman-filter-based object tracks.'''
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        """
        self.frame_count += 1
        
        #Predict the next position of every existing track.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        # Remove tracks whose predictions are invalid.
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        
        # Update matched tracks using their assigned detections.
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            
        # Start a new track for every unmatched detection.
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
            
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Return tracks that are actively being tracked and have reached minimum hits
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1)) 
            i -= 1
            # Delete tracks that have been unmatched for too long.
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

