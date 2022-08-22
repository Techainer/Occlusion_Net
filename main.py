import cv2
import numpy as np

from lib.config import cfg
from lib.predictor import COCODemo


class OcclusionNetModel:
    def __init__(self, 
                config_file: str="data/occlusion_net_test.yaml",
                target: str="car",
                min_test_size: int=800,
                confidence_threshold: int=0.4,
                viz: bool=True):
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        self.target = target
        self.viz = viz
        self.coco_demo = COCODemo(
            cfg,
            min_image_size=min_test_size,
            confidence_threshold=confidence_threshold,
        )
    
    def predict(self, img: np.ndarray):
        infer_result = {
            'boxes': [],
            'scores': [],
            'height_lines': [],
            'vized_img': None
        }
        
        predictions = self.coco_demo.compute_prediction(img)
        top_predictions = self.coco_demo.select_top_predictions(predictions)
        scores = top_predictions.get_field("scores")
        labels = top_predictions.get_field("labels")
        boxes = top_predictions.bbox

        if len(boxes) == 0:
            return infer_result

        kps = top_predictions.get_field("keypoints").keypoints

        vized_img = img.copy() if self.viz else None
        for box, score, label, current_kps in zip(boxes, scores, labels, kps):
            if label.item() != 1:
                continue

            boxpoints = [item for item in box.tolist()]
            infer_result['boxes'].append(boxpoints)
            infer_result['scores'].append(score.item())

            # Because kps also contain the third dimension which is not needed.
            current_kps = current_kps[:,0:2]
            infer_result['height_lines'].append(self._kps_to_height_line(current_kps.cpu().numpy()))
            if self.viz:
                x1, y1, x2, y2 = list(map(lambda x: int(x), boxpoints))
                vized_img = cv2.rectangle(vized_img, (x1,y1), (x2,y2), (200,200,0), 2)
                vized_img = cv2.line(vized_img, infer_result['height_lines'][-1][0].astype(int), infer_result['height_lines'][-1][1].astype(int), (0,255,0), 5)

        infer_result['vized_img'] = vized_img
        
        return infer_result
    
    def predict_batch(self, imgs):
        result = []
        for img in imgs:
            result.append(self.predict(img))

        return result

    def _kps_to_height_line(self, kps):
        NAMES = [
                'Right_Front_wheel',
                'Left_Front_wheel',
                'Right_Back_wheel',
                'Left_Back_wheel',
                'Right_Front_HeadLight',
                'Left_Front_HeadLight',
                'Right_Back_HeadLight',
                'Left_Back_HeadLight',
                'Exhaust',
                'Right_Front_Top',
                'Left_Front_Top',
                'Right_Back_Top',
                'Left_Back_Top',
                'Center'
            ]
        # Get all indices of interest
        top_point_index = NAMES.index("Right_Front_Top")
        bottom_line_point_1_index = NAMES.index(f"Right_Front_wheel")
        bottom_line_point_2_index = NAMES.index(f"Right_Back_wheel")

        # Get all points
        top_point = kps[top_point_index]
        dummy_point = top_point.copy()
        dummy_point[1] += 10

        bottom_line_point_1 = kps[bottom_line_point_1_index]
        bottom_line_point_2 = kps[bottom_line_point_2_index]

        bottom_point =  get_intersection(dummy_point, top_point, bottom_line_point_1, bottom_line_point_2)
        return np.array([top_point, bottom_point])

# @numba.njit()
def get_intersection(l1p1, l1p2, l2p1, l2p2):
    a1a2b1b2 = np.empty((4,2))
    a1a2b1b2[0] = l1p1
    a1a2b1b2[1] = l1p2
    a1a2b1b2[2] = l2p1
    a1a2b1b2[3] = l2p2

    # Get intersection of height line and top line
    _h = np.ones((4,3))
    _h[:,:2] = a1a2b1b2

    l1 = np.cross(_h[0], _h[1])
    l2 = np.cross(_h[2], _h[3])
    x, y, z = np.cross(l1, l2)

    # We compromise that top line and height line will always intersect.
    # if z == 0:
    #     return (float('inf'), float('inf'))
    intersection = np.empty((2,))
    intersection[0] = x/z
    intersection[1] = y/z

    return intersection

import scipy


def bbox_iou(boxA, boxB):
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # ^^ corrected.

    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = xB - xA + 1
    interH = yB - yA + 1

    # Correction: reject non-overlapping boxes
    if interW <=0 or interH <=0 :
        return -1.0

    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])

    if n_pred > n_true:
      # there are more predictions than ground-truth - add dummy rows
      diff = n_pred - n_true
      iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((diff, n_pred), MIN_IOU)), 
                                  axis=0)

    if n_true > n_pred:
      # more ground-truth than predictions - add dummy columns
      diff = n_true - n_pred
      iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((n_true, diff), MIN_IOU)), 
                                  axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label 

if __name__=="__main__":
    import random

    model = OcclusionNetModel()
    input_image = cv2.imread("demo/demo.jpg")
    # input_image = cv2.imread("crop_truck.png")

    result = model.predict(input_image)

    boxes = np.array(result['boxes'])
    fake_boxes = result['boxes'].copy()
    random.shuffle(fake_boxes)
    fake_boxes = np.array(fake_boxes)*1.4
    print(f"boxes_true: {boxes} - boxes_pred: {fake_boxes}")

    idxs_true, idxs_pred, _, _ = match_bboxes(boxes, fake_boxes)
    print(f"idxs_true: {idxs_true} - idxs_pred: {idxs_pred}")
