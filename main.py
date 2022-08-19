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

if __name__=="__main__":
    model = OcclusionNetModel()
    input_image = cv2.imread("demo/demo.jpg")
    # input_image = cv2.imread("crop_truck.png")

    result = model.predict(input_image)
    cv2.imwrite("docker_inference_img.png",result['vized_img'])
