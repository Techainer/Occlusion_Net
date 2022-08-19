import cv2
from mlchain.client import Client

model = Client(api_address='http://localhost:8001').model()

input_image = cv2.imread("crop_truck.png")

result, vized_img = model.predict(input_image)
print(result)
if vized_img is not None:
    cv2.imwrite("inference_result.jpg", vized_img)
