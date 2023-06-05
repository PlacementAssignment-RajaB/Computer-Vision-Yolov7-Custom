import numpy as np
import cv2
import onnxruntime as ort

# Load the ONNX model
model_path = 'yolov7-tiny.onnx'
session = ort.InferenceSession(model_path)

# Preprocess input image
image_path = 'img_1.png'
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (640, 640))
input_data = np.expand_dims(resized_image.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0

# Run inference
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]
outputs = session.run(output_names, {input_name: input_data})

# Process the output
# Example: Retrieve bounding boxes and class labels
boxes = outputs[0]
class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
print(boxes)

# Perform further post-processing or visualization based on the specific requirements of your application
import cv2

image = cv2.imread(image_path)


# Draw bounding boxes on the image
for box in boxes[[]]:
    x, y, w, h = box[:4]  # Extract box coordinates (adjust based on the structure of your bounding box data)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around the object

# Display the image
cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
