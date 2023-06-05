import cv2
import numpy as np
import onnxruntime as ort

class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]


# Load the YOLOv7 ONNX model
model_path = 'yolov7-tiny.onnx'
session = ort.InferenceSession(model_path)

# Get the input name
input_name = session.get_inputs()[0].name

# Load and preprocess the input image
image_path = 'img_1.png'
image = cv2.imread(image_path)
original_height, original_width, _ = image.shape

input_size = (640, 640)  # The input size expected by the model
resized_image = cv2.resize(image, input_size)
input_data = np.transpose(resized_image, (2, 0, 1))  # Transpose to (C, H, W)
input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
input_data = input_data.astype(np.float32) / 255.0  # Normalize to [0, 1]

# Run inference
outputs = session.run(None, {input_name: input_data})
# Process the outputs
# ...
def rescale_boxes(boxes, input_width, input_height, original_width, original_height):
    rescaled_boxes = []
    width_ratio = original_width / input_width
    height_ratio = original_height / input_height

    for box in boxes:
        x = box[1]
        y = box[2]
        w = box[3]
        h = box[4]
        label = box[5]
        conf = box[6]


        # Rescale the bounding box coordinates
        x_rescaled = int(x * width_ratio)
        y_rescaled = int(y * height_ratio)
        w_rescaled = int(w * width_ratio)
        h_rescaled = int(h * height_ratio)

        rescaled_boxes.append([x_rescaled, y_rescaled, w_rescaled, h_rescaled,label,conf])

    return rescaled_boxes

rescaled_boxes = rescale_boxes(outputs[0], 640, 640, original_width, original_height)

print(rescaled_boxes)

# Perform post-processing and visualization of the results
# ...
bounding_boxes = outputs[0]
print(len(bounding_boxes))
for box in rescaled_boxes:
    print(box)
    x = box[0]
    y = box[1]
    width = box[2]
    height = box[3]
    label = class_names[int(box[4])]
    conf = box[5]
    cv2.rectangle(image, (int(x), int(y)), (int(x + width), int(y + height)), (255, 0, 0), 2)

    text = f"{label}: {conf:.2f}"
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0 , 255), 2)

# Display the image
cv2.imshow('Image with Bounding Boxes', image)
cv2.imwrite('img_predicted.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



print(class_names[int(outputs[0][0][5])])