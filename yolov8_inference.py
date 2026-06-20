import numpy as np
import cv2
import onnxruntime as rt

# Step 1: Load the ONNX model
model_path = "best.onnx"
session = rt.InferenceSession(model_path)

# Step 2: Get input and output node names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Input node: {input_name}")
print(f"Output node: {output_name}")

# Step 3: Load a test image
image = cv2.imread("test_image.jpg")
print(f"Image shape: {image.shape}")

# Step 4: Preprocess the image
# Resize to 640x640 (what YOLOv8 expects)
resized = cv2.resize(image, (640, 640))

# Normalize to [0, 1]
normalized = resized.astype(np.float32) / 255.0

# Transpose from HxWx3 to 3xHxW
transposed = np.transpose(normalized, (2, 0, 1))

# Add batch dimension: 3xHxW -> 1x3xHxW
input_tensor = np.expand_dims(transposed, axis=0)

print(f"Input tensor shape: {input_tensor.shape}")

# Step 5: Run inference
outputs = session.run([output_name], {input_name: input_tensor})

print(f"Output shape: {outputs[0].shape}")
print(f"First few predictions: {outputs[0][0][0][:10]}")