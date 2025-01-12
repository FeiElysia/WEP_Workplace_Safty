#!/usr/bin/env python
# coding: utf-8

# In[12]:

# Step 2: Import required modules
from ultralytics import YOLO
import cv2

# Step 3: Use the local video file
video_path = "test_video_2.mp4"  # Replace with the actual path to your video
output_path = "output_video_2.mp4"  # Path for the output video

# Step 4: Load the YOLOv11n model and process the video
model = YOLO("yolo11n.pt")  # Load the lightweight YOLOv11n model

# Open the video
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer for saving the result
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Annotate the frame with the results
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

cap.release()
out.release()

# Step 5: Output the path of the resulting video
print(f"Output video saved at: {output_path}")


# In[ ]:




