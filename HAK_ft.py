from ultralytics import YOLO
import cv2

video_path = "data/builders.mp4"  
output_path = "outputs/builders_out_ft.mp4"  

model = YOLO("yolo_custom.pt")  

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[2, 4, 5, 8, 9, 10, 14, 15, 16])
#    results = model(frame)
    annotated_frame = results[0].plot()

    out.write(annotated_frame)

cap.release()
out.release()

print(f"Output video saved at: {output_path}")





