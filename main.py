import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import argparse
import os
from PIL import Image, ImageDraw

def draw_boxes(img, boxes, outpath):
    
    x, y, w, h = boxes
    
    draw = ImageDraw.Draw(img)
    draw.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline="red", width=5)

    img.save(outpath)

    return img

def answering(model, key_frames, outpath):
    
    outpath = '.'.join(outpath.split(".")[:-1])
    os.makedirs(outpath, exist_ok = True)
    
    img1 = key_frames["start"]["img"]
    img2 = key_frames["mid"]["img"]
    img3 = key_frames["end"]["img"]

    # all tracked id
    start = [track_id for track_id in key_frames["start"] if track_id != "img"]
    mid = [track_id for track_id in key_frames["mid"] if track_id != "img"]
    end = [track_id for track_id in key_frames["end"] if track_id != "img"]
    track_ids = list(set(start + mid + end))
    track_ids.sort()
    
    track_person = []
    
    for track_id in track_ids:
        x1, y1, w1, h1 = key_frames["start"].get(track_id, (0, 0, 0, 0))
        x2, y2, w2, h2 = key_frames["mid"].get(track_id, (0, 0, 0, 0))
        x3, y3, w3, h3 = key_frames["end"].get(track_id, (0, 0, 0, 0))
        draw_img1 = draw_boxes(img1, (x1, y1, w1, h1), os.path.join(outpath, f"{track_id}_start.jpg")) if w1 != 0 else None
        draw_img2 = draw_boxes(img2, (x2, y2, w2, h2), os.path.join(outpath, f"{track_id}_mid.jpg")) if w2 != 0 else None
        draw_img3 = draw_boxes(img3, (x3, y3, w3, h3), os.path.join(outpath, f"{track_id}_end.jpg")) if w3 != 0 else None

        track_person.append([draw_img1, draw_img2, draw_img3])
        
        

def main(args):
    
    # load model
    track_model = YOLO(args.track_model)
    mllm = 1
    
    # load and save video
    cap = cv2.VideoCapture(args.video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outpath = os.path.join(args.output_path, f"track_{os.path.basename(args.video_path)}")
    out = cv2.VideoWriter(outpath, fourcc, fps, (frame_width, frame_height))
    
    # tracking
    track_history = defaultdict(lambda: [])
    key_frames = {
        "start": {},
        "mid": {},
        "end": {}
    }
    mid_frames = total_frames // 2
    count = 0
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = track_model.track(frame, persist=True)
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            if count == 0:
                selected_frame = "start"
            elif count == mid_frames:
                selected_frame = "mid"
            else: # record the lastest frame
                selected_frame = "end"
                key_frames["end"] = {} # clear cache
                
            key_frames[selected_frame]["img"] = Image.fromarray(results[0].orig_img[:,:, [2, 1, 0]])

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                
                key_frames[selected_frame][track_id] = (x, y, w, h)
                
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            # cv2.imshow("YOLO11 Tracking", annotated_frame)
            out.write(annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
            
        count += 1

    # Release the video capture object and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    answering(mllm, key_frames, outpath)
    
    
if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--track_model", type=str, default="ckpts/yolo11n-pose.pt", help="Path to the detection and pose estimation model")
    args.add_argument("--mllm", type=str, default="ckpts/qwen2-vl", help="Path to VQA model")
    args.add_argument("--video_path", type=str, default="data/video1.mp4")
    args.add_argument("--output_path", type=str, default="outputs") 
    
    args = args.parse_args()
    
    main(args)