import cv2
import torch
from ultralytics import YOLO
import os
import json

LOCAL_MODEL_PATH = r"C:\Users\LENOVO\Downloads\best.pt"
VIDEO_PATHS = [r"C:\Users\LENOVO\Downloads\15sec_input_720p.mp4"]
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRACKER_NAME = "botsort.yaml"

def load_yolo_model():
    return YOLO(LOCAL_MODEL_PATH).to(DEVICE)

def track_and_annotate(video_path, model, output_dir, tracker_name="botsort.yaml"):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")
    output_json_path = os.path.join(output_dir, f"{video_name}_tracks.json")
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    all_tracks = []
    frame_idx = 0
    print(f"Processing {video_path} ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(
            source=frame,
            persist=True,
            tracker=tracker_name,
            show=False,
            stream=False,
            verbose=False
        )
        frame_tracks = []
        for det in results:
            if det.boxes is not None:
                for i, box in enumerate(det.boxes.xyxy):
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    player_id = int(det.boxes.id[i]) if det.boxes.id is not None else -1
                    conf = float(det.boxes.conf[i])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"ID:{player_id}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    frame_tracks.append({
                        "frame": frame_idx,
                        "id": player_id,
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf
                    })
        all_tracks.extend(frame_tracks)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    with open(output_json_path, "w") as f:
        json.dump(all_tracks, f, indent=2)
    print(f"Annotated video saved to {output_video_path}")
    print(f"Tracking data saved to {output_json_path}")

def main():
    model = load_yolo_model()
    for video_path in VIDEO_PATHS:
        track_and_annotate(video_path, model, OUTPUT_DIR, TRACKER_NAME)

if __name__ == "__main__":
    main()
