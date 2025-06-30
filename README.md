Overview
This project performs automated detection, tracking, and re-identification of soccer players (and the ball) in video footage using a fine-tuned Ultralytics YOLOv11 model. The pipeline processes input video(s), assigns consistent IDs to each player across frames, and outputs both an annotated video and a JSON file with detailed tracking data.
Model: Fine-tuned YOLOv11 (best.pt) for players and ball.

Model Download
Download the provided YOLOv11 model here: https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view

Setup and Usage
1. Clone the Repository
git clone https://github.com/Projectworkpls/Soccer-15sec.git
cd Soccer-15sec
2. Prepare Your Files
Place the downloaded YOLOv11 model (best.pt) in the project directory.

Place your input video file(s) (e.g., .mp4) in the project directory.

3. Install Dependencies
Ensure Python 3.8+ is installed.

Install required packages:
pip install ultralytics opencv-python torch

4. Configure Paths
Open internshalasoccer.py and set:

LOCAL_MODEL_PATH to the path of your best.pt model.

VIDEO_PATHS to your input video file(s).

Example:

python
LOCAL_MODEL_PATH = r"C:\Users\LENOVO\Downloads\best.pt"
VIDEO_PATHS = [r"C:\Users\LENOVO\Downloads\15sec_input_720p.mp4"]
5. Run the Code

python internshalasoccer.py
Output
Annotated Video:
Saved in the outputs/ folder as <inputname>_tracked.mp4 (bounding boxes and IDs).

Tracking Data:
Saved as <inputname>_tracks.json in the outputs/ folder.
Each entry contains the frame number, player ID, bounding box, and confidence.

Example JSON entry:

json
{
  "frame": 0,
  "id": 1,
  "bbox": [794, 470, 864, 566],
  "conf": 0.92
}
Advanced Usage
Tracker Selection:
By default, BoT-SORT is used for tracking and re-identification.
You can modify the tracker by changing the TRACKER_NAME variable (e.g., "bytetrack.yaml")2.

Re-Identification:
ReID is enabled by default with BoT-SORT. For custom ReID settings, see Ultralytics YOLOv11 tracking documentation.

Parameter Tuning:
You may adjust confidence thresholds or tracker parameters in the script for your specific use case.

References
Ultralytics YOLOv11 Documentation1

YOLOv11-JDE: Fast and Accurate Multi-Object Tracking with Self-Supervised Re-ID4

Contact
For questions or issues, please contact parth.acharya2003@gmail.com.
