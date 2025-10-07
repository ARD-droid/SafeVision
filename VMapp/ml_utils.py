import os
import cv2
import numpy as np
import ffmpeg
import ultralytics
from PIL import Image
from opennsfw2 import predict_image
from nudenet import NudeDetector
import torch

# Ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load YOLO weapon detection model (adjust path if necessary)
with torch.serialization.safe_globals([ultralytics.nn.tasks.DetectionModel]):
   weapon_model = torch.hub.load(
    "ultralytics/yolov5", 
    "custom", 
    path="C:\\Users\\DELL\\VS Code\\VMproject\\Weapon-Detection-YOLO\\best (3).pt", 
    device='cpu'  # Force using CPU
)
# Extract frames from video
def extract_frames(video_path, frame_dir):
    ensure_dir(frame_dir)
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_path = os.path.join(frame_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    video_capture.release()
    print(f"[INFO] Extracted {frame_count} frames to {frame_dir}")

# Extract audio from video (if present)
def extract_audio(video_path, audio_output_path):
    try:
        probe = ffmpeg.probe(video_path)
        has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])

        if not has_audio:
            print("[WARNING] No audio stream found in video.")
            return False

        ffmpeg.input(video_path).output(audio_output_path).run()
        print(f"[INFO] Audio extracted to {audio_output_path}")
        return True

    except ffmpeg.Error as e:
        print(f"[ERROR] ffmpeg error during audio extraction: {e}")
        return False

# Analyze frames using OpenNSFW2 and filter based on threshold
def analyze_frames(frame_dir, threshold=0.2):
    flagged_frames = []
    for frame_name in os.listdir(frame_dir):
        if frame_name.endswith(".jpg"):
            frame_path = os.path.join(frame_dir, frame_name)
            try:
                image = Image.open(frame_path).convert("RGB")
                nsfw_score = predict_image(image)
                print(f"[DEBUG] Frame: {frame_name}, NSFW Score: {nsfw_score}")
                if nsfw_score > threshold:
                    flagged_frames.append(frame_path)
            except Exception as e:
                print(f"[ERROR] Error processing {frame_path}: {e}")
    return flagged_frames

# Function to detect weapons in a frame using YOLO
def detect_weapons(frame):
    results = weapon_model(frame)  # Run the model on the frame
    weapons_detected = []
    
    # Check for weapons detection (we assume weapon class ID is 0, adjust if needed)
    for *box, conf, cls in results.xyxy[0]:
        if conf > 0.5:  # Confidence threshold
            label = results.names[int(cls)]
            if 'weapon' in label.lower():  # Adjust class names based on the model
                weapons_detected.append(box)
    return weapons_detected

# Blur specific sensitive content in flagged frames
def blur_flagged_frames(flagged_frames, moderated_dir, nsfw_threshold=0.2, blur_margin=10):
    ensure_dir(moderated_dir)
    detector = NudeDetector()

    deeper_blur_labels = {
        'BUTTOCKS_EXPOSED', 'BUTTOCKS_COVERED',
        'FEMALE_GENITALIA_EXPOSED', 'FEMALE_GENITALIA_COVERED',
        'MALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_COVERED',
        'ANUS_EXPOSED'
    }

    all_sensitive_labels = {
        'FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED',
        'MALE_GENITALIA_EXPOSED', 'BUTTOCKS_EXPOSED', 'ANUS_EXPOSED',
        'FEMALE_BREAST_COVERED', 'FEMALE_GENITALIA_COVERED',
        'MALE_GENITALIA_COVERED', 'BUTTOCKS_COVERED',
        'BELLY_EXPOSED', 'BELLY_COVERED',
        'FEET_EXPOSED', 'FEET_COVERED',
        'ARMPITS_EXPOSED', 'ARMPITS_COVERED',
        'BACK_EXPOSED', 'BACK_COVERED',
        'CLOTHED_BUT_SEXY', 'CLOTHED_BUT_VERY_SEXY', 'SEXY_POSE',
        'LINGERIE', 'BDSM', 'SEXUAL_ACTIVITY',
        'SEXUAL_OBJECT', 'SEX_TOY', 'KISS', 'INTIMATE_KISSING'
    }

    for frame_path in flagged_frames:
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[WARNING] Unable to read frame: {frame_path}")
            continue

        height, width, _ = frame.shape
        detections = detector.detect(frame_path)

        if detections:
            print(f"[INFO] NudeNet detections found in {frame_path}")
            print(f"[DEBUG] Raw detections from NudeNet: {detections}")
            for detection in detections:
                label = detection['class']
                if label in all_sensitive_labels:
                    x, y, w, h = map(int, detection['box'])
                    x1 = max(x - blur_margin, 0)
                    y1 = max(y - blur_margin, 0)
                    x2 = min(x + w + blur_margin, width)

                    # Apply deeper vertical blur for some labels
                    if label in deeper_blur_labels:
                        extra_height = int(0.5 * h)
                        y2 = min(y + h + extra_height, height)
                    else:
                        y2 = min(y + h + blur_margin, height)

                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                        black_overlay = np.zeros_like(blurred_roi)
                        alpha = 0.6
                        darker_roi = cv2.addWeighted(blurred_roi, 1 - alpha, black_overlay, alpha, 0)
                        frame[y1:y2, x1:x2] = darker_roi
                        print(f"[INFO] Blurred: {label} in {frame_path}")
        
        # Weapon detection
        weapon_boxes = detect_weapons(frame)
        for box in weapon_boxes:
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                frame[y1:y2, x1:x2] = blurred_roi
                print(f"[INFO] Blurred weapon in {frame_path}")

        moderated_frame_path = os.path.join(moderated_dir, os.path.basename(frame_path))
        cv2.imwrite(moderated_frame_path, frame)
        print(f"[DEBUG] Moderated frame saved: {moderated_frame_path}")

# Simple decision helper
def make_decision(flagged_frames):
    return "Content Moderated" if flagged_frames else "Content Safe"

# Full moderation pipeline function
def moderate_video(video_path, frame_dir, moderated_dir, audio_output_path):
    print("[INFO] Starting video moderation...")

    # Extract frames
    extract_frames(video_path, frame_dir)

    # Extract audio
    extract_audio(video_path, audio_output_path)

    # Analyze NSFW frames
    flagged_frames = analyze_frames(frame_dir)

    # Blur flagged frames
    blur_flagged_frames(flagged_frames, moderated_dir)

    # Make a moderation decision
    decision = make_decision(flagged_frames)
    print(f"[RESULT] Moderation Decision: {decision}")
    return decision
