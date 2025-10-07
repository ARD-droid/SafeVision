import os
import cv2
import torch
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .forms import VideoForm
from .models import Video
from .ml_utils import extract_frames, extract_audio, ensure_dir, analyze_frames, blur_flagged_frames, make_decision, moderate_video, detect_and_blur_violent_objects

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=False)

def home(request):
    return render(request, 'home.html')

def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the video instance but don't commit yet
            video_instance = form.save(commit=False)
            user_title = form.cleaned_data['title'].replace(" ", "_")  # Sanitize title for folder naming
            uploaded_file = request.FILES['video_file']
            ext = os.path.splitext(uploaded_file.name)[1]  # Get the file extension (e.g., .mp4)

            # Save the video file with the user-defined title
            video_filename = f"{user_title}{ext}"
            video_path = os.path.join(settings.MEDIA_ROOT, 'videos', video_filename)

            # Save the video manually to the videos folder in the media directory
            with open(video_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Set the video file path in the model
            video_instance.video_file.name = os.path.join('videos', video_filename)
            video_instance.save()

            # === Frame and Audio output directories ===
            frame_output_dir = os.path.join(settings.MEDIA_ROOT, 'frames', user_title)  # Folder name same as title
            audio_output_path = os.path.join(settings.MEDIA_ROOT, 'audio', f"{user_title}.mp3")

            os.makedirs(frame_output_dir, exist_ok=True)  # Create frames directory if it doesn't exist

            # Extract frames
            print("Extracting frames...")
            extract_frames(video_path, frame_output_dir)
            print(f"Frames extracted to: {frame_output_dir}")

            # Extract audio
            print("Extracting audio...")
            if extract_audio(video_path, audio_output_path):
                print(f"Audio extracted to: {audio_output_path}")
            else:
                print("Audio extraction failed or not present.")

            # Blur flagged frames here
            moderated_dir = os.path.join(settings.MEDIA_ROOT, 'moderated_frames', user_title)
            os.makedirs(moderated_dir, exist_ok=True)

            # Analyze frames (dummy flagged frames used for testing)
            print("[DEBUG] Analyzing frames for inappropriate content...")
            flagged_frames = analyze_frames(frame_output_dir)  # Get flagged frames (inappropriate areas detected)
            print(f"[DEBUG] Flagged frames: {flagged_frames}")

            # Blurring flagged frames
            print("[DEBUG] Blurring flagged frames...")  
            blur_flagged_frames(flagged_frames, moderated_dir)
            print("[DEBUG] Finished blurring flagged frames.")

            # Detect and blur violent objects (knives, guns, etc.)
            print("[DEBUG] Detecting and blurring violent content...")
            detect_and_blur_violent_objects(frame_output_dir, moderated_dir, model)  # Updated function call
            print("[DEBUG] Finished detecting and blurring violent objects.")

            # Redirect to the extracted frames view with the video ID and folder name
            return redirect('extracted_frames', video_id=video_instance.id, folder=user_title)
    else:
        form = VideoForm()
    return render(request, 'upload_video.html', {'form': form})

def video_list(request):
    videos = Video.objects.all()
    video_data = []

    for video in videos:
        video_path = video.video_file.path
        filename_wo_ext = os.path.splitext(os.path.basename(video_path))[0]
        frame_folder = f"{filename_wo_ext}_frames"
        video_data.append({
            'id': video.id,
            'name': os.path.basename(video.video_file.name),
            'frame_folder': frame_folder
        })

    return render(request, 'video_list.html', {'video_data': video_data})

def moderated_frames_view(request, video_id, folder):
    frame_folder = os.path.join(settings.MEDIA_ROOT, 'moderated_frames', folder)
    frame_urls = []

    print(f"[FRAME VIEW] Looking for frames in: {frame_folder}")
    
    # Check if the frames folder exists
    if os.path.exists(frame_folder):
        # Loop through the files in the folder and append valid image paths
        for frame_file in sorted(os.listdir(frame_folder)):
            if frame_file.endswith('.jpg') or frame_file.endswith('.png'):
                relative_path = os.path.join('frames', folder, frame_file)
                frame_urls.append(f"{settings.MEDIA_URL}frames/{folder}/{frame_file}")

    if not frame_urls:
        print("[FRAME VIEW] No frames found.")
    
    return render(request, 'moderated_frames.html', {
        'video_id': video_id,
        'frames': frame_urls
    })

def preprocess_video(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    video_path = video.video_file.path

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # === Directories ===
    frame_dir = os.path.join(settings.MEDIA_ROOT, "frames", base_name)
    moderated_dir = os.path.join(settings.MEDIA_ROOT, "Moderated_Frames", base_name)
    audio_output_path = os.path.join(settings.MEDIA_ROOT, 'censored_audio', f'{base_name}.wav')
    transcribe_audio = os.path.join(settings.MEDIA_ROOT, 'transcripts', f'{base_name}.txt')

    # === Create directories ===
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(moderated_dir, exist_ok=True)
    os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(transcribe_audio), exist_ok=True)

    # === Step 1: Extract Frames and Audio ===
    print(f"[PREPROCESS] Extracting frames for {video_path}")
    frames = extract_frames(video_path, frame_dir)
    print(f"[PREPROCESS] Total frames extracted: {len(frames)}")
    print(f"[PREPROCESS] Frames extracted to: {frame_dir}")

    print(f"[PREPROCESS] Extracting audio to {audio_output_path}")
    extract_audio(video_path, audio_output_path)

    # === Step 2: Analyze and Blur Frames ===
    print(f"[PREPROCESS] Analyzing frames for inappropriate content...")
    flagged_frames = analyze_frames(frames)
    print(f"[DEBUG] Flagged frames: {flagged_frames}")

    # Blur flagged frames
    print(f"[PREPROCESS] Blurring flagged frames to {moderated_dir}...")
    blur_flagged_frames(flagged_frames, moderated_dir)

    # Detect and blur violent objects (knives, guns, etc.)
    print(f"[PREPROCESS] Detecting and blurring violent content...")
    detect_and_blur_violent_objects(frame_dir, moderated_dir, model)
    print(f"[PREPROCESS] Finished detecting and blurring violent objects.")

    return redirect('moderated_frames', video_id=video.id, folder=base_name)

def transcribe_audio(request, video_id):
    video = get_object_or_404(Video, id=video_id)

    base_name = os.path.splitext(os.path.basename(video.video_file.name))[0]
    audio_path = os.path.join(settings.MEDIA_ROOT, 'censored_audio', f'{base_name}.wav')
    transcript_path = os.path.join(settings.MEDIA_ROOT, 'transcripts', f'{base_name}.txt')

    return render(request, 'transcription.html', {
        'transcript': open(transcript_path).read() if os.path.exists(transcript_path) else None,
        'audio': os.path.join(settings.MEDIA_URL, f'censored_audio/{base_name}.wav'),
        'video_id': video.id,
        'video': video
    })

def live_options_view(request):
    return render(request, 'live_options.html')
