import os
import cv2

from ml_utils import blur_flagged_frames   # <-- adjust if your file has a different name

frame_dir = "C:/Users/DELL/VS Code/VMproject/myproject/media/frames/ARD1"  # put your actual folder name here
moderated_dir = "C:/Users/DELL/VS Code/VMproject/myproject/media/moderated_frames"

blur_flagged_frames(frame_dir, moderated_dir)