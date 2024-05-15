import cv2
from ultralytics import YOLO
import numpy as np
import json
from court_line_detector import CourtLineDetector
import subprocess
import os
import pickle
import argparse
from process_video import process_video
from cache_videos import cache_results,load_cached_results

# Setup argument parser
parser = argparse.ArgumentParser(description='Run tennis ball tracking model.')
parser.add_argument('--model', type=int, required=True, help='Model number to use (1, 2, 3, or 4)')
parser.add_argument('--input_video_path', type=str, required=True, help='Name of the input video file')
parser.add_argument('--output_video_path', type=str, required=True, help='Name of the output video file')
parser.add_argument('--conf', type=str, required=True, help='Name of the input video file')

# Parse arguments
args = parser.parse_args()

# Assign arguments to variables
model_num = args.model
video_path = args.input_video_path
result_video_path = args.output_video_path
confidence = float(args.conf)

# MODEL
model_path =f"YOLO/runs/detect/train{model_num}/weights/best.pt"

def main(video_path,result_video_path,model_path,confidence):
    print("Running ...")

    # Initialize the model
    
    model = YOLO(model_path)
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        exit()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ball_detection_count = 0  # Initialize counter for frames with ball detection
    
    # Prepare Video Writer using the input video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    output_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'mp4v'), output_frame_rate, (frame_width, frame_height))
    
    # Load source keypoints and initialize CourtLineDetector
    with open('mini_court/keypoints.json', 'r') as file:
        source_keypoints = json.load(file)['keypoints']
    source_points = np.array([source_keypoints[str(i)] for i in range(4)], dtype='float32')
    
    
    # Load source keypoints and initialize CourtLineDetector
    court_model_path = "mini_court/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    
    # Detect keypoints from the first frame for overlay in every frame
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
    else:
        court_keypoints = court_line_detector.predict(first_frame)
    
    court_keypoints = court_line_detector.predict(first_frame)
    keypoints_list = [[court_keypoints[i], court_keypoints[i + 1]] for i in range(0, len(court_keypoints), 2)]
    destination_points = np.array(keypoints_list[:4])
    
    # Mini court image dimensions for overlay
    mini_court_width = 149   # Double the size for visibility
    mini_court_height = 276 

    # Load and resize the mini court image
    tennis_court_img = cv2.imread('mini_court/tennis_court_with_keypoints.png')
    tennis_court_img_resized = cv2.resize(tennis_court_img, (mini_court_width, mini_court_height))
    
    ball_detection_count = 0  # Initialize counter for frames with ball detection
    video_cached = False

    ball_detections = load_cached_results(result_video_path)

    # This checks if this video is old or new
    if ball_detections is None: 
        print("No Cache found")
        ball_detections = {} # Initializing dicitonary to store new ball detections
    else:
        print("Cached results found. Skipping detection...")
        video_cached = True
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        ball_center = None

        tennis_court_img_resized = cv2.resize(tennis_court_img, (mini_court_width, mini_court_height))

        if video_cached: # How to process cached videos
            ball_bbox = ball_detections[current_frame_index]

            if ball_bbox:
                x1, y1, x2, y2 = [int(coord) for coord in ball_bbox]

                ball_center = ((ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2)

                # Perspective transformation and drawing logic
                M = cv2.getPerspectiveTransform(destination_points, source_points)
                ball_pos = np.array([ball_center], dtype='float32').reshape(-1, 1, 2)
                ball_pos_mini_court = cv2.perspectiveTransform(ball_pos, M).reshape(-1, 2)
                ball_detection_count += 1  # Increment the detection count

                # Draw the bounding box around the detected tennis ball on the original frame 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow bounding box

                # Draw the ball position on the resized tennis court image for overlay
                cv2.circle(tennis_court_img_resized, (int(ball_pos_mini_court[0][0]), int(ball_pos_mini_court[0][1])), 10, (0, 255, 255), -1)
        

        else: # How to process New videos
            # Detect the tennis ball
            results = model.predict(frame, verbose=False,conf=confidence )[0] #conf=confidence

            if results.boxes:
                # Access the first box directly assuming it is the first detection
                ball_bbox = results.boxes[0].xyxy.tolist()[0]
                x1, y1, x2, y2 = [int(coord) for coord in ball_bbox]

                # Calculate the center of the ball's bb
                ball_center = ((ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2)
                
                ball_detection_count += 1  # Increment the detection count

                # Perspective transformation and drawing logic
                M = cv2.getPerspectiveTransform(destination_points, source_points)
                ball_pos = np.array([ball_center], dtype='float32').reshape(-1, 1, 2)
                ball_pos_mini_court = cv2.perspectiveTransform(ball_pos, M).reshape(-1, 2)

                # Draw the bounding box around the detected tennis ball on the original frame 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow bounding box

                # Draw the ball position on the resized tennis court image for overlay
                cv2.circle(tennis_court_img_resized, (int(ball_pos_mini_court[0][0]), int(ball_pos_mini_court[0][1])), 10, (0, 255, 255), -1)
                
            else:
                ball_bbox = None
               
        
        # Draw the court keypoints on the original frame
        if ret:
            frame = court_line_detector.draw_keypoints(frame, court_keypoints)
    
        # Overlay the mini court image in the top right
        overlay_x_start = frame_width - mini_court_width
        overlay_y_start = 0  # Top corner
        frame[overlay_y_start:overlay_y_start+mini_court_height, overlay_x_start:overlay_x_start+mini_court_width] = tennis_court_img_resized
    
        # Write the frame with the overlay to the output video
        #print("writing frame's ball's bbox:",current_frame_index,  ball_bbox)
        ball_detections[current_frame_index] = ball_bbox
        #print(f"Frame: {current_frame_index}/{total_frames} Ball Position -> {ball_bbox}")
        print(f"progress {current_frame_index / total_frames * 100}", flush=True)
        out.write(frame)

    # After processing, save the results to cache
    cache_results(result_video_path, ball_detections)
    
    # Cleanup
    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    print(f"progress {total_frames / total_frames * 100}", flush=True)
    print("Detections Completed!")
    
    print(f"Total frames in video: {total_frames}")
    print(f"Frames with ball detection: {ball_detection_count}")
    process_video(result_video_path)

main(video_path,result_video_path,model_path,confidence)
print("Finished!")
print(f"Video saved at: {result_video_path}")


