import subprocess
import os

def process_video(video_path):
    print("Saving ..")
    temp_output_path = './output_videos/mini_court_video3_temp.mp4'  # Temporary output file
    
    # FFmpeg command for conversion
    command = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '22',
        '-c:a', 'copy',
        temp_output_path
    ]
    # Execute the command
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Check the result and replace the original file if successful
    if process.returncode == 0:
        print("Saving successful.")
        # Remove the original file
        os.remove(video_path)
        # Rename the temporary file to the original file name
        os.rename(temp_output_path, video_path)
    else:
        print("Conversion failed.")
        print(process.stderr)