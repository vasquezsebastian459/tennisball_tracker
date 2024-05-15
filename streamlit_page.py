import streamlit as st
import subprocess
import os
import tempfile
from datetime import datetime

# Function to update Streamlit progress bar
def update_progress(progress):
    progress_bar.progress(progress)

# Function to run main.py with selected options and update progress
def run_main(model_number, input_video_path, output_video_path,confidence ,progress_callback):
    # Build the command to run main2.py with arguments
    
    command = f"python -u main3.py --model {model_number} --input_video_path {input_video_path} --output_video_path {output_video_path} --conf {confidence}"
    # python -u main2.py --model_num 3 --video_name djokovic_sonego.mp4 --conf 0.1
    # python -u main3.py --model 3 --input_video_path input_videos/djokovic_sonego.mp4 --output_video_path output_videos/djokovic_sonego_model2_conf1.mp4 --conf 0.1
    print(command)
    # Start the subprocess
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, universal_newlines=True)
    
    # Read output continuously
    for line in process.stdout:
        print(line.strip())  # Optional: Output to console for debugging
        if 'progress' in line:
            progress_percent = float(line.split()[-1])
            progress_callback(progress_percent / 100)  # Normalize to 0-1 if needed
    

    # Read errors continuously
    for line in process.stderr:
        st.error(line.strip())  # Output errors to Streamlit
        print(f"Error: {line.strip()}")  # Output errors to console for debugging
    
    # Finalize process
    process.stdout.close()
    process.stderr.close()
    return_code = process.wait()
    # Check return code
    if return_code == 0:
        return True, "Clip processed Sucessfully!"
    else:
        return False, "Error in processing"

# Streamlit User Interface
st.title('Tennis Ball Tracking')

# Model selection dropdown
# model_number = st.selectbox('Choose a model:', options=[1, 2, 3, 4], format_func=lambda x: f'Model {x}')

st.header('Models Descriptions')
st.subheader('1. YOLOv8 Initial Model')
st.write( "Trained with default settings for 100 epochs, achieving a robust mAP50 of 0.689 on unseen test data, demonstrating a solid foundational performance.")

st.subheader('2. YOLOv8 Extended Training')
st.write( "Enhanced through an additional 200 epochs of training, this model version improved its mAP50 to 0.73, showcasing increased accuracy and data pattern capture without overfitting.")

st.subheader('3. YOLOv8 Optimized Model')
st.write(  "Optimized with targeted hyperparameter tuning using Ray Tune, achieving a refined mAP50 of 0.733, marking a 5% precision increase and superior adaptation to varying conditions.")

st.header('Try the Model Yourself!')


# List of models
models = ['YOLOv8 - Baseline', 'YOLOv8 - Enhanced', 'YOLOv8 - Optimized']

# Dropdown to select a model
selected_model = st.selectbox("Select a Model:", models)

model2num = {
    'YOLOv8 - Baseline': 1,
    'YOLOv8 - Enhanced': 2,
    'YOLOv8 - Optimized': 4
}

# Get the model number based on the selected model
model_number = model2num[selected_model]
# Confidence interval slider
confidence = st.slider('Select Confidence Interval:', min_value=0.05, max_value=0.5, value=0.1, step=0.05)


option = st.radio(
    "How would you like to proceed?",
    ('Choose from preloaded videos', 'Upload your own video clip')
)
if option == 'Choose from preloaded videos':
    # Video selection dropdown
    video_options = ['djokovic_korda.mp4', 'djokovic_sonego.mp4', 'nadal_clay.mp4', 'nadal_thiem.mp4']
    video_name = st.selectbox('Choose a tennis clip:', options=video_options)

    

    # Display selected video if it exists
    input_video_path = f'./input_videos/{video_name}'

    if os.path.exists(input_video_path):
        st.video(input_video_path)
    else:
        st.error("Selected video file does not exist. Please check the file path.")

    # Process video on button click
    if st.button('Run Tracking'):
        progress_bar = st.progress(0)
        with st.spinner('Running Tennis Tracker...'):

            output_video_path = f"./output_videos/{video_name.split('.')[0]}_model{model_number}_conf{str(confidence).split('.')[1]}.mp4"


            success, message = run_main(model_number, input_video_path,output_video_path, confidence,update_progress)
            if success:
                st.success(message)
    
                #output_video_path = f"./output_videos/{video_name.split('.')[0]}_model{model_number}_conf{str(confidence).split('.')[1]}.mp4"

                st.subheader(f'Results for: {video_name}') # replace with video_name
                if os.path.exists(output_video_path):
                    st.video(output_video_path)

                    st.write("If results are not satisfactory try increasing or decreasing confidence interval or changing model!")
            else:
                st.error(message)

elif option == 'Upload your own video clip':
    # File uploader widget
    uploaded_file = st.file_uploader("Upload your own tennis video clip", type=['mp4'])

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Use tempfile to save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name

        # Display the video
        st.video(video_path)
        st.write(f"Your uploaded video is ready for processing!")

        # Process video on button click
        if st.button('Run Tracking'):
            progress_bar = st.progress(0)
            with st.spinner('Running Tennis Tracker...'):
                # Get the current date and time
                current_time = datetime.now()

                # Format the time as a string in the specified format
                time_string = current_time.strftime('%Y%m%d%H%M%S')

                output_video_path = f'/var/folders/g8/p4xg79w95m51sym2wv8f09t80000gn/T/output_video_{time_string}.mp4'

                success, message = run_main(model_number, video_path,output_video_path, confidence,update_progress)
                if success:
                    st.success(message)
                    # Construct the output path using filename from the tempfile
                    if os.path.exists(output_video_path):
                        st.subheader(f'Results for your uploaded video:')
                        st.video(output_video_path)
                        st.write("If results are not satisfactory try increasing or decreasing confidence interval or changing model!")
                    else:
                        st.error(f"Output video not found: {output_video_path}")
                else:
                    st.error(message)
        
    else:
        st.write("Please upload a video file to proceed.")
