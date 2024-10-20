# Tennis Ball AI Tracker

An object detection model designed to track a tennis ball during a tennis match and map its position onto a 2D tennis court.

## Overview

This repository contains a **Streamlit-based application** that allows users to upload their own tennis video clips or select from preloaded videos to track the tennis ball using an object detection model. The app displays the video with the tracked ball and offers options to adjust the model's confidence level for more accurate tracking.

## Features

- Upload your own tennis video clip or choose from preloaded videos.
- Track the tennis ball using a custom object detection model.
- Adjust the model's confidence level for detection sensitivity.
- Display the video with the tracked ball directly in the application.

## Output Example

![Tennis Ball AI Tracker Output](https://github.com/vasquezsebastian459/tennisball_tracker/blob/main/output_videos/tennis_test_video.gif)

## Requirements

- Python 3.x
- Streamlit
- OpenCV
- Subprocess
- Tempfile
- Datetime
- Sys

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/vasquezsebastian459/tennisball_tracker.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:

    ```bash
    streamlit run streamlit_page.py
    ```

## Usage

1. Upload your own tennis video clip or select from preloaded videos.
2. Adjust the model's confidence level for better detection accuracy.
3. Click the **Run** button to start the tracking process.
4. The application will display the video with the tracked ball in real time.

## Model

The object detection model used in this application is a custom-trained model using YOLO v8. It was trained on a dataset of tennis videos and optimized specifically for tracking tennis balls in real-time.

## Future Work

- Improve the model's accuracy and speed for more efficient tracking.
- Add additional features like ball speed and trajectory tracking.
- Integrate the application with other advanced tennis analytics tools.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a description of your changes.

---

Feel free to reach out for any questions or collaboration ideas. Thank you for checking out the **Tennis Ball AI Tracker**!
