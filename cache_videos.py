import os
import pickle


def cache_results(video_path, ball_detections):
    # Create a directory for cache files if it doesn't exist
    cache_dir = './cache'
    os.makedirs(cache_dir, exist_ok=True)

    # Define the cache file path
    cache_file = os.path.join(cache_dir, f"{os.path.basename(video_path)}.pkl")
    
    # Write results to the cache file
    with open(cache_file, 'wb') as f:
        pickle.dump(ball_detections, f)


def load_cached_results(video_path):
    cache_file = f'./cache/{os.path.basename(video_path)}.pkl'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None
