import pandas as pd
from moviepy import ImageSequenceClip
import os

def create_video_from_csv(csv_file, output_mp4, fps=1):
    # 1. Load the CSV created in the previous step
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    
    # 2. Extract the file paths in the order of the 'index'
    # We sort just in case the CSV isn't ordered by index already
    df = df.sort_values('index')
    image_files = df['file_path'].tolist()

    # 3. Filter to ensure files actually exist on disk
    valid_images = [img for img in image_files if os.path.exists(img)]
    
    if not valid_images:
        print("No valid image files found from the CSV paths.")
        return

    print(f"Creating video with {len(valid_images)} frames...")

    # 4. Create the clip
    # fps=1 means 1 frame per second (each image stays for 1s)
    clip = ImageSequenceClip(valid_images, fps=fps)

    # 5. Write the file
    # codec='libx264' is standard for high compatibility MP4s
    clip.write_videofile(output_mp4, codec='libx264')

if __name__ == "__main__":
    create_video_from_csv(
        csv_file='image_coordinates.csv', 
        output_mp4='path_timelapse.mp4'
    )