import pandas as pd
from moviepy import ImageSequenceClip
import os

def create_video_from_csv(csv_file, fps=5):
    # 1. Load the CSV
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)

    # 2. Get available sequences from the ./data/ folder
    data_dir = './data/'
    if not os.path.exists(data_dir):
        print(f"❌ Error: {data_dir} directory not found.")
        return

    # Scan for folders only
    sequences = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not sequences:
        print("❌ No sequence folders found in ./data/")
        return

    # 3. Ask user which sequence to process
    print("\n--- Available Sequences ---")
    for idx, seq in enumerate(sequences):
        print(f"[{idx}] {seq}")
    
    try:
        choice = int(input("\nEnter the number of the sequence you want to process: "))
        selected_seq = sequences[choice]
    except (ValueError, IndexError):
        print("❌ Invalid selection. Exiting.")
        return

    # 4. Filter the CSV for the selected sequence
    # This matches the folder name in your path: ./data/sequenceid/
    seq_path_part = f"{selected_seq}"
    filtered_df = df[df['file_path'].str.contains(seq_path_part, case=False, na=False)]
    
    # Sort by index to maintain chronological walking order
    filtered_df = filtered_df.sort_values('index')
    image_files = filtered_df['file_path'].tolist()

    # 5. Filter to ensure files actually exist on disk
    valid_images = [img for img in image_files if os.path.exists(img)]
    
    if not valid_images:
        print(f"❌ No valid images found for sequence: {selected_seq}")
        return

    # 6. Create and compress the video
    output_name = f"{selected_seq}.mp4"
    print(f"🎬 Creating {output_name} ({len(valid_images)} frames)...")

    try:
        # Load images
        clip = ImageSequenceClip(valid_images, fps=fps)

        # COMPRESSION SETTINGS:
        # Resize to 720p height (keeps aspect ratio) to slash file size
        if clip.h > 720:
            clip = clip.resize(height=720)

        clip.write_videofile(
            output_name, 
            codec='libx264', 
            audio=False, 
            ffmpeg_params=[
                "-crf", "28",           # Higher number = more compression (23-28 is sweet spot)
                "-preset", "veryslow",  # Takes longer to encode, but makes file smaller
                "-pix_fmt", "yuv420p"   # Ensures maximum compatibility
            ]
        )
        print(f"\n✅ Successfully created {output_name}")
        print(f"📍 Location: {os.path.abspath(output_name)}")
        
    except Exception as e:
        print(f"❌ Error during video creation: {e}")

if __name__ == "__main__":
    # You can change fps here (e.g., fps=1 for slow, fps=10 for fast timelapse)
    create_video_from_csv(csv_file='image_coordinates.csv', fps=1)