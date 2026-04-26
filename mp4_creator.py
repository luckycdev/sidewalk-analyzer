import pandas as pd
from moviepy import ImageSequenceClip, vfx  # Added vfx here
import os

def create_video_from_csv(fps=1):
    csv_dir = './csvs/'
    
    if not os.path.exists(csv_dir):
        print(f"❌ Error: {csv_dir} directory not found. Run extract_metadata.py first!")
        return

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ No CSV files found in {csv_dir}")
        return

    print("\n--- Available CSV Sequences ---")
    for idx, filename in enumerate(csv_files):
        print(f"[{idx}] {filename.replace('.csv', '')}")
    
    try:
        choice = int(input("\nEnter the number of the sequence you want to turn into video: "))
        selected_csv_name = csv_files[choice]
        sequence_id = selected_csv_name.replace('.csv', '')
    except (ValueError, IndexError):
        print("❌ Invalid selection. Exiting.")
        return

    csv_path = os.path.join(csv_dir, selected_csv_name)
    print(f"📖 Reading {selected_csv_name}...")
    df = pd.read_csv(csv_path)

    df = df.sort_values('index')
    image_files = df['file_path'].tolist()

    valid_images = [img for img in image_files if os.path.exists(img)]

    if not valid_images:
        print(f"❌ No valid images found on disk for this sequence.")
        return

    output_name = f"{sequence_id}.mp4"
    print(f"🎬 Creating {output_name} ({len(valid_images)} frames)...")

    try:
        clip = ImageSequenceClip(valid_images, fps=fps)

        # UPDATED RESIZE LOGIC FOR MOVIEPY 2.0+
        if clip.h > 720:
            print("📉 Resizing to 720p...")
            clip = clip.with_effects([vfx.Resize(height=720)]) # New syntax

        clip.write_videofile(
            output_name, 
            codec='libx264', 
            audio=False, 
            ffmpeg_params=[
                "-crf", "28",
                "-preset", "veryslow",
                "-pix_fmt", "yuv420p"
            ]
        )
        print(f"\n✅ Successfully created {output_name}")
        
    except Exception as e:
        print(f"❌ Error during video creation: {e}")

if __name__ == "__main__":
    create_video_from_csv(fps=1)