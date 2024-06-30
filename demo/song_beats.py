# %%
import cv2
import librosa
import numpy as np
from moviepy.editor import AudioFileClip, VideoFileClip


# Load audio file
AUDIO_FILE = "/Users/matbreotten/Downloads/song01.wav"
OUTPUT_VIDEO = "/Users/matbreotten/Downloads/output_video.mp4"
y, sr = librosa.load(AUDIO_FILE)

# Calculate tempo (beats per minute)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beats, sr=sr)

# Set video parameters
width, height = 640, 480  # Adjust as needed
fps = 30
seconds = len(y) / sr
total_frames = int(seconds * fps)
color_palette = [
    (255, 0, 0),  # Red
    (220, 20, 60),  # Crimson
    (178, 34, 34),  # Firebrick
    (205, 92, 92),  # Indian Red
    (240, 128, 128),  # Light Coral
    (233, 150, 122),  # Dark Salmon
    (250, 128, 114),  # Salmon
    (255, 99, 71),  # Tomato
    (255, 69, 0),  # Orange Red
    (255, 0, 0),  # Red
]

# Create VideoWriter object
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Track active droplets
active_droplets = []

# Track index of beat_times
beat_idx = 0

# Iterate through each frame
for frame_idx in range(total_frames):
    frame = np.zeros((height, width, 3), dtype=np.uint8)  # Black background frame

    # Check if we have a beat within the current frame
    if beat_idx < len(beat_times) and librosa.time_to_frames(beat_times[beat_idx], sr=sr) == frame_idx:
        # Add new droplet based on beat event
        droplet_x = int(np.random.rand() * width)
        droplet_y = int(np.random.rand() * height)
        color_idx = np.random.randint(len(color_palette))  # Select a random index
        droplet_color = color_palette[color_idx]
        droplet_radius = np.random.randint(10, 30)  # Initial radius for droplet
        active_droplets.append({"x": droplet_x, "y": droplet_y, "color": droplet_color, "radius": droplet_radius})

        # Move to the next beat
        beat_idx += 1

    # Update and draw active droplets
    for droplet in active_droplets:
        droplet["radius"] -= 1  # Decrease radius each frame

        if droplet["radius"] > 0:
            cv2.circle(frame, (droplet["x"], droplet["y"]), droplet["radius"], droplet["color"], -1)
        else:
            active_droplets.remove(droplet)  # Remove droplet if radius is zero or negative

    # Write frame to video
    out.write(frame)

# Release VideoWriter
out.release()

# Combine video and audio using moviepy
video_clip = VideoFileClip(OUTPUT_VIDEO)
audio_clip = AudioFileClip(AUDIO_FILE)
video_clip = video_clip.set_audio(audio_clip)
video_clip.write_videofile(OUTPUT_VIDEO.replace(".mp4", "_with_audio.mp4"), codec="libx264", audio_codec="aac")

# Close all video clips
video_clip.close()
audio_clip.close()

# %%
