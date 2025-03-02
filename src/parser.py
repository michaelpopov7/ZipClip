import cv2
import librosa
import numpy as np
from moviepy.editor import VideoFileClip
import whisper
from pathlib import Path
import os
import sys

def analyze_audio_segment(audio_path):
    print(f"Analyzing audio segment from {audio_path}...")
    y, sr = librosa.load(audio_path)
    energy = librosa.feature.rms(y=y).mean()
    print(f"Extracted audio energy: {energy:.4f}")
    
    print("Transcribing audio with Whisper...")
    try:
        model = whisper.load_model("base")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        sys.exit(1)
    try:
        result = model.transcribe(audio_path)
        text = result["text"]
        word_count = len(text.split())
        print(f"Found {word_count} words in segment")
    except Exception as e:
        print(f"Error during transcription: {e}")
        word_count = 0
    return energy, word_count

def analyze_video_segment(start, end, video_path):
    print(f"Analyzing video segment from {start:.1f}s to {end:.1f}s...")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames = []
    while cap.isOpened() and (cap.get(cv2.CAP_PROP_POS_MSEC) < end * 1000):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"Calculating motion score across {len(frames)} frames...")
    motion_score = 0
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i-1])
        motion_score += np.mean(diff)
    final_score = motion_score / len(frames) if frames else 0
    print(f"Motion score: {final_score:.4f}")
    return final_score

def score_segment(start, end, video_path, temp_audio_path):
    print(f"\nScoring segment {start:.1f}s - {end:.1f}s")
    video = VideoFileClip(video_path).subclip(start, end)
    video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
    energy, word_count = analyze_audio_segment(temp_audio_path)
    motion = analyze_video_segment(start, end, video_path)
    score = (0.4 * energy) + (0.3 * motion) + (0.3 * word_count)
    print(f"Final segment score: {score:.4f}")
    return score

def extract_clips(video_path, min_duration, max_duration, target_clips):
    print(f"\nStarting clip extraction from {video_path}")
    print(f"Parameters: min_duration={min_duration}s, max_duration={max_duration}s, target_clips={target_clips}")
    
    video = VideoFileClip(video_path)
    duration = video.duration
    print(f"Video duration: {duration:.1f} seconds")
    chunk_size = 5  # Analyze in smaller 5-second chunks for granularity
    segments = []
    temp_audio = "assets/extras/temp_audio.wav"

    print("\nScoring all segments...")
    # Score all small segments
    for start in np.arange(0, duration, chunk_size):
        end = min(start + chunk_size, duration)
        score = score_segment(start, end, video_path, temp_audio)
        segments.append((start, end, score))

    # Sort by score
    segments.sort(key=lambda x: x[2], reverse=True)

    print("\nBuilding variable-length clips...")
    clips = []
    used_times = set()  # Track used time ranges to avoid overlap
    score_threshold = np.mean([s[2] for s in segments])  # Dynamic threshold for "interesting"

    for seed_start, seed_end, seed_score in segments:
        if len(clips) >= target_clips:
            break
        # Skip if this seed overlaps with an existing clip
        if any(seed_start < e and seed_end > s for s, e in clips):
            continue

        # Expand from seed
        start = seed_start
        end = seed_end
        current_duration = end - start
        current_score = seed_score

        # Expand forward
        next_idx = segments.index((seed_start, seed_end, seed_score)) + 1
        while (current_duration < max_duration and 
               next_idx < len(segments) and 
               segments[next_idx][0] == end and 
               segments[next_idx][2] >= score_threshold):
            new_end = segments[next_idx][1]
            if any(start < e and new_end > s for s, e in clips):  # Check overlap
                break
            end = new_end
            current_duration = end - start
            current_score = (current_score + segments[next_idx][2]) / 2  # Average score
            next_idx += 1

        # Expand backward
        prev_idx = segments.index((seed_start, seed_end, seed_score)) - 1
        while (current_duration < max_duration and 
               prev_idx >= 0 and 
               segments[prev_idx][1] == start and 
               segments[prev_idx][2] >= score_threshold):
            new_start = segments[prev_idx][0]
            if any(new_start < e and end > s for s, e in clips):  # Check overlap
                break
            start = new_start
            current_duration = end - start
            current_score = (current_score + segments[prev_idx][2]) / 2
            prev_idx -= 1

        # Only add if it meets min_duration
        if min_duration <= current_duration <= max_duration:
            clips.append((start, end))
            print(f"Added clip: {start:.1f}s - {end:.1f}s (duration: {current_duration:.1f}s, score: {current_score:.4f})")

    print("\nExporting final clips...")
    output_clips = []
    os.makedirs("assets/results/clips", exist_ok=True)
    for idx, (start, end) in enumerate(clips):
        print(f"Exporting clip {idx+1}/{len(clips)} ({start:.1f}s - {end:.1f}s)...")
        clip = video.subclip(start, end)
        output_path = f"assets/results/clips/clip_{idx+1}.mp4"
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        output_clips.append(output_path)
    
    print("\nCleaning up temporary files...")
    Path(temp_audio).unlink(missing_ok=True)
    video.close()
    print("Extraction complete!")
    return output_clips

# Example usage
video_path = "assets/videos/ski.mp4"
if not os.path.exists(video_path):
    print(f"Error: video path does not exist: {video_path}")
    sys.exit(1)

min_duration = 15  # seconds
max_duration = 45  # seconds
target_clips = 20
clips = extract_clips(video_path, min_duration, max_duration, target_clips)
print(f"Generated clips: {clips}")