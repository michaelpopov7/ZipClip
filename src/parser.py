import cv2
import librosa
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
import whisper
from pathlib import Path

def analyze_audio_segment(audio_path):
    # Load audio and extract features
    y, sr = librosa.load(audio_path)
    energy = librosa.feature.rms(y=y).mean()  # Audio energy (louder = more exciting?)
    # Use Whisper to transcribe
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    text = result["text"]
    # Simple heuristic: more words might indicate dialogue/interest
    word_count = len(text.split())
    print(energy, word_count)
    
    return energy, word_count

def analyze_video_segment(start, end, video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames = []
    while cap.isOpened() and (cap.get(cv2.CAP_PROP_POS_MSEC) < end * 1000):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # Motion detection (simple differencing)
    motion_score = 0
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i-1])
        motion_score += np.mean(diff)
    return motion_score / len(frames) if frames else 0

def score_segment(start, end, video_path, temp_audio_path):
    # Extract audio for this segment
    video = VideoFileClip(video_path).subclip(start, end)
    video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
    
    # Analyze audio
    energy, word_count = analyze_audio_segment(temp_audio_path)
    # Analyze video
    motion = analyze_video_segment(start, end, video_path)
    
    # Combine into an "interestingness" score (weights are arbitrary, tune them!)
    score = (0.4 * energy) + (0.3 * motion) + (0.3 * word_count)
    return score

def extract_clips(video_path, min_duration, max_duration, target_clips):
    video = VideoFileClip(video_path)
    duration = video.duration
    chunk_size = 5  # Analyze in 5-second chunks
    segments = []

    # Temporary audio file for analysis
    temp_audio = "temp_audio.wav"

    # Score all segments
    for start in np.arange(0, duration, chunk_size):
        end = min(start + chunk_size, duration)
        score = score_segment(start, end, video_path, temp_audio)
        segments.append((start, end, score))

    # Sort by score and pick top segments
    segments.sort(key=lambda x: x[2], reverse=True)
    top_segments = segments[:target_clips * 2]  # Overshoot to allow merging

    # Merge adjacent segments to meet duration constraints
    clips = []
    i = 0
    while i < len(top_segments) and len(clips) < target_clips:
        start, end, _ = top_segments[i]
        while (end - start < min_duration) and (i + 1 < len(top_segments)):
            i += 1
            end = top_segments[i][1]  # Extend to next segment
        if min_duration <= (end - start) <= max_duration:
            clips.append((start, end))
        i += 1

    # Export clips
    output_clips = []
    for idx, (start, end) in enumerate(clips):
        clip = video.subclip(start, end)
        output_path = f"clip_{idx+1}.mp4"
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        output_clips.append(output_path)
    
    # Clean up
    Path(temp_audio).unlink(missing_ok=True)
    video.close()
    return output_clips

# Example usage
video_path = "test.mp4"
min_duration = 5  # seconds
max_duration = 30  # seconds
target_clips = 3
clips = extract_clips(video_path, min_duration, max_duration, target_clips)
print(f"Generated clips: {clips}")
