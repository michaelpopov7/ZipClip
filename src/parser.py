import os
import argparse
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from scipy.signal import find_peaks
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoClipExtractor:
    def __init__(self, min_clip_duration=3, max_clip_duration=10, target_clip_count=5, output_dir="extracted_clips"):
        """
        Initialize the VideoClipExtractor.
        
        Args:
            min_clip_duration (int): Minimum duration of clips in seconds
            max_clip_duration (int): Maximum duration of clips in seconds
            target_clip_count (int): Target number of clips to extract
            output_dir (str): Directory to save extracted clips
        """
        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration
        self.target_clip_count = target_clip_count
        self.min_clip_count = max(1, int(target_clip_count * 0.33))  # Ensure at least 33% of target clips
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Initialize AI model for image analysis
        logger.info("Loading AI models...")
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
            self.model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info(f"Models loaded successfully. Using device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading AI models: {e}")
            logger.info("Falling back to OpenCV-based analysis")
            self.feature_extractor = None
            self.model = None
    
    def extract_clips(self, video_path):
        """
        Extract interesting clips from the video.
        
        Args:
            video_path (str): Path to the input video
            
        Returns:
            list: List of extracted clip file paths
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return []
        
        logger.info(f"Analyzing video: {video_path}")
        
        # Load video
        try:
            video = VideoFileClip(video_path)
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            return []
        
        video_duration = video.duration
        logger.info(f"Video duration: {video_duration:.2f} seconds")
        
        # Calculate interestingness scores throughout the video
        interest_scores = self._calculate_interest_scores(video)
        
        # Find peaks in interest scores with adaptive threshold to ensure minimum clips
        saved_clips = []
        min_distance = int(self.min_clip_duration * video.fps)
        prominence_threshold = 0.3  # Starting threshold for peak prominence
        
        while prominence_threshold > 0.01:  # Lower limit for threshold
            peaks, _ = find_peaks(interest_scores, distance=min_distance, prominence=prominence_threshold)
            
            # If we have enough peaks, break the loop
            if len(peaks) >= self.min_clip_count:
                logger.info(f"Found {len(peaks)} interesting moments with prominence threshold {prominence_threshold:.3f}")
                break
                
            # Otherwise lower the threshold and try again
            prominence_threshold *= 0.7  # Reduce threshold by 30%
            logger.info(f"Reducing prominence threshold to {prominence_threshold:.3f} to find more clips")
        
        # If we still don't have enough peaks, use evenly spaced segments
        if len(peaks) < self.min_clip_count:
            logger.warning(f"Could not find {self.min_clip_count} interesting clips even with lowest threshold")
            logger.info("Falling back to evenly spaced segments")
            
            # Calculate how many evenly spaced clips we need
            remaining_clips = self.min_clip_count - len(peaks)
            segment_duration = video_duration / (remaining_clips + 1)
            
            # Add evenly spaced frame indices
            even_peaks = [int((i + 1) * segment_duration * video.fps) for i in range(remaining_clips)]
            peaks = np.sort(np.concatenate([peaks, even_peaks]))
            logger.info(f"Added {remaining_clips} evenly spaced segments to reach minimum clip count")
        
        # Convert peak frames to timestamps
        peak_times = [peak / video.fps for peak in peaks]
        
        # Sort peaks by interest score value (for the detected peaks)
        peak_scores = [(time, interest_scores[int(min(time * video.fps, len(interest_scores)-1))]) for time in peak_times]
        peak_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to target clip count if needed
        clip_count = min(self.target_clip_count, len(peak_scores))
        logger.info(f"Extracting {clip_count} clips (minimum required: {self.min_clip_count})")
        
        # Extract and save clips
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        for i, (peak_time, score) in enumerate(peak_scores[:clip_count]):
            # Calculate clip start and end times
            half_duration = min(self.max_clip_duration / 2, 
                               (peak_time if peak_time < video_duration / 2 else video_duration - peak_time))
            
            # Ensure clip duration is at least min_clip_duration
            if half_duration * 2 < self.min_clip_duration:
                half_duration = self.min_clip_duration / 2
            
            start_time = max(0, peak_time - half_duration)
            end_time = min(video_duration, peak_time + half_duration)
            
            # Ensure minimum clip duration
            if end_time - start_time < self.min_clip_duration:
                end_time = min(video_duration, start_time + self.min_clip_duration)
            
            # Check for overlap with previously extracted clips
            overlap = False
            for prev_start, prev_end in [(c['start'], c['end']) for c in saved_clips]:
                if (start_time < prev_end and end_time > prev_start):
                    overlap = True
                    break
            
            if overlap:
                logger.info(f"Skipping clip {i+1} due to overlap with previously extracted clip")
                continue
            
            # Extract and save clip
            clip_path = os.path.join(self.output_dir, f"{video_name}_clip_{i+1}.mp4")
            try:
                clip = video.subclip(start_time, end_time)
                clip.write_videofile(clip_path, codec="libx264", audio_codec="aac", 
                                    temp_audiofile="temp-audio.m4a", remove_temp=True,
                                    logger=None)
                logger.info(f"Saved clip {i+1}/{clip_count} to {clip_path}")
                saved_clips.append({
                    'path': clip_path,
                    'start': start_time,
                    'end': end_time,
                    'score': score
                })
            except Exception as e:
                logger.error(f"Error saving clip {i+1}: {e}")
        
        # If we still don't have enough clips (due to overlaps or errors), add more from remaining peaks
        remaining_peaks = peak_scores[clip_count:]
        additional_index = 0
        
        while len(saved_clips) < self.min_clip_count and additional_index < len(remaining_peaks):
            peak_time, score = remaining_peaks[additional_index]
            
            # Calculate clip duration
            half_duration = min(self.max_clip_duration / 2, 
                              (peak_time if peak_time < video_duration / 2 else video_duration - peak_time))
            
            start_time = max(0, peak_time - half_duration)
            end_time = min(video_duration, peak_time + half_duration)
            
            # Check for overlap
            overlap = False
            for prev_start, prev_end in [(c['start'], c['end']) for c in saved_clips]:
                if (start_time < prev_end and end_time > prev_start):
                    overlap = True
                    break
            
            if not overlap:
                clip_path = os.path.join(self.output_dir, f"{video_name}_clip_{len(saved_clips)+1}.mp4")
                try:
                    clip = video.subclip(start_time, end_time)
                    clip.write_videofile(clip_path, codec="libx264", audio_codec="aac", 
                                        temp_audiofile="temp-audio.m4a", remove_temp=True,
                                        logger=None)
                    logger.info(f"Saved additional clip {len(saved_clips)+1} to {clip_path}")
                    saved_clips.append({
                        'path': clip_path,
                        'start': start_time,
                        'end': end_time,
                        'score': score
                    })
                except Exception as e:
                    logger.error(f"Error saving additional clip: {e}")
            
            additional_index += 1
        
        # If we still don't have enough clips, create evenly spaced clips as last resort
        if len(saved_clips) < self.min_clip_count:
            logger.warning(f"Could not meet minimum clip count with detected peaks. Creating evenly spaced clips.")
            
            for i in range(self.min_clip_count - len(saved_clips)):
                # Calculate an evenly spaced position
                position = (i + 1) * video_duration / (self.min_clip_count - len(saved_clips) + 1)
                
                # Adjusted for clip duration
                start_time = max(0, position - self.min_clip_duration / 2)
                end_time = min(video_duration, position + self.min_clip_duration / 2)
                
                # Check for overlap
                overlap = False
                for prev_start, prev_end in [(c['start'], c['end']) for c in saved_clips]:
                    if (start_time < prev_end and end_time > prev_start):
                        overlap = True
                        break
                
                if overlap:
                    continue
                
                clip_path = os.path.join(self.output_dir, f"{video_name}_clip_fallback_{i+1}.mp4")
                try:
                    clip = video.subclip(start_time, end_time)
                    clip.write_videofile(clip_path, codec="libx264", audio_codec="aac", 
                                        temp_audiofile="temp-audio.m4a", remove_temp=True,
                                        logger=None)
                    logger.info(f"Saved fallback clip {i+1} to {clip_path}")
                    saved_clips.append({
                        'path': clip_path,
                        'start': start_time,
                        'end': end_time,
                        'score': 0.0
                    })
                except Exception as e:
                    logger.error(f"Error saving fallback clip: {e}")
        
        video.close()
        return [clip['path'] for clip in saved_clips]
    
    def _calculate_interest_scores(self, video):
        """
        Calculate interestingness scores for each frame in the video.
        
        Args:
            video (VideoFileClip): The loaded video
            
        Returns:
            numpy.ndarray: Array of interestingness scores
        """
        # Sample frames at regular intervals
        frame_count = int(video.duration * video.fps)
        sample_rate = max(1, int(video.fps / 2))  # Sample at half the frame rate
        
        logger.info(f"Analyzing approximately {frame_count // sample_rate} frames...")
        
        # Make array one element larger to handle potential rounding issues at the end
        interest_scores = np.zeros(frame_count + 1)
        prev_frame = None
        
        # Process video frames
        for i, frame_time in enumerate(np.arange(0, video.duration, 1/video.fps)):
            if i % sample_rate != 0 and i < frame_count - 1:
                continue
                
            # Safety check to prevent index out of bounds
            if i >= len(interest_scores):
                logger.warning(f"Frame index {i} exceeds allocated array size {len(interest_scores)}. Skipping.")
                continue
                
            try:
                frame = video.get_frame(frame_time)
                
                # Calculate various interest metrics
                if self.model is not None and self.feature_extractor is not None:
                    # Use AI model for more advanced analysis
                    try:
                        # Convert frame to RGB and resize
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        inputs = self.feature_extractor(images=frame_rgb, return_tensors="pt").to(self.device)
                        
                        # Get model outputs
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                        
                        # Use logits variance as a measure of "interestingness"
                        # Higher variance suggests the model is more confident about the content
                        logits_variance = outputs.logits.var().item()
                        
                        interest_scores[i] = logits_variance
                    except Exception as e:
                        logger.warning(f"Error in AI analysis for frame at {frame_time:.2f}s: {e}")
                        # Fall back to OpenCV analysis
                        interest_scores[i] = self._calculate_opencv_interest(frame, prev_frame)
                else:
                    # Use OpenCV-based analysis
                    interest_scores[i] = self._calculate_opencv_interest(frame, prev_frame)
                
                prev_frame = frame
            except Exception as e:
                logger.warning(f"Error processing frame at {frame_time:.2f}s: {e}")
                # Use previous score or zero if no previous score
                if i > 0:
                    interest_scores[i] = interest_scores[i-1]
                
            # Show progress periodically
            if i % (10 * sample_rate) == 0:
                logger.info(f"Analysis progress: {i/frame_count*100:.1f}%")
        
        # Trim array to actual used size
        interest_scores = interest_scores[:frame_count]
        
        # Normalize scores
        if interest_scores.max() > interest_scores.min():
            interest_scores = (interest_scores - interest_scores.min()) / (interest_scores.max() - interest_scores.min())
        
        # Apply smoothing to reduce noise
        window_size = int(video.fps * 1.5)  # 1.5 second window
        kernel = np.ones(window_size) / window_size
        smoothed_scores = np.convolve(interest_scores, kernel, mode='same')
        
        return smoothed_scores

    def _calculate_opencv_interest(self, frame, prev_frame):
        """
        Calculate interesting metrics using OpenCV.
        
        Args:
            frame: Current video frame
            prev_frame: Previous video frame
            
        Returns:
            float: Interest score for the frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics
        score = 0
        
        # 1. Measure visual complexity (using Laplacian variance)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        score += lap_var * 0.01  # Scale factor
        
        # 2. Check for motion if we have a previous frame
        if prev_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate motion magnitude
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion = np.mean(mag)
            score += motion * 5.0  # Scale factor
        
        # 3. Face detection for human interest
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        score += len(faces) * 50.0  # High interest for frames with faces
        
        # 4. Color diversity as an interest factor
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_std = np.std(hsv[:,:,0])  # Hue standard deviation
        score += color_std * 0.5  # Scale factor
        
        return score

def main():
    parser = argparse.ArgumentParser(description="Extract interesting clips from a video")
    parser.add_argument("--video_path", type=str, help="Path to the input video file")
    parser.add_argument("--min_duration", type=float, default=3.0, help="Minimum clip duration in seconds")
    parser.add_argument("--max_duration", type=float, default=45.0, help="Maximum clip duration in seconds")
    parser.add_argument("--target_clips", type=int, default=20, help="Target number of clips to extract")
    parser.add_argument("--output_dir", type=str, default="extracted_clips", help="Directory to save extracted clips")
    
    args = parser.parse_args()
    
    if args.video_path is None:
        # Prompt for video path if not provided
        args.video_path = input("Enter the path to the video file: ")
    
    extractor = VideoClipExtractor(
        min_clip_duration=args.min_duration,
        max_clip_duration=args.max_duration,
        target_clip_count=args.target_clips,
        output_dir=args.output_dir
    )
    
    saved_clips = extractor.extract_clips(args.video_path)
    
    if saved_clips:
        logger.info(f"Successfully extracted {len(saved_clips)} clips:")
        for clip in saved_clips:
            logger.info(f" - {clip}")
        logger.info(f"Clips saved to directory: {args.output_dir}")
    else:
        logger.warning("No clips were extracted from the video.")

if __name__ == "__main__":
    main()