import argparse
import os
import subprocess
import sys
import logging
import time
import random
import json
import numpy as np
import cv2
from datetime import datetime
from moviepy.editor import VideoFileClip
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from scipy.signal import find_peaks
import requests
from urllib.parse import quote

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("youtube_clip_hunter.log"),
        logging.StreamHandler()
    ]
)
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
        
        # Create array with exact size needed for sampled frames
        sampled_frame_count = (frame_count + sample_rate - 1) // sample_rate
        interest_scores = np.zeros(sampled_frame_count)
        prev_frame = None
        
        # Process video frames
        for i, frame_time in enumerate(np.arange(0, video.duration, 1/video.fps)):
            if i % sample_rate != 0:
                continue
                
            # Calculate the index in our interest_scores array
            score_index = i // sample_rate
            
            # Skip if we've somehow gone out of bounds
            if score_index >= len(interest_scores):
                logger.warning(f"Skipping frame at {frame_time:.2f}s due to index bounds ({score_index} >= {len(interest_scores)})")
                continue
                
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
                    
                    interest_scores[score_index] = logits_variance
                except Exception as e:
                    logger.warning(f"Error in AI analysis for frame at {frame_time:.2f}s: {e}")
                    # Fall back to OpenCV analysis
                    interest_scores[score_index] = self._calculate_opencv_interest(frame, prev_frame)
            else:
                # Use OpenCV-based analysis
                interest_scores[score_index] = self._calculate_opencv_interest(frame, prev_frame)
            
            prev_frame = frame
            
            # Show progress periodically
            if i % (10 * sample_rate) == 0:
                logger.info(f"Analysis progress: {i/frame_count*100:.1f}%")
        
        # Normalize scores
        if interest_scores.max() > interest_scores.min():
            interest_scores = (interest_scores - interest_scores.min()) / (interest_scores.max() - interest_scores.min())
        
        # Apply smoothing to reduce noise
        window_size = max(1, int(video.fps * 1.5) // sample_rate)  # 1.5 second window
        kernel = np.ones(window_size) / window_size
        smoothed_scores = np.convolve(interest_scores, kernel, mode='same')
        
        # Resample back to original frame count for compatibility with the rest of the code
        # Using linear interpolation to map from sampled frames to all frames
        x_sampled = np.linspace(0, 1, len(smoothed_scores))
        x_full = np.linspace(0, 1, frame_count)
        final_scores = np.interp(x_full, x_sampled, smoothed_scores)
        
        return final_scores
    
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
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Calculate motion magnitude
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion = np.mean(mag)
                score += motion * 5.0  # Scale factor
            except Exception as e:
                logger.warning(f"Error calculating optical flow: {e}")
        
        # 3. Face detection for human interest
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            score += len(faces) * 50.0  # High interest for frames with faces
        except Exception as e:
            logger.warning(f"Error in face detection: {e}")
        
        # 4. Color diversity as an interest factor
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_std = np.std(hsv[:,:,0])  # Hue standard deviation
        score += color_std * 0.5  # Scale factor
        
        return score

class YouTubeScavenger:
    def __init__(self, download_dir="downloads", min_duration=3600, api_key=None):
        """
        Initialize the YouTube video scavenger.
        
        Args:
            download_dir (str): Directory to save downloaded videos
            min_duration (int): Minimum video duration in seconds (default: 1 hour)
            api_key (str): YouTube Data API key (optional)
        """
        self.download_dir = download_dir
        self.min_duration = min_duration  # At least 1 hour
        self.api_key = api_key
        self.download_history_file = "download_history.json"
        self.download_history = self._load_download_history()
        
        # Create download directory if it doesn't exist
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
        
        # Check if yt-dlp is installed
        self._check_dependencies()
    
    def _load_download_history(self):
        """Load download history to avoid downloading the same video twice."""
        if os.path.exists(self.download_history_file):
            try:
                with open(self.download_history_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Error loading download history, starting with empty history.")
                return {"videos": []}
        return {"videos": []}
    
    def _save_download_history(self):
        """Save download history to file."""
        with open(self.download_history_file, 'w') as f:
            json.dump(self.download_history, f)
    
    def _check_dependencies(self):
        """Check if yt-dlp is installed, if not, try to install it."""
        try:
            subprocess.run(["yt-dlp", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            logger.info("yt-dlp is already installed.")
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("yt-dlp not found. Attempting to install...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"], check=True)
                logger.info("yt-dlp successfully installed.")
                return True
            except subprocess.SubprocessError:
                logger.error("Failed to install yt-dlp. Please install it manually: pip install yt-dlp")
                return False
    
    def search_videos(self, query, max_results=10):
        """
        Search for videos on YouTube using either API or yt-dlp.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of search results to return
            
        Returns:
            list: List of video IDs and titles
        """
        if self.api_key:
            return self._search_with_api(query, max_results)
        else:
            return self._search_with_ytdlp(query, max_results)
    
    def _search_with_api(self, query, max_results=10):
        """Search videos using YouTube Data API."""
        endpoint = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={quote(query)}&type=video&maxResults={max_results}&key={self.api_key}"
        try:
            response = requests.get(endpoint)
            if response.status_code == 200:
                data = response.json()
                videos = []
                for item in data.get('items', []):
                    video_id = item['id']['videoId']
                    title = item['snippet']['title']
                    videos.append({
                        'id': video_id,
                        'title': title,
                        'url': f"https://www.youtube.com/watch?v={video_id}"
                    })
                return videos
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error searching with API: {e}")
            return []
    
    def _search_with_ytdlp(self, query, max_results=10):
        """Search videos using yt-dlp when no API key is available."""
        try:
            cmd = [
                "yt-dlp", 
                f"ytsearch{max_results}:{query}", 
                "--flat-playlist", 
                "--print", "id,title",
                "--no-download"
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = process.stdout.strip().split('\n')
            
            videos = []
            for line in lines:
                if line and '\t' in line:
                    video_id, title = line.split('\t', 1)
                    videos.append({
                        'id': video_id,
                        'title': title,
                        'url': f"https://www.youtube.com/watch?v={video_id}"
                    })
            
            return videos
        except subprocess.CalledProcessError as e:
            logger.error(f"Error searching with yt-dlp: {str(e)}")
            if e.stderr:
                logger.error(f"yt-dlp error: {e.stderr}")
            return []
    
    def filter_long_videos(self, videos):
        """
        Filter videos to include only those longer than min_duration.
        
        Args:
            videos (list): List of video dictionaries with id, title, and url
            
        Returns:
            list: Filtered list of videos
        """
        long_videos = []
        
        for video in videos:
            # Skip videos we've already downloaded
            if video['id'] in [v['id'] for v in self.download_history['videos']]:
                logger.info(f"Skipping already downloaded video: {video['title']}")
                continue
            
            try:
                cmd = [
                    "yt-dlp",
                    "--skip-download",
                    "--print", "duration",
                    video['url']
                ]
                
                process = subprocess.run(cmd, capture_output=True, text=True, check=True)
                duration_str = process.stdout.strip()
                
                try:
                    duration = int(duration_str)
                    video['duration'] = duration
                    
                    if duration >= self.min_duration:
                        logger.info(f"Found long video ({duration}s): {video['title']}")
                        long_videos.append(video)
                    else:
                        logger.info(f"Skipping short video ({duration}s): {video['title']}")
                except ValueError:
                    logger.warning(f"Could not parse duration for video: {video['title']}")
            
            except subprocess.CalledProcessError as e:
                logger.error(f"Error getting duration for {video['title']}: {str(e)}")
        
        return long_videos
    
    def download_video(self, video, resolution="best"):
        """
        Download a YouTube video.
        
        Args:
            video (dict): Video dictionary with url, id, and title
            resolution (str): Video quality option
            
        Returns:
            str: Path to downloaded video file if successful, None otherwise
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
        
        # Format resolution for yt-dlp
        if resolution not in ["best", "worst"]:
            # If resolution is a number like 720, format it as 'bestvideo[height<=720]+bestaudio/best[height<=720]'
            try:
                height = int(resolution.replace("p", ""))
                format_option = f"bestvideo[height<={height}]+bestaudio/best[height<={height}]"
            except ValueError:
                logger.warning(f"Invalid resolution: {resolution}. Using best quality.")
                format_option = "best"
        else:
            format_option = resolution
        
        # Sanitize the output filename
        safe_title = ''.join(c if c.isalnum() or c in ' .-_' else '_' for c in video['title'])
        output_template = os.path.join(self.download_dir, f"{video['id']}_{safe_title[:50]}.%(ext)s")
        
        # Build the command
        cmd = [
            "yt-dlp",
            video['url'],
            "-f", format_option,
            "-o", output_template,
            "--no-playlist",
            "--progress"
        ]
        
        # Run the download command
        try:
            logger.info(f"Starting download: {video['title']}")
            logger.info(f"URL: {video['url']}")
            
            process = subprocess.run(cmd, text=True, capture_output=True)
            
            if process.returncode != 0:
                logger.error(f"Download failed: {process.stderr}")
                return None
            
            # Extract the output filename from the process output
            output_file = None
            for line in process.stdout.split('\n'):
                if "[download]" in line and "Destination:" in line:
                    output_file = line.split("Destination: ", 1)[1].strip()
                    break
                elif "[download]" in line and "has already been downloaded" in line:
                    output_file = line.split("[download] ", 1)[1].split(" has already", 1)[0].strip()
                    break
            
            if output_file and os.path.exists(output_file):
                logger.info(f"Download successful: {output_file}")
                
                # Add to download history
                self.download_history['videos'].append({
                    'id': video['id'],
                    'title': video['title'],
                    'url': video['url'],
                    'file_path': output_file,
                    'downloaded_at': datetime.now().isoformat()
                })
                self._save_download_history()
                
                return output_file
            else:
                logger.warning("Download may have failed or file path not found in output")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Download process error: {e}")
            if e.stderr:
                logger.error(f"Error details: {e.stderr}")
            return None

class YouTubeClipHunter:
    def __init__(self, topics, videos_per_topic=2, clip_extractor_args=None, scavenger_args=None):
        """
        Initialize the YouTube Clip Hunter.
        
        Args:
            topics (list): List of topics to search for
            videos_per_topic (int): Number of videos to find per topic
            clip_extractor_args (dict): Arguments for VideoClipExtractor
            scavenger_args (dict): Arguments for YouTubeScavenger
        """
        self.topics = topics
        self.videos_per_topic = videos_per_topic
        
        # Default arguments for VideoClipExtractor
        extractor_defaults = {
            'min_clip_duration': 5,
            'max_clip_duration': 15,
            'target_clip_count': 3,
            'output_dir': "extracted_clips"
        }
        
        # Default arguments for YouTubeScavenger
        scavenger_defaults = {
            'download_dir': "downloads",
            'min_duration': 3600,  # 1 hour minimum
            'api_key': None
        }
        
        # Update defaults with provided arguments
        if clip_extractor_args:
            extractor_defaults.update(clip_extractor_args)
        
        if scavenger_args:
            scavenger_defaults.update(scavenger_args)
        
        # Initialize the components
        self.extractor = VideoClipExtractor(**extractor_defaults)
        self.scavenger = YouTubeScavenger(**scavenger_defaults)
        
        # State tracking
        self.state_file = "clip_hunter_state.json"
        self.state = self._load_state()
    
    def _load_state(self):
        """Load the current state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Error loading state, starting with empty state.")
        
        # Initial state
        return {
            'topics_processed': {},
            'videos_downloaded': [],
            'clips_extracted': [],
            'last_run': None
        }
    
    def _save_state(self):
        """Save the current state to file."""
        self.state['last_run'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def run(self, continuous=False, sleep_between_runs=3600):
        """
        Run the clip hunter process.
        
        Args:
            continuous (bool): Whether to run continuously
            sleep_between_runs (int): Seconds to sleep between runs in continuous mode
        """
        try:
            while True:
                logger.info(f"Starting YouTube Clip Hunter run at {datetime.now().isoformat()}")
                
                # Process each topic
                for topic in self.topics:
                    # Check if we've already processed this topic recently
                    if topic in self.state['topics_processed']:
                        last_processed = datetime.fromisoformat(self.state['topics_processed'][topic])
                        time_diff = (datetime.now() - last_processed).total_seconds()
                        
                        # Skip if processed in the last 24 hours
                        if time_diff < 86400:  # 24 hours in seconds
                            logger.info(f"Skipping topic '{topic}' - processed recently ({time_diff/3600:.1f} hours ago)")
                            continue
                    
                    logger.info(f"Processing topic: {topic}")
                    
                    # Search for videos on the topic
                    videos = self.scavenger.search_videos(topic, max_results=15)
                    if not videos:
                        logger.warning(f"No videos found for topic: {topic}")
                        continue
                    
                    # Filter for long videos
                    long_videos = self.scavenger.filter_long_videos(videos)
                    if not long_videos:
                        logger.warning(f"No long videos found for topic: {topic}")
                        continue
                    
                    logger.info(f"Found {len(long_videos)} long videos for topic '{topic}'")
                    
                    # Sort by duration (optional)
                    long_videos.sort(key=lambda x: x.get('duration', 0), reverse=True)
                    
                    # Limit to required videos per topic
                    selected_videos = long_videos[:self.videos_per_topic]
                    
                    # Download videos
                    for video in selected_videos:
                        # Check if we've already downloaded this video
                        if video['id'] in [v['id'] for v in self.state['videos_downloaded']]:
                            logger.info(f"Skipping already downloaded video: {video['title']}")
                            continue
                        
                        # Download the video
                        video_path = self.scavenger.download_video(video)
                        if not video_path:
                            logger.warning(f"Failed to download video: {video['title']}")
                            continue
                        
                        # Add to our state
                        video_info = {
                            'id': video['id'],
                            'title': video['title'],
                            'url': video['url'],
                            'topic': topic,
                            'file_path': video_path,
                            'downloaded_at': datetime.now().isoformat(),
                            'clips_extracted': False
                        }
                        self.state['videos_downloaded'].append(video_info)
                        self._save_state()
                    
                    # Mark topic as processed
                    self.state['topics_processed'][topic] = datetime.now().isoformat()
                    self._save_state()
                
                # Extract clips from videos that haven't been processed yet
                self._extract_clips_from_pending_videos()
                
                if not continuous:
                    break
                
                logger.info(f"Sleeping for {sleep_between_runs} seconds before next run...")
                time.sleep(sleep_between_runs)
        
        except KeyboardInterrupt:
            logger.info("Process interrupted by user. Exiting...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def _extract_clips_from_pending_videos(self):
        """Extract clips from videos that haven't been processed yet."""
        # Find videos that need clips extracted
        pending_videos = [v for v in self.state['videos_downloaded'] if not v.get('clips_extracted', False)]
        
        if not pending_videos:
            logger.info("No pending videos requiring clip extraction")
            return
        
        logger.info(f"Found {len(pending_videos)} videos waiting for clip extraction")
        
        for video in pending_videos:
            logger.info(f"Extracting clips from: {video['title']}")
            
            # Check if the video file still exists
            if not os.path.exists(video['file_path']):
                logger.warning(f"Video file not found: {video['file_path']}")
                continue
            
            # Extract clips
            clip_paths = self.extractor.extract_clips(video['file_path'])
            
            if not clip_paths:
                logger.warning(f"No clips extracted from video: {video['title']}")
                # Still mark as processed to avoid repeated attempts
                video['clips_extracted'] = True
                self._save_state()
                continue
            
            # Record extracted clips in our state
            for clip_path in clip_paths:
                clip_info = {
                    'path': clip_path,
                    'source_video_id': video['id'],
                    'source_video_title': video['title'],
                    'topic': video['topic'],
                    'extracted_at': datetime.now().isoformat()
                }
                self.state['clips_extracted'].append(clip_info)
            
            # Mark the video as processed
            video['clips_extracted'] = True
            video['clips_count'] = len(clip_paths)
            video['clips_extracted_at'] = datetime.now().isoformat()
            
            logger.info(f"Extracted {len(clip_paths)} clips from video: {video['title']}")
            self._save_state()
    
    def get_statistics(self):
        """Get statistics about the current state."""
        # Count clips by topic
        clips_by_topic = {}
        for clip in self.state['clips_extracted']:
            topic = clip.get('topic', 'unknown')
            if topic not in clips_by_topic:
                clips_by_topic[topic] = 0
            clips_by_topic[topic] += 1
        
        # Count videos by topic
        videos_by_topic = {}
        for video in self.state['videos_downloaded']:
            topic = video.get('topic', 'unknown')
            if topic not in videos_by_topic:
                videos_by_topic[topic] = 0
            videos_by_topic[topic] += 1
        
        # Build statistics
        stats = {
            'total_topics': len(self.topics),
            'total_videos_downloaded': len(self.state['videos_downloaded']),
            'total_clips_extracted': len(self.state['clips_extracted']),
            'videos_by_topic': videos_by_topic,
            'clips_by_topic': clips_by_topic,
            'last_run': self.state.get('last_run', 'never'),
        }
        
        return stats
    
    def cleanup(self, delete_source_videos=False, days_to_keep=30):
        """
        Clean up old files to save disk space.
        
        Args:
            delete_source_videos (bool): Whether to delete source videos after clips are extracted
            days_to_keep (int): Only delete files older than this many days
        """
        logger.info("Starting cleanup process...")
        now = datetime.now()
        files_removed = 0
        
        # Clean up videos if needed
        if delete_source_videos:
            for video in self.state['videos_downloaded']:
                # Only delete if clips have been extracted
                if video.get('clips_extracted', False) and 'clips_extracted_at' in video:
                    # Check if the video is old enough to delete
                    extracted_date = datetime.fromisoformat(video['clips_extracted_at'])
                    days_old = (now - extracted_date).days
                    
                    if days_old >= days_to_keep and os.path.exists(video['file_path']):
                        try:
                            os.remove(video['file_path'])
                            logger.info(f"Deleted source video: {video['file_path']}")
                            video['file_deleted'] = True
                            video['file_deleted_at'] = now.isoformat()
                            files_removed += 1
                        except Exception as e:
                            logger.error(f"Failed to delete video file: {e}")
        
        # Save the updated state
        self._save_state()
        logger.info(f"Cleanup completed: {files_removed} files removed")


def main():
    """Main entry point for the YouTube Clip Hunter."""
    parser = argparse.ArgumentParser(description="YouTube Clip Hunter - Find and extract interesting clips")
    
    # Basic arguments
    parser.add_argument("--topics", type=str, default="nature,science,history,technology,education",
                       help="Comma-separated list of topics to search for")
    parser.add_argument("--videos-per-topic", type=int, default=2,
                       help="Number of videos to download per topic")
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuously, checking for new videos periodically")
    parser.add_argument("--sleep-time", type=int, default=3600,
                       help="Seconds to sleep between runs in continuous mode")
    
    # Clip extractor arguments
    parser.add_argument("--min-clip-duration", type=int, default=5,
                       help="Minimum duration of extracted clips in seconds")
    parser.add_argument("--max-clip-duration", type=int, default=15,
                       help="Maximum duration of extracted clips in seconds")
    parser.add_argument("--target-clip-count", type=int, default=3,
                       help="Target number of clips to extract per video")
    parser.add_argument("--output-dir", type=str, default="extracted_clips",
                       help="Directory to save extracted clips")
    
    # YouTube scavenger arguments
    parser.add_argument("--download-dir", type=str, default="downloads",
                       help="Directory to save downloaded videos")
    parser.add_argument("--min-video-duration", type=int, default=3600,
                       help="Minimum duration of videos to download in seconds")
    parser.add_argument("--api-key", type=str, default=None,
                       help="YouTube Data API key (optional)")
    parser.add_argument("--video-resolution", type=str, default="720p",
                       help="Video resolution to download (e.g., best, 720p, 480p)")
    
    # Other options
    parser.add_argument("--cleanup", action="store_true",
                       help="Delete source videos after extracting clips")
    parser.add_argument("--cleanup-days", type=int, default=7,
                       help="Days to keep videos before cleanup")
    
    args = parser.parse_args()
    
    # Parse topics
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    
    # Set up the clip extractor args
    clip_extractor_args = {
        'min_clip_duration': args.min_clip_duration,
        'max_clip_duration': args.max_clip_duration,
        'target_clip_count': args.target_clip_count,
        'output_dir': args.output_dir
    }
    
    # Set up the YouTube scavenger args
    scavenger_args = {
        'download_dir': args.download_dir,
        'min_duration': args.min_video_duration,
        'api_key': args.api_key
    }
    
    # Create and run the clip hunter
    clip_hunter = YouTubeClipHunter(
        topics=topics,
        videos_per_topic=args.videos_per_topic,
        clip_extractor_args=clip_extractor_args,
        scavenger_args=scavenger_args
    )
    
    try:
        clip_hunter.run(continuous=args.continuous, sleep_between_runs=args.sleep_time)
        
        # Perform cleanup if requested
        if args.cleanup:
            clip_hunter.cleanup(delete_source_videos=True, days_to_keep=args.cleanup_days)
        
        # Show statistics
        stats = clip_hunter.get_statistics()
        logger.info("Run statistics:")
        logger.info(f"Topics processed: {', '.join(topics)}")
        logger.info(f"Total videos downloaded: {stats['total_videos_downloaded']}")
        logger.info(f"Total clips extracted: {stats['total_clips_extracted']}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Error running clip hunter: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()