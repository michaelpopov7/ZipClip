#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
import subprocess
import multiprocessing
from datetime import datetime
import logging
import torch
import cv2
import numpy as np
from PIL import Image
import whisper
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from transformers import BlipForConditionalGeneration, AutoProcessor, pipeline

# Setup simple console logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class VideoDownloader:
    """Downloads videos from YouTube or other supported platforms"""
    
    def __init__(self, output_dir="./downloads"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def download(self, video_url, channel_name=None):
        """
        Download a video from URL
        
        Args:
            video_url: URL of the video to download
            channel_name: Optional channel name for organizing downloads
            
        Returns:
            str: Path to the downloaded video file
        """
        logger.info(f"Downloading video from: {video_url}")
        
        # Create output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if channel_name:
            channel_output_dir = os.path.join(self.output_dir, channel_name)
            os.makedirs(channel_output_dir, exist_ok=True)
            output_template = f"{channel_output_dir}/{channel_name}_{timestamp}"
        else:
            output_template = f"{self.output_dir}/video_{timestamp}"
        
        # Build command to download with yt-dlp
        download_command = [
            "yt-dlp",
            video_url,
            "-o", f"{output_template}.%(ext)s",
            "--no-playlist"
        ]
        
        try:
            # Execute the download
            result = subprocess.run(download_command, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Download failed: {result.stderr}")
                return None
            
            # Find the downloaded file (extension might vary)
            for file in os.listdir(os.path.dirname(output_template)):
                if file.startswith(os.path.basename(output_template)) and file.endswith(('.mp4', '.mkv', '.webm')):
                    downloaded_path = os.path.join(os.path.dirname(output_template), file)
                    logger.info(f"Video downloaded to: {downloaded_path}")
                    return downloaded_path
            
            logger.error("Could not locate downloaded video file")
            return None
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return None


class VideoParser:
    """Extracts clips from a longer video"""
    
    def __init__(self, min_clip_duration=3, max_clip_duration=30, target_duration=None):
        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration
        self.target_duration = target_duration or max_clip_duration
        logger.info(f"VideoParser initialized with target_duration={self.target_duration}")
        
    def parse_video(self, video_path, num_clips, output_base_dir):
        """
        Extract a specified number of clips from a video
        
        Args:
            video_path: Path to the input video
            num_clips: Number of clips to extract
            output_base_dir: Base directory for extracted clips (videos/channel_name)
            
        Returns:
            list: List of dictionaries with clip info (index, path, etc.)
        """
        logger.info(f"Parsing video into {num_clips} clips: {video_path}")
        
        # Load video
        try:
            video = VideoFileClip(video_path)
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            return []
        
        video_duration = video.duration
        logger.info(f"Video duration: {video_duration:.2f} seconds")
        
        # Use target duration for clips, respecting min/max bounds
        clip_duration = min(self.max_clip_duration, max(self.min_clip_duration, self.target_duration))
        logger.info(f"Using clip duration of {clip_duration} seconds")
        
        # Calculate start times for evenly spaced clips
        available_duration = video_duration - (num_clips * clip_duration)
        if available_duration <= 0:
            # If video is too short, overlap clips
            spacing = max(0, video_duration / num_clips - clip_duration)
        else:
            spacing = available_duration / (num_clips - 1) if num_clips > 1 else 0
        
        start_times = [i * (clip_duration + spacing) for i in range(num_clips)]
        
        # Cap the last clip to the video duration
        if start_times and start_times[-1] + clip_duration > video_duration:
            start_times[-1] = max(0, video_duration - clip_duration)
        
        # Extract clips directly to their final destination folders
        clips_info = []
        for i, start_time in enumerate(start_times):
            if start_time >= video_duration:
                logger.warning(f"Skipping clip {i+1} - start time exceeds video duration")
                continue
                
            end_time = min(video_duration, start_time + clip_duration)
            
            # Create clip directory structure
            clip_num = i + 1
            clip_folder_name = f"clip_{clip_num}"
            clip_dir = os.path.join(output_base_dir, clip_folder_name)
            os.makedirs(clip_dir, exist_ok=True)
            
            # Create a temporary file for processing
            temp_clip_path = os.path.join(clip_dir, "temp_original.mp4")
            
            try:
                clip = video.subclip(start_time, end_time)
                clip.write_videofile(temp_clip_path, codec="libx264", audio_codec="aac", 
                                    temp_audiofile="temp-audio.m4a", remove_temp=True,
                                    logger=None)
                logger.info(f"Extracted clip {i+1}/{num_clips} to {clip_dir}")
                
                clips_info.append({
                    'index': i,
                    'clip_num': clip_num,
                    'path': temp_clip_path,
                    'clip_dir': clip_dir,
                    'start_time': start_time,
                    'end_time': end_time
                })
            except Exception as e:
                logger.error(f"Error saving clip {i+1}: {e}")
        
        # Close video
        video.close()
        
        return clips_info


class VerticalTransformer:
    """Transforms videos to vertical format for social media"""
    
    def __init__(self, width=1080, height=1920, background_color="black"):
        self.width = width
        self.height = height
        self.background_color = background_color
        
    def transform(self, video_path, output_path, crop_mode=True):
        """
        Transform a video to vertical format
        
        Args:
            video_path: Path to the input video
            output_path: Path where the transformed video will be saved
            crop_mode: Whether to crop (True) or scale with padding (False)
            
        Returns:
            bool: Success status
        """
        logger.info(f"Transforming video to vertical: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"Input file not found: {video_path}")
            return False
        
        # Get video dimensions using ffprobe
        try:
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                video_path
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Error getting video dimensions: {result.stderr}")
                return False
            
            original_width, original_height = map(int, result.stdout.strip().split(','))
            logger.info(f"Original dimensions: {original_width}x{original_height}")
            
            # Build FFmpeg command based on transformation mode
            if crop_mode:
                # Center crop the video
                target_aspect_ratio = self.width / self.height
                original_aspect_ratio = original_width / original_height
                
                if original_aspect_ratio > target_aspect_ratio:
                    # Video is wider than target, crop width
                    crop_height = original_height
                    crop_width = int(original_height * target_aspect_ratio)
                    x_offset = (original_width - crop_width) // 2
                    y_offset = 0
                    
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-i", video_path,
                        "-vf", f"crop={crop_width}:{crop_height}:{x_offset}:{y_offset},scale={self.width}:{self.height}",
                        "-c:a", "copy",
                        output_path
                    ]
                else:
                    # Video is taller than target, crop height
                    crop_width = original_width
                    crop_height = int(original_width / target_aspect_ratio)
                    x_offset = 0
                    y_offset = (original_height - crop_height) // 2
                    
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-i", video_path,
                        "-vf", f"crop={crop_width}:{crop_height}:{x_offset}:{y_offset},scale={self.width}:{self.height}",
                        "-c:a", "copy",
                        output_path
                    ]
            else:
                # Scale with padding
                scale_factor = min(self.width / original_width, self.height / original_height)
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                x_offset = (self.width - new_width) // 2
                y_offset = (self.height - new_height) // 2
                
                # Use solid color background
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-vf", f"scale={new_width}:{new_height},pad={self.width}:{self.height}:{x_offset}:{y_offset}:{self.background_color}",
                    "-c:a", "copy",
                    output_path
                ]
            
            # Run FFmpeg command
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error transforming video: {result.stderr}")
                return False
            
            logger.info(f"Video transformed to vertical format: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error in vertical transformation: {e}")
            return False


class VideoSubtitler:
    """Adds subtitles to videos using Whisper for speech recognition"""
    
    def __init__(self, whisper_model="base", y_position=0.75):
        self.whisper_model_name = whisper_model
        self.whisper_model = None  # Lazy loading
        self.y_position = y_position  # Position of subtitles (0=top, 1=bottom)
        
    def _load_model(self):
        """Lazy-load the Whisper model"""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper model: {self.whisper_model_name}")
            self.whisper_model = whisper.load_model(self.whisper_model_name)
        return self.whisper_model
    
    def add_subtitles(self, video_path, output_path):
        """
        Add subtitles to a video using Whisper for transcription
        
        Args:
            video_path: Path to the input video
            output_path: Path where subtitled video will be saved
            
        Returns:
            bool: Success status
        """
        logger.info(f"Adding subtitles to video: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False
        
        try:
            # Load video
            video = VideoFileClip(video_path)
            
            # Generate subtitles with Whisper
            model = self._load_model()
            logger.info("Transcribing audio with Whisper...")
            result = model.transcribe(video_path, verbose=False)
            
            # Format subtitles
            subtitles = []
            for segment in result["segments"]:
                text = segment["text"].strip()
                if text:  # Only add non-empty segments
                    subtitles.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": text
                    })
            
            logger.info(f"Generated {len(subtitles)} subtitle segments")
            
            if not subtitles:
                logger.warning("No subtitles generated, returning failure")
                return False
            
            # Define max width for text (80% of video width)
            max_text_width = int(video.w * 0.8)
            
            # Create subtitle clips
            subtitle_clips = []
            for sub in subtitles:
                txt_clip = (TextClip(sub["text"], 
                        fontsize=50, 
                        color='white',
                        font='Arial-Bold',
                        stroke_color='black',
                        stroke_width=1.5,
                        size=(max_text_width, None),
                        method='caption',
                        align='center')
                    .set_position(('center', int(video.h * self.y_position)))
                    .set_start(sub["start"])
                    .set_end(sub["end"]))
                subtitle_clips.append(txt_clip)
            
            # Combine video with subtitles
            final_video = CompositeVideoClip([video] + subtitle_clips)
            
            # Write the result to file
            logger.info(f"Writing subtitled video to: {output_path}")
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=video.fps
            )
            
            # Close video clips to free memory
            video.close()
            final_video.close()
            
            logger.info(f"Subtitled video saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding subtitles: {e}")
            return False


class VideoCaptioner:
    """Generates AI captions for videos"""
    
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", num_frames=8):
        self.model_name = model_name
        self.num_frames = num_frames
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _load_model(self):
        """Lazy-load the captioning model"""
        if self.model is None:
            logger.info(f"Loading captioning model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        return self.model, self.processor
    
    def extract_frames(self, video_path):
        """Extract evenly spaced frames from video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        
        cap.release()
        return frames
    
    def generate_caption(self, video_path):
        """
        Generate a caption for a video
        
        Args:
            video_path: Path to the video
            
        Returns:
            tuple: (caption text, hashtags list)
        """
        logger.info(f"Generating caption for video: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None, None
        
        try:
            # Extract frames from the video
            frames = self.extract_frames(video_path)
            if not frames:
                logger.error("Failed to extract frames from video")
                return None, None
            
            # Load model
            model, processor = self._load_model()
            
            # Generate captions for frames
            captions = []
            for i, frame in enumerate(frames):
                inputs = processor(frame, return_tensors="pt").to(self.device)
                output = model.generate(**inputs, max_length=50)
                caption = processor.decode(output[0], skip_special_tokens=True)
                captions.append(caption)
                logger.info(f"Frame {i+1} caption: {caption}")
            
            # Combine captions into a 2-3 sentence summary
            if captions:
                # Use the two longest captions as they might be the most informative
                sorted_captions = sorted(captions, key=len, reverse=True)[:2]
                combined_caption = " ".join(sorted_captions)
                logger.info(f"Combined caption: {combined_caption}")
                
                # Generate hashtags
                hashtags = self._generate_hashtags(combined_caption)
                logger.info(f"Generated hashtags: {hashtags}")
                
                return combined_caption, hashtags
            else:
                logger.warning("No captions generated")
                return None, None
                
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return None, None
    
    def _generate_hashtags(self, caption, num_tags=15):
        """Generate hashtags based on the caption content"""
        try:
            # Initialize zero-shot classifier for hashtag generation
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            
            # Set up potential topics for video content
            candidate_labels = [
                "travel", "nature", "food", "fashion", "fitness", "technology", 
                "music", "art", "photography", "lifestyle", "motivation", "education",
                "business", "health", "sports", "beauty", "comedy", "family", 
                "dance", "gaming", "science", "books", "movie", "adventure", 
                "creative", "inspiration", "fun", "entertainment", "DIY", "cooking",
                "pets", "animals", "architecture", "design", "cars", "finance",
                "wellness", "vacation", "workout", "meditation", "history", "culture"
            ]
            
            # Get classification results
            result = classifier(caption, candidate_labels, multi_label=True)
            
            # Select top hashtags based on scores
            selected_hashtags = result['labels'][:num_tags]
            
            # Add some common social media hashtags
            common_hashtags = ["video", "content", "viral", "trending", "follow"]
            
            # Combine and ensure we have num_tags hashtags
            all_hashtags = selected_hashtags + common_hashtags
            final_hashtags = list(dict.fromkeys(all_hashtags))[:num_tags]  # Remove duplicates
            
            return final_hashtags
        except Exception as e:
            logger.error(f"Error generating hashtags: {e}")
            return ["video", "content", "trending", "viral", "follow", "share", 
                    "like", "comment", "social", "media", "clip", "watch", 
                    "amazing", "cool", "fun"]


class VideoProcessor:
    """Main class that orchestrates the entire video processing pipeline"""
    
    def __init__(self, config_path="assets/config.yaml"):
        self.config = self._load_config(config_path)
        self.downloader = VideoDownloader(output_dir="downloads")
        
        # Initialize parser with target duration from config
        target_duration = self.config.get("target_duration", 10)
        logger.info(f"Configured target duration from config: {target_duration} seconds")
        
        # Set max_clip_duration higher to accommodate larger target_duration values
        max_clip_duration = max(30, target_duration + 5)  # Allow some flexibility
        self.parser = VideoParser(
            target_duration=target_duration,
            max_clip_duration=max_clip_duration
        )
        
        self.transformer = VerticalTransformer()
        
        # Initialize subtitler with config values
        whisper_model = self.config.get("whisper", {}).get("model", "base")
        self.subtitler = VideoSubtitler(whisper_model=whisper_model)
        
        self.captioner = VideoCaptioner()
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def process_channel(self, channel_name, channel_config):
        """Process a single channel from start to finish"""
        logger.info(f"Processing channel: {channel_name}")
        
        # Create necessary directory structure
        channel_video_dir = os.path.join("videos", channel_name)
        os.makedirs(channel_video_dir, exist_ok=True)
        
        # Get video URL from channel configuration
        video_url = channel_config.get("path")
        if not video_url:
            logger.error(f"No video URL found for channel {channel_name}")
            return False
        
        # 1. Download video
        downloaded_video = self.downloader.download(video_url, channel_name)
        if not downloaded_video:
            logger.error(f"Failed to download video for {channel_name}")
            return False
        
        # 2. Parse video into clips directly to final directories
        num_clips = self.config.get("number_of_clips", 20)
        clips_info = self.parser.parse_video(downloaded_video, num_clips, channel_video_dir)
        if not clips_info:
            logger.error(f"Failed to parse video for {channel_name}")
            return False
        
        # 3. Process each clip in its final directory
        for clip_info in clips_info:
            clip_num = clip_info['clip_num']
            clip_dir = clip_info['clip_dir']
            original_clip_path = clip_info['path']
            
            # Define final output filenames in clip directory
            vertical_path = os.path.join(clip_dir, "vertical_st.mp4")
            
            # Apply vertical transformation directly to final location
            if not self.transformer.transform(original_clip_path, vertical_path, crop_mode=True):
                logger.error(f"Failed to transform clip {clip_num} to vertical format")
                # Delete original temp file and continue to next clip
                try:
                    os.remove(original_clip_path)
                except:
                    pass
                continue
                
            # Add subtitles directly to final location
            if not self.subtitler.add_subtitles(vertical_path, vertical_path):
                logger.error(f"Failed to add subtitles to clip {clip_num}")
                # No need to continue, vertical_st.mp4 already exists
            
            # Generate caption for the clip
            caption_output_file = os.path.join(clip_dir, "caption.txt")
            caption, hashtags = self.captioner.generate_caption(vertical_path)
            
            if caption and hashtags:
                # Format caption and hashtags
                hashtag_str = " ".join([f"#{tag}" for tag in hashtags])
                formatted_output = f"{caption}\n\n{hashtag_str}"
                
                # Save to file
                with open(caption_output_file, "w") as f:
                    f.write(formatted_output)
                logger.info(f"Caption saved to {caption_output_file}")
            else:
                logger.error(f"Failed to generate caption for clip {clip_num}")
            
            # Clean up original temporary clip
            try:
                os.remove(original_clip_path)
                logger.info(f"Removed temporary file: {original_clip_path}")
            except Exception as e:
                logger.error(f"Error removing temporary file: {e}")
        
        logger.info(f"Finished processing for channel: {channel_name}")
        return True

    def run(self):
        """Execute the entire pipeline for all channels"""
        if not self.config:
            logger.error("No valid configuration loaded")
            return
        
        channels = self.config.get("channels", {})
        if not channels:
            logger.error("No channels found in configuration")
            return
        
        # Process each channel in parallel
        processes = []
        for channel_name, channel_data in channels.items():
            if isinstance(channel_data, dict) and "path" in channel_data:
                p = multiprocessing.Process(
                    target=self.process_channel, 
                    args=(channel_name, channel_data)
                )
                processes.append(p)
                p.start()
            else:
                logger.warning(f"Skipping channel '{channel_name}': Invalid configuration")
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        logger.info("All channels processed successfully")


def main():
    parser = argparse.ArgumentParser(description="ZipClip Video Processing Pipeline")
    parser.add_argument("--config", "-c", default="assets/config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Initialize and run the pipeline
    processor = VideoProcessor(config_path=args.config)
    processor.run()


if __name__ == "__main__":
    main() 