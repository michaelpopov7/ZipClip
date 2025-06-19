#!/usr/bin/env python3
"""
ZipClip - Main Pipeline
Converts long videos into interesting short clips using AI-powered transcript analysis.

Pipeline:
1. User selects a long video
2. Extract transcript using Whisper
3. Send transcript to LLM for interesting clip timestamp analysis
4. Extract clips based on LLM timestamps
5. Add captions to extracted clips
"""

import os
import sys
import yaml
import argparse
import logging
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import whisper
from moviepy.editor import VideoFileClip
from openai import OpenAI

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class TranscriptExtractor:
    """Extracts transcript from video using Whisper"""
    
    def __init__(self, model_name="base"):
        self.model_name = model_name
        self.model = None
        
    def _load_model(self):
        """Lazy load Whisper model"""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
        return self.model
    
    def extract_transcript(self, video_path: str) -> Optional[Dict]:
        """
        Extract transcript with timestamps from video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with transcript segments containing text, start, and end times
        """
        logger.info(f"Extracting transcript from: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
            
        try:
            model = self._load_model()
            result = model.transcribe(video_path, verbose=False)
            
            logger.info(f"Transcript extracted with {len(result['segments'])} segments")
            return result
        except Exception as e:
            logger.error(f"Error extracting transcript: {e}")
            return None


class ClipAnalyzer:
    """Uses LLM to analyze transcript and identify interesting clip timestamps"""
    
    def __init__(self, api_key=None, model="gpt-3.5-turbo", use_local_llm=False, local_model="llama3.2"):
        self.model = model
        self.use_local_llm = use_local_llm
        self.local_model = local_model
        
        if use_local_llm and OLLAMA_AVAILABLE:
            self.llm_type = "local"
            logger.info(f"Using local LLM: {local_model}")
        elif api_key:
            self.client = OpenAI(api_key=api_key)
            self.llm_type = "openai"
            logger.info(f"Using OpenAI: {model}")
        else:
            # Try to get from environment
            try:
                self.client = OpenAI()  # Uses OPENAI_API_KEY env var
                self.llm_type = "openai"
                logger.info(f"Using OpenAI: {model}")
            except Exception:
                logger.warning("No OpenAI API key provided and local LLM not configured. Using fallback analysis.")
                self.client = None
                self.llm_type = "fallback"
    
    def analyze_transcript(self, transcript_result: Dict, prompt: str, num_clips: int = 5) -> List[Dict]:
        """
        Analyze transcript using LLM to find interesting moments
        
        Args:
            transcript_result: Whisper transcript result
            prompt: User prompt describing what makes clips interesting
            num_clips: Number of clips to identify
            
        Returns:
            List of clip info dicts with start_time, end_time, reason
        """
        logger.info(f"Analyzing transcript for interesting moments using prompt: '{prompt}'")
        
        if not transcript_result or 'segments' not in transcript_result:
            logger.error("Invalid transcript result")
            return []
        
        # Create full transcript text with timestamps for LLM analysis
        transcript_text = self._format_transcript_for_llm(transcript_result['segments'])
        
        if self.llm_type == "local":
            return self._analyze_with_local_llm(transcript_text, prompt, num_clips)
        elif self.llm_type == "openai" and self.client:
            return self._analyze_with_openai(transcript_text, prompt, num_clips)
        else:
            return self._analyze_fallback(transcript_result['segments'], num_clips)
    
    def _format_transcript_for_llm(self, segments: List[Dict]) -> str:
        """Format transcript segments for LLM analysis"""
        formatted_lines = []
        for segment in segments:
            start_time = self._format_time(segment['start'])
            end_time = self._format_time(segment['end'])
            text = segment['text'].strip()
            formatted_lines.append(f"[{start_time} - {end_time}] {text}")
        
        return "\n".join(formatted_lines)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def _analyze_with_local_llm(self, transcript_text: str, prompt: str, num_clips: int) -> List[Dict]:
        """Use local LLM via Ollama to analyze transcript"""
        try:
            system_prompt = f"""You are an expert video editor tasked with identifying the most interesting moments in a video transcript for creating short clips.

Based on the user's criteria: "{prompt}"

Analyze the following transcript and identify {num_clips} interesting moments that would make engaging short clips.

For each moment, provide:
1. Start time (in seconds)
2. End time (in seconds) - clips should be 15-60 seconds long
3. Brief reason why this moment is interesting

Respond in valid JSON format:
{{
  "clips": [
    {{
      "start_time": 120.5,
      "end_time": 180.0,
      "reason": "Explanation of why this moment is interesting"
    }}
  ]
}}

Transcript:
{transcript_text}"""

            response = ollama.chat(
                model=self.local_model,
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                options={
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            )
            
            # Parse JSON response
            response_text = response['message']['content'].strip()
            
            # Clean up response if it has markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(response_text)
            clips = result.get("clips", [])
            
            logger.info(f"Local LLM identified {len(clips)} interesting moments")
            return clips
            
        except Exception as e:
            logger.error(f"Error analyzing with local LLM: {e}")
            logger.info("Falling back to simple analysis")
            # Pass segments instead of transcript_text for fallback
            segments = transcript_text.split('\n')
            segment_dicts = []
            for line in segments:
                if line.strip() and '] ' in line:
                    try:
                        time_part = line.split('] ')[0][1:]  # Remove [ and get time part
                        start_time = float(time_part.split(' - ')[0].replace(':', '')) * 60  # Simple conversion
                        segment_dicts.append({'start': start_time, 'end': start_time + 30})
                    except:
                        continue
            return self._analyze_fallback(segment_dicts, num_clips)

    def _analyze_with_openai(self, transcript_text: str, prompt: str, num_clips: int) -> List[Dict]:
        """Use OpenAI API to analyze transcript"""
        try:
            system_prompt = f"""You are an expert video editor tasked with identifying the most interesting moments in a video transcript for creating short clips.

Based on the user's criteria: "{prompt}"

Analyze the following transcript and identify {num_clips} interesting moments that would make engaging short clips.

For each moment, provide:
1. Start time (in seconds)
2. End time (in seconds) - clips should be 15-60 seconds long
3. Brief reason why this moment is interesting

Respond in valid JSON format:
{{
  "clips": [
    {{
      "start_time": 120.5,
      "end_time": 180.0,
      "reason": "Explanation of why this moment is interesting"
    }}
  ]
}}

Transcript:
{transcript_text}"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                temperature=0.3
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Clean up response if it has markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(response_text)
            clips = result.get("clips", [])
            
            logger.info(f"OpenAI LLM identified {len(clips)} interesting moments")
            return clips
            
        except Exception as e:
            logger.error(f"Error analyzing with OpenAI: {e}")
            logger.info("Falling back to simple analysis")
            # Pass segments instead of transcript_text for fallback
            segments = transcript_text.split('\n')
            segment_dicts = []
            for line in segments:
                if line.strip() and '] ' in line:
                    try:
                        time_part = line.split('] ')[0][1:]  # Remove [ and get time part
                        start_time = float(time_part.split(' - ')[0].replace(':', '')) * 60  # Simple conversion
                        segment_dicts.append({'start': start_time, 'end': start_time + 30})
                    except:
                        continue
            return self._analyze_fallback(segment_dicts, num_clips)
    
    def _analyze_fallback(self, segments: List[Dict], num_clips: int) -> List[Dict]:
        """Fallback analysis when LLM is not available"""
        logger.info("Using fallback analysis (evenly spaced clips)")
        
        if not segments:
            return []
        
        total_duration = segments[-1]['end'] if segments else 0
        clip_duration = min(45, total_duration / num_clips * 0.8)  # 45 sec max, leave gaps
        
        clips = []
        for i in range(num_clips):
            start_pos = i * (total_duration / num_clips)
            start_time = start_pos
            end_time = min(total_duration, start_time + clip_duration)
            
            clips.append({
                "start_time": start_time,
                "end_time": end_time,
                "reason": f"Evenly spaced clip {i+1}"
            })
        
        return clips


class ClipExtractor:
    """Extracts video clips based on timestamps"""
    
    def extract_clips(self, video_path: str, clip_info_list: List[Dict], output_dir: str) -> List[str]:
        """
        Extract clips from video based on timestamp information
        
        Args:
            video_path: Path to source video
            clip_info_list: List of clip info with start_time, end_time
            output_dir: Directory to save clips
            
        Returns:
            List of paths to extracted clip files
        """
        logger.info(f"Extracting {len(clip_info_list)} clips from {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return []
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            video = VideoFileClip(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            extracted_clips = []
            
            for i, clip_info in enumerate(clip_info_list):
                start_time = clip_info['start_time']
                end_time = clip_info['end_time']
                reason = clip_info.get('reason', f'Clip {i+1}')
                
                # Ensure times are within video duration
                start_time = max(0, start_time)
                end_time = min(video.duration, end_time)
                
                if end_time <= start_time:
                    logger.warning(f"Invalid clip times: {start_time}-{end_time}. Skipping.")
                    continue
                
                # Create clip directory
                clip_dir = os.path.join(output_dir, f"clip_{i+1}")
                os.makedirs(clip_dir, exist_ok=True)
                
                # Extract clip
                clip_path = os.path.join(clip_dir, "clip.mp4")
                
                try:
                    clip = video.subclip(start_time, end_time)
                    clip.write_videofile(
                        clip_path,
                        codec="libx264",
                        audio_codec="aac",
                        temp_audiofile="temp-audio.m4a",
                        remove_temp=True,
                        logger=None
                    )
                    
                    # Save clip metadata
                    metadata_path = os.path.join(clip_dir, "metadata.json")
                    metadata = {
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time,
                        "reason": reason,
                        "source_video": video_path
                    }
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    logger.info(f"Extracted clip {i+1}: {reason}")
                    extracted_clips.append(clip_path)
                    
                except Exception as e:
                    logger.error(f"Error extracting clip {i+1}: {e}")
            
            video.close()
            return extracted_clips
            
        except Exception as e:
            logger.error(f"Error loading video for clip extraction: {e}")
            return []


class VideoCaptioner:
    """Adds captions to video files"""
    
    def __init__(self, whisper_model="base"):
        self.whisper_model = whisper_model
        self.model = None
    
    def _load_model(self):
        """Lazy load Whisper model"""
        if self.model is None:
            logger.info(f"Loading Whisper model for captioning: {self.whisper_model}")
            self.model = whisper.load_model(self.whisper_model)
        return self.model
    
    def add_captions(self, video_path: str, output_path: str) -> bool:
        """
        Add captions to a video file
        
        Args:
            video_path: Input video path
            output_path: Output video path with captions
            
        Returns:
            Success status
        """
        logger.info(f"Adding captions to: {os.path.basename(video_path)}")
        
        try:
            from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
            
            # Load video
            video = VideoFileClip(video_path)
            
            # Generate transcript
            model = self._load_model()
            result = model.transcribe(video_path, verbose=False)
            
            # Create subtitle clips
            subtitle_clips = []
            for segment in result["segments"]:
                text = segment["text"].strip()
                if text:
                    txt_clip = (TextClip(text,
                                        fontsize=50,
                                        color='white',
                                        font='Arial-Bold',
                                        stroke_color='black',
                                        stroke_width=2,
                                        size=(int(video.w * 0.8), None),
                                        method='caption',
                                        align='center')
                                .set_position(('center', int(video.h * 0.75)))
                                .set_start(segment["start"])
                                .set_end(segment["end"]))
                    
                    subtitle_clips.append(txt_clip)
            
            # Combine video with subtitles
            if subtitle_clips:
                final_video = CompositeVideoClip([video] + subtitle_clips)
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    fps=video.fps,
                    logger=None
                )
                final_video.close()
            else:
                # No subtitles, just copy video
                video.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)
            
            video.close()
            logger.info(f"Captioned video saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding captions: {e}")
            return False


class ZipClipPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path="assets/config.yaml"):
        self.config = self._load_config(config_path)
        
        # Initialize components
        whisper_model = self.config.get("whisper", {}).get("model", "base")
        self.transcript_extractor = TranscriptExtractor(whisper_model)
        
        # LLM configuration
        openai_key = self.config.get("openai_api_key")
        llm_model = self.config.get("llm_model", "gpt-3.5-turbo")
        use_local_llm = self.config.get("local_llm", {}).get("enabled", False)
        local_model = self.config.get("local_llm", {}).get("model", "llama3.2")
        
        self.clip_analyzer = ClipAnalyzer(
            api_key=openai_key, 
            model=llm_model,
            use_local_llm=use_local_llm,
            local_model=local_model
        )
        
        self.clip_extractor = ClipExtractor()
        self.captioner = VideoCaptioner(whisper_model)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config or {}
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def process_video(self, video_path: str, prompt: str, output_dir: str = "output", num_clips: int = 5) -> bool:
        """
        Process a single video through the complete pipeline
        
        Args:
            video_path: Path to input video
            prompt: User prompt describing interesting moments
            output_dir: Directory for output clips
            num_clips: Number of clips to extract
            
        Returns:
            Success status
        """
        logger.info(f"Starting pipeline for video: {video_path}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Extracting {num_clips} clips")
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        final_output_dir = os.path.join(output_dir, f"{video_name}_{timestamp}")
        os.makedirs(final_output_dir, exist_ok=True)
        
        try:
            # Step 1: Extract transcript
            logger.info("Step 1: Extracting transcript...")
            transcript_result = self.transcript_extractor.extract_transcript(video_path)
            if not transcript_result:
                logger.error("Failed to extract transcript")
                return False
            
            # Save transcript
            transcript_path = os.path.join(final_output_dir, "transcript.json")
            with open(transcript_path, 'w') as f:
                json.dump(transcript_result, f, indent=2)
            logger.info(f"Transcript saved to: {transcript_path}")
            
            # Step 2: Analyze transcript with LLM
            logger.info("Step 2: Analyzing transcript for interesting moments...")
            clip_info_list = self.clip_analyzer.analyze_transcript(transcript_result, prompt, num_clips)
            if not clip_info_list:
                logger.error("Failed to identify interesting moments")
                return False
            
            # Save analysis results
            analysis_path = os.path.join(final_output_dir, "analysis.json")
            analysis_data = {
                "prompt": prompt,
                "num_clips_requested": num_clips,
                "clips_identified": len(clip_info_list),
                "clips": clip_info_list
            }
            with open(analysis_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            logger.info(f"Analysis saved to: {analysis_path}")
            
            # Step 3: Extract clips
            logger.info("Step 3: Extracting clips...")
            extracted_clips = self.clip_extractor.extract_clips(video_path, clip_info_list, final_output_dir)
            if not extracted_clips:
                logger.error("Failed to extract clips")
                return False
            
            # Step 4: Add captions to clips
            logger.info("Step 4: Adding captions to clips...")
            for i, clip_path in enumerate(extracted_clips):
                clip_dir = os.path.dirname(clip_path)
                captioned_path = os.path.join(clip_dir, "captioned_clip.mp4")
                
                if self.captioner.add_captions(clip_path, captioned_path):
                    # Remove original uncaptioned clip
                    try:
                        os.remove(clip_path)
                        # Rename captioned clip to final name
                        final_clip_path = os.path.join(clip_dir, "final_clip.mp4")
                        os.rename(captioned_path, final_clip_path)
                        logger.info(f"Completed clip {i+1}")
                    except Exception as e:
                        logger.warning(f"Error finalizing clip {i+1}: {e}")
                else:
                    logger.warning(f"Failed to add captions to clip {i+1}")
            
            logger.info(f"Pipeline completed successfully! Output saved to: {final_output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="ZipClip - AI-Powered Video Clip Extraction")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--prompt", "-p", required=True, help="Describe what makes clips interesting")
    parser.add_argument("--clips", "-n", type=int, default=5, help="Number of clips to extract (default: 5)")
    parser.add_argument("--output", "-o", default="output", help="Output directory (default: output)")
    parser.add_argument("--config", "-c", default="assets/config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ZipClipPipeline(config_path=args.config)
    
    # Process video
    success = pipeline.process_video(
        video_path=args.video_path,
        prompt=args.prompt,
        output_dir=args.output,
        num_clips=args.clips
    )
    
    if success:
        logger.info("Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()