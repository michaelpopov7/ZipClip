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

try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("MoviePy not available. Video processing features will be limited.")

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
    
    def __init__(self, api_key=None, model="gpt-3.5-turbo", use_local_llm=True, local_model="llama3.2"):
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
                # Force local LLM as default instead of fallback
                logger.warning("No OpenAI API key provided. Forcing local LLM usage.")
                self.llm_type = "local"
                self.use_local_llm = True
    
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
            raise ValueError("Invalid transcript result - cannot proceed without transcript")
        
        # Create full transcript text with timestamps for LLM analysis
        transcript_text = self._format_transcript_for_llm(transcript_result['segments'])
        
        if self.llm_type == "local":
            return self._analyze_with_local_llm(transcript_text, prompt, num_clips)
        elif self.llm_type == "openai" and self.client:
            return self._analyze_with_openai(transcript_text, prompt, num_clips)
        else:
            raise ValueError("No LLM available for analysis - please configure Ollama or OpenAI")
    
    def _format_transcript_for_llm(self, segments: List[Dict]) -> str:
        """Format transcript segments for LLM analysis"""
        logger.info(f"=== TRANSCRIPT FORMATTING DEBUG ===")
        logger.info(f"Number of transcript segments: {len(segments)}")
        
        formatted_lines = []
        for i, segment in enumerate(segments):
            start_time = self._format_time(segment['start'])
            end_time = self._format_time(segment['end'])
            text = segment['text'].strip()
            formatted_line = f"[{start_time} - {end_time}] {text}"
            formatted_lines.append(formatted_line)
            
            # Log first few and last few segments for debugging
            if i < 3 or i >= len(segments) - 3:
                logger.info(f"  Segment {i+1}: {formatted_line}")
            elif i == 3 and len(segments) > 6:
                logger.info(f"  ... ({len(segments) - 6} more segments) ...")
        
        full_transcript = "\n".join(formatted_lines)
        logger.info(f"Complete transcript length: {len(full_transcript)} characters")
        logger.info(f"Total lines in transcript: {len(formatted_lines)}")
        logger.info(f"Transcript is sent as ONE COMPLETE TEXT, not in parts")
        
        return full_transcript
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def _analyze_with_local_llm(self, transcript_text: str, prompt: str, num_clips: int) -> List[Dict]:
        """Use local LLM via Ollama to analyze transcript"""
        if not OLLAMA_AVAILABLE:
            raise ValueError("Ollama is not available - please install ollama package")
        
        try:
            logger.info(f"=== TRANSCRIPT ANALYSIS DEBUG ===")
            logger.info(f"Transcript length: {len(transcript_text)} characters")
            logger.info(f"User prompt: '{prompt}'")
            logger.info(f"Requested clips: {num_clips}")
            logger.info(f"Transcript preview (first 500 chars): {transcript_text[:500]}...")
            logger.info(f"Transcript sample (last 500 chars): ...{transcript_text[-500:]}")
            
            # Check if transcript is too long (Ollama/LLM token limits)
            max_transcript_chars = 50000  # Conservative limit
            if len(transcript_text) > max_transcript_chars:
                logger.warning(f"Transcript is very long ({len(transcript_text)} chars). Truncating to {max_transcript_chars} chars to avoid LLM issues.")
                # Take first and last parts to preserve context
                half_limit = max_transcript_chars // 2
                truncated_transcript = transcript_text[:half_limit] + "\n\n[... MIDDLE CONTENT TRUNCATED ...]\n\n" + transcript_text[-half_limit:]
                transcript_text = truncated_transcript
                logger.info(f"Truncated transcript length: {len(transcript_text)} characters")
            
            system_prompt = f"""Find {num_clips} interesting moments from this video transcript for creating short clips.

User wants: {prompt}

Instructions:
- Each clip should be 15-60 seconds long
- Respond with ONLY valid JSON, no other text
- Use this exact format:

{{
  "clips": [
    {{
      "start_time": 0.0,
      "end_time": 30.0,
      "reason": "brief description"
    }}
  ]
}}

Transcript:
{transcript_text}"""

            logger.info(f"System prompt length: {len(system_prompt)} characters")
            logger.info(f"System prompt preview: {system_prompt[:200]}...")

            # Configure ollama client to connect to the correct hostname
            import ollama
            client = ollama.Client(host='http://ollama:11434')
            
            logger.info("Sending transcript to local LLM for analysis...")
            logger.info(f"Using model: {self.local_model}")
            
            response = client.chat(
                model=self.local_model,
                messages=[
                    {"role": "user", "content": system_prompt}  # Using user role instead of system for better compatibility
                ],
                stream=False,  # CRITICAL FIX: Disable streaming to get complete response
                options={
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            )
            
            logger.info(f"Raw response object type: {type(response)}")
            logger.info(f"Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
            
            # Parse JSON response - handle new response structure
            response_content = response.message.content if hasattr(response, 'message') and hasattr(response.message, 'content') else ""
            
            logger.info(f"Response message content type: {type(response_content)}")
            logger.info(f"Response content length: {len(response_content) if response_content else 0}")
            logger.info(f"Full raw response content: '{response_content}'")
            
            if not response_content or not response_content.strip():
                logger.error(f"Empty response detected!")
                logger.error(f"Response object: {response}")
                if hasattr(response, 'message'):
                    logger.error(f"Message object: {response.message}")
                raise ValueError("Empty response from local LLM")
            
            response_text = response_content.strip()
            logger.info(f"Raw LLM response length: {len(response_text)} characters")
            logger.info(f"Raw response (first 1000 chars): {response_text[:1000]}")
            
            # Clean up response if it has markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
                logger.info("Removed ```json markdown formatting")
            elif response_text.startswith("```"):
                # Handle case where it's just ```
                lines = response_text.split('\n')
                if len(lines) > 2 and lines[0] == "```" and lines[-1] == "```":
                    response_text = '\n'.join(lines[1:-1]).strip()
                    logger.info("Removed ``` markdown formatting")
            
            # Look for JSON content between any surrounding text
            if '{' in response_text and '}' in response_text:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                original_text = response_text
                response_text = response_text[start_idx:end_idx]
                logger.info(f"Extracted JSON from position {start_idx} to {end_idx}")
                logger.info(f"Original text: {original_text}")
            
            logger.info(f"Final cleaned response: {response_text}")
            
            result = json.loads(response_text)
            clips = result.get("clips", [])
            
            if not clips:
                logger.error("LLM returned valid JSON but no clips array")
                logger.error(f"JSON result: {result}")
                raise ValueError("LLM returned no clips")
            
            # Validate clip format
            for i, clip in enumerate(clips):
                logger.info(f"Validating clip {i+1}: {clip}")
                if not isinstance(clip.get('start_time'), (int, float)) or not isinstance(clip.get('end_time'), (int, float)):
                    raise ValueError(f"Clip {i+1} has invalid time format: {clip}")
                if clip['start_time'] >= clip['end_time']:
                    raise ValueError(f"Clip {i+1} has invalid time range: {clip['start_time']}-{clip['end_time']}")
            
            logger.info(f"✅ Local LLM successfully identified {len(clips)} interesting moments")
            for i, clip in enumerate(clips):
                logger.info(f"  Clip {i+1}: {clip['start_time']}s-{clip['end_time']}s | {clip['reason']}")
            
            return clips
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text that failed to parse: {response_text}")
            raise ValueError(f"Local LLM returned invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Error analyzing with local LLM: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise ValueError(f"Local LLM analysis failed: {e}")

    def _analyze_with_openai(self, transcript_text: str, prompt: str, num_clips: int) -> List[Dict]:
        """Use OpenAI API to analyze transcript"""
        try:
            system_prompt = f"""Find {num_clips} interesting moments from this video transcript for creating short clips.

User wants: {prompt}

Instructions:
- Each clip should be 15-60 seconds long
- Respond with ONLY valid JSON, no other text
- Use this exact format:

{{
  "clips": [
    {{
      "start_time": 0.0,
      "end_time": 30.0,
      "reason": "brief description"
    }}
  ]
}}

Transcript:
{transcript_text}"""

            logger.info("Sending transcript to OpenAI for analysis...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                temperature=0.3
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content
            
            if not response_text or not response_text.strip():
                raise ValueError("Empty response from OpenAI")
            
            response_text = response_text.strip()
            logger.info(f"Raw OpenAI response length: {len(response_text)} characters")
            
            # Clean up response if it has markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                lines = response_text.split('\n')
                if len(lines) > 2 and lines[0] == "```" and lines[-1] == "```":
                    response_text = '\n'.join(lines[1:-1]).strip()
            
            # Look for JSON content between any surrounding text
            if '{' in response_text and '}' in response_text:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                response_text = response_text[start_idx:end_idx]
            
            logger.info(f"Cleaned response: {response_text[:200]}...")
            
            result = json.loads(response_text)
            clips = result.get("clips", [])
            
            if not clips:
                raise ValueError("OpenAI returned no clips")
            
            # Validate clip format
            for i, clip in enumerate(clips):
                if not isinstance(clip.get('start_time'), (int, float)) or not isinstance(clip.get('end_time'), (int, float)):
                    raise ValueError(f"Clip {i+1} has invalid time format")
                if clip['start_time'] >= clip['end_time']:
                    raise ValueError(f"Clip {i+1} has invalid time range: {clip['start_time']}-{clip['end_time']}")
            
            logger.info(f"✅ OpenAI successfully identified {len(clips)} interesting moments")
            return clips
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text: {response_text}")
            raise ValueError(f"OpenAI returned invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Error analyzing with OpenAI: {e}")
            raise ValueError(f"OpenAI analysis failed: {e}")


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
        
        if not MOVIEPY_AVAILABLE:
            logger.error("MoviePy is not available. Cannot extract video clips.")
            return []
        
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
                    clip = video.subclipped(start_time, end_time)
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
        
        if not MOVIEPY_AVAILABLE:
            logger.error("MoviePy is not available. Cannot add captions to video.")
            return False
        
        try:
            from moviepy import VideoFileClip, TextClip, CompositeVideoClip
            
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