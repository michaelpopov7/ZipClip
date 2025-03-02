import whisper
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import os
import yaml

class Subtitler:
    def __init__(self):
        self.config = self.load_config()
        self.video_path = self.config['video']['path']

    @staticmethod 
    def load_config():
        with open('assets/config.yaml', 'r') as file:
            return yaml.safe_load(file)

    def generate_subtitles(self, video_path):
        '''Generate subtitles using Whisper'''
        model = whisper.load_model(self.config['whisper']['model'])
        
        # Transcribe video
        result = model.transcribe(video_path, verbose=True)

        # Format segments into subtitle data
        subtitles = []
        for segment in result["segments"]:
            text = segment["text"].strip()
            subtitles.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": text.capitalize()
            })
        return subtitles
    
    def add_subtitles_to_video(self, video_path, subtitles):
        """Overlay subtitles onto video using MoviePy"""
        # Load the video
        video = VideoFileClip(video_path)
        
        # Define max width for text (e.g., 80% of video width for padding)
        max_text_width = int(video.w * 0.8)
        
        # Create subtitle clips
        subtitle_clips = []
        for sub in subtitles:
            txt_clip = (TextClip(sub["text"], 
                    fontsize=100, 
                    color='white',
                    font='Arial-Bold',
                    stroke_color='black',
                    stroke_width=1,
                    size=(max_text_width, None),
                    method='caption',
                    align='center')
                .set_position(('center', int(video.h * self.config['video']['y_pos'])))  # Using explicit y-coordinate
                .set_start(sub["start"])
                .set_end(sub["end"]))
            subtitle_clips.append(txt_clip)
        
        # Combine video with subtitles
        final_video = CompositeVideoClip([video] + subtitle_clips)
        
        # Generate output path
        output_path = os.path.splitext(video_path)[0] + "_st.mp4"
        
        # Write the result to file
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
        
        return output_path
    
    def process_video(self):
        """Main function to process video and add subtitles"""
        
        try:
            # Verify file exists
            if not os.path.exists(self.video_path):
                raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
            print(f"Processing {self.video_path}...")
            print("Generating subtitles...")
            
            # Generate subtitles
            subtitles = self.generate_subtitles(self.video_path)
            print(f"Generated {len(subtitles)} subtitle segments")
            print("Adding subtitles to video...")
            
            # Add subtitles to video
            output_path = self.add_subtitles_to_video(self.video_path, subtitles)
            
            print(f"Video with subtitles saved as: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    s = Subtitler()
    s.process_video()