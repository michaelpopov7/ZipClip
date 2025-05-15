# ZipClip
Clipify, caption, and post.


ZipClip cuts down your large, unedited video files into short-form, captioned clips; ready to capture millions of views.
ZipClip completes the automation circuit by posting to social media platforms of your choosing, without you doing a thing.

## Integrated Pipeline

ZipClip now features an integrated pipeline that automates the entire process:

1. Downloads videos from configured channels
2. Parses videos into interesting clips
3. Transforms clips to vertical format for social media
4. Adds AI-generated subtitles using Whisper
5. Generates engaging captions and hashtags with AI

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure ffmpeg is installed on your system:
```bash
# On macOS
brew install ffmpeg

# On Ubuntu/Debian
sudo apt install ffmpeg
```

3. Configure your channels in `assets/config.yaml`:
```yaml
whisper:
  model: base  # tiny, base, small, medium, large

number_of_clips: 20  # Number of clips to extract per video
target_duration: 15  # Target duration in seconds for each clip (default: 10)

channels:
  family_guy:
    path: https://www.youtube.com/watch?v=6DmXlJwaP0I
  rick_and_morty:
    path: https://www.youtube.com/watch?v=EXAMPLE_URL
```

### Usage

Run the pipeline with:
```bash
python src/video_pipeline.py
```

This will process all channels in parallel. For each channel:
- Original videos are saved to `downloads/`
- Processed clips are placed directly in `videos/<channel_name>/clip_<num>/` with:
  - `vertical_st.mp4`: The final vertical video with subtitles
  - `caption.txt`: AI-generated caption with hashtags

All intermediate files are automatically cleaned up.

### Advanced Configuration

Adjust settings in `assets/config.yaml` to control:
- Whisper model size (affects subtitle accuracy)
- Number of clips per video
- Target duration for clips in seconds
- Video channels and sources

Example of a more detailed configuration:
```yaml
whisper:
  model: base  # tiny, base, small, medium, large

number_of_clips: 20  # Number of clips to extract per video
target_duration: 15  # Target duration in seconds for each clip (default: 10)

channels:
  family_guy:
    path: https://www.youtube.com/watch?v=6DmXlJwaP0I
  rick_and_morty:
    path: https://www.youtube.com/watch?v=EXAMPLE_URL
```

### Output Structure

The pipeline creates a clean, organized output structure:
```
videos/
  └── channel_name/
      ├── clip_1/
      │   ├── vertical_st.mp4  # Final video with vertical format and subtitles
      │   └── caption.txt      # AI-generated caption with hashtags
      ├── clip_2/
      │   ├── vertical_st.mp4
      │   └── caption.txt
      └── ...
```

All intermediate processing files are automatically cleaned up.

pip3 install --upgrade yt-dlp