#!/usr/bin/env python3
import yaml
import os
import subprocess
import multiprocessing
from datetime import datetime

# Assuming your scripts are in the same directory or accessible via PYTHONPATH
# If not, you might need to adjust sys.path or how you call them
# For simplicity, this example assumes they are executable and in PATH or same dir

def run_command(command_list):
    """Helper function to run a shell command and print its output."""
    try:
        print(f"Executing: {' '.join(command_list)}")
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error executing {' '.join(command_list)}:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False
        print(f"Successfully executed: {' '.join(command_list)}")
        print(f"Output:\n{stdout.decode()}")
        return True
    except Exception as e:
        print(f"Exception during command {' '.join(command_list)}: {e}")
        return False

def process_channel(channel_name, channel_config, global_config):
    """
    Processes a single channel: download, parse, transform, subtitle, caption.
    """
    print(f"Processing channel: {channel_name}")
    
    # 0. Create necessary directories
    download_dir = "downloads"
    channel_video_dir = os.path.join("videos", channel_name)
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(channel_video_dir, exist_ok=True)

    video_url = channel_config.get("path")
    if not video_url:
        print(f"No path (URL) found for channel {channel_name}. Skipping.")
        return

    # 1. Download video
    # Assuming single_downloader.py takes URL and output directory
    # Modify this if your downloader works differently
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    downloaded_video_filename = f"{channel_name}_{timestamp}.mp4" # Or get from downloader output
    downloaded_video_path = os.path.join(download_dir, downloaded_video_filename)
    
    # We need to know the exact output filename from the downloader.
    # For now, let's assume a fixed name or that downloader returns it.
    # Using a placeholder. You'll need to adjust based on single_downloader.py behavior.
    # A better single_downloader.py would take an output path argument.
    # Let's assume single_downloader.py saves to downloads/{channel_name}.mp4 for now
    # And we will rename it
    
    # Let's try to make single_downloader.py more flexible or get its output.
    # For now, constructing the command. Adjust as per single_downloader.py capabilities.
    # It seems single_downloader.py has a hardcoded output path pattern.
    # This will be tricky without modifying it. Let's assume for now it downloads
    # to a predictable location or that we can find the latest file.
    
    # Let's call single_downloader.py and then find the downloaded file.
    # This is a common pattern if the script doesn't allow specifying output.
    print(f"Downloading video for {channel_name}...")
    download_cmd = [
        "python", "src/single_downloader.py", 
        "--url", video_url, 
        "--output_dir", download_dir 
    ] # Assuming single_downloader.py can take --output_dir
    if not run_command(download_cmd):
        print(f"Failed to download video for {channel_name}. Skipping further processing.")
        return

    # After download, we need to find the exact file name.
    # This is a placeholder. The downloader should ideally return the path or take an output path.
    # Let's assume the downloader script is modified to output the downloaded file path.
    # If single_downloader.py prints the path of the downloaded file, we can capture it.
    # For now, let's assume it's the newest .mp4 file in downloads/
    try:
        list_of_files = [os.path.join(download_dir, f) for f in os.listdir(download_dir) if f.endswith('.mp4')]
        if not list_of_files:
            print(f"No MP4 file found in {download_dir} after download for {channel_name}.")
            return
        downloaded_video_path = max(list_of_files, key=os.path.getctime) # Get the latest one
        print(f"Downloaded video identified as: {downloaded_video_path}")
    except Exception as e:
        print(f"Error finding downloaded video for {channel_name}: {e}")
        return

    # 2. Parse video into clips
    # Assuming parser.py takes input video, output dir, and number of clips
    num_clips = global_config.get("number_of_clips", 1) # Default to 1 clip if not specified
    print(f"Parsing {downloaded_video_path} into {num_clips} clips for {channel_name}...")
    
    # The parser.py expects a specific structure, let's adapt.
    # It seems parser.py might create its own subdirectories.
    # We need to ensure clips are in videos/<channel_name>/clip_<n>/
    # Let's assume parser.py can take an output_base_dir and channel_name
    
    # We might need to modify parser.py to fit this structure or call it per clip.
    # For now, let's assume parser.py takes an input video and a base output directory
    # and creates subfolders like 'clip_1', 'clip_2' etc. inside it.
    # Output for parser will be videos/<channel_name>/
    
    parse_cmd = [
        "python", "src/parser.py",
        "--input_video", downloaded_video_path,
        "--output_dir", channel_video_dir, # Clips will go into videos/channel_name/
        "--num_clips", str(num_clips),
        "--channel_name", channel_name # If parser can use this for subfolder naming
    ]
    # The parser script seems to create `extracted_clips/channel_name/video_name_clip_N.mp4`
    # We need to adapt this.
    # Let's simplify: assume parser.py will put clips directly into `channel_video_dir/clip_X/original.mp4`
    # This requires parser.py to be flexible or to be modified.

    # For now, I'll assume parser.py is called and then we find the clips.
    # This part is highly dependent on parser.py's actual behavior.
    # Let's call a hypothetical modified parser.
    # A more robust way would be to modify parser.py to take specific output paths per clip
    # or to output a manifest of created clips.

    # Calling parser.py (this is a simplified call, actual parser might need different args)
    # Example: parser.py --input_video downloaded.mp4 --output_base videos/family_guy --num_clips 3
    # This would ideally create videos/family_guy/clip_1.mp4, videos/family_guy/clip_2.mp4 etc.
    # Or videos/family_guy/clip_1/original.mp4 etc.
    
    # Let's assume parser.py puts them in `channel_video_dir` with names like `clip_1.mp4`, `clip_2.mp4`...
    # And we will then process each clip.

    # We need to modify parser.py to output to the desired structure.
    # For now, this part is a placeholder for how parser.py should be invoked.
    # The desired structure is videos/channel_name/clip_X/original_clip.mp4
    
    # Let's assume parser.py is run and creates files like:
    # extracted_clips/channel_name/downloaded_video_filename_clip_0.mp4
    # We need to move these to the target structure.

    # Call parser.py as is, then reorganize.
    # The parser.py in the context seems to save to `extracted_clips/CHANNEL_NAME/VID_NAME_clip_X.mp4`
    temp_clip_output_dir = os.path.join("extracted_clips", channel_name)
    
    parse_cmd_orig = [
        "python", "src/parser.py", # This might be test.py based on previous logs or parser.py
        downloaded_video_path, # input_video
        channel_name,          # channel_name
        str(num_clips)         # num_clips
        # output_dir is not explicitly taken by the parser.py call structure.
        # It seems to be derived.
    ]
    print(f"Running parser: {' '.join(parse_cmd_orig)}")
    if not run_command(parse_cmd_orig):
         print(f"Failed to parse video for {channel_name}. Skipping clip processing.")
         return

    # Find generated clips from the parser's default output directory
    parsed_clip_files = []
    source_video_name_without_ext = os.path.splitext(os.path.basename(downloaded_video_path))[0]
    
    # construct the path where parser.py saves clips
    # e.g., extracted_clips/family_guy/family_guy_20231027_120000_clip_0.mp4
    # The parser script seems to be `parser.py` not `test.py` from the list_dir output
    
    for i in range(num_clips):
        # This filename pattern needs to match what parser.py actually produces
        parsed_clip_name = f"{source_video_name_without_ext}_clip_{i}.mp4"
        # The parser.py from context saves to extracted_clips/channel_name/video_name_clip_X.mp4
        # So, source_video_name_without_ext should be the name of the *downloaded* video.
        
        expected_clip_path = os.path.join(temp_clip_output_dir, source_video_name_without_ext, f"{source_video_name_without_ext}_clip_{i}.mp4")
        # The parser.py creates an intermediate directory with the video name.
        # Example: extracted_clips/test_channel/my_video_123/my_video_123_clip_0.mp4

        # Let's list files in temp_clip_output_dir/{source_video_name_without_ext}
        clip_source_subdir = os.path.join(temp_clip_output_dir, source_video_name_without_ext)
        if os.path.exists(clip_source_subdir):
            for f_name in os.listdir(clip_source_subdir):
                if f_name.startswith(f"{source_video_name_without_ext}_clip_") and f_name.endswith(".mp4"):
                     parsed_clip_files.append(os.path.join(clip_source_subdir, f_name))
        else:
            print(f"Clip source subdirectory {clip_source_subdir} not found.")

    if not parsed_clip_files:
        print(f"No clips found after parsing for {channel_name} in {clip_source_subdir}. Check parser output.")
        return
        
    parsed_clip_files.sort() # Ensure order if needed

    for i, original_clip_path in enumerate(parsed_clip_files):
        if not os.path.exists(original_clip_path):
            print(f"Parsed clip {original_clip_path} not found. Skipping.")
            continue

        clip_folder_name = f"clip_{i+1}"
        clip_specific_dir = os.path.join(channel_video_dir, clip_folder_name)
        os.makedirs(clip_specific_dir, exist_ok=True)
        
        # Move original clip to its final structured directory
        structured_original_clip_path = os.path.join(clip_specific_dir, "original.mp4")
        try:
            os.rename(original_clip_path, structured_original_clip_path)
            print(f"Moved {original_clip_path} to {structured_original_clip_path}")
        except Exception as e:
            print(f"Error moving {original_clip_path} to {structured_original_clip_path}: {e}")
            continue # Skip this clip if move fails

        current_clip_path = structured_original_clip_path

        # 3. Make clip vertical
        print(f"Transforming {current_clip_path} to vertical for {channel_name}/{clip_folder_name}...")
        # vertical_transformer.py input_video_path output_video_path
        vertical_clip_path = os.path.join(clip_specific_dir, "vertical.mp4")
        vertical_cmd = [
            "python", "src/vertical_transformer.py",
            current_clip_path,
            vertical_clip_path
        ]
        if not run_command(vertical_cmd):
            print(f"Failed to transform {current_clip_path} to vertical. Skipping further processing for this clip.")
            continue
        current_clip_path = vertical_clip_path # Update current path to the processed one

        # 4. Create subtitles for the clip
        print(f"Creating subtitles for {current_clip_path} for {channel_name}/{clip_folder_name}...")
        # subtitler.py input_video_path output_video_path whisper_model (optional)
        subtitled_clip_path = os.path.join(clip_specific_dir, "subtitled.mp4")
        whisper_model = global_config.get("whisper", {}).get("model", "base")
        subtitler_cmd = [
            "python", "src/subtitler.py",
            current_clip_path, # Input is the vertical clip
            subtitled_clip_path,
            "--model", whisper_model
        ]
        if not run_command(subtitler_cmd):
            print(f"Failed to create subtitles for {current_clip_path}. Skipping further processing for this clip.")
            # We might still want to caption it, or maybe not. For now, let's continue.
            # If subtitling fails, the captioner will use the vertical clip.
            pass # Continue to captioning even if subtitling fails
        else:
            current_clip_path = subtitled_clip_path # Update current path

        # 5. Create caption for the clip
        # video_captioner.py <video_path>
        # It prints to stdout. We need to save this to a file.
        print(f"Creating caption for {current_clip_path} for {channel_name}/{clip_folder_name}...")
        caption_output_file = os.path.join(clip_specific_dir, "caption.txt")
        
        # We need to modify video_captioner.py to accept an output file argument
        # or capture its stdout. Let's try to capture stdout.
        try:
            caption_process = subprocess.Popen(
                ["python", "src/video_captioner.py", current_clip_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            caption_stdout, caption_stderr = caption_process.communicate()
            if caption_process.returncode == 0:
                with open(caption_output_file, "w") as f:
                    f.write(caption_stdout.decode())
                print(f"Caption saved to {caption_output_file}")
            else:
                print(f"Error generating caption for {current_clip_path}:")
                print(f"STDOUT: {caption_stdout.decode()}")
                print(f"STDERR: {caption_stderr.decode()}")
        except Exception as e:
            print(f"Exception during caption generation for {current_clip_path}: {e}")

    print(f"Finished processing for channel: {channel_name}")


def main():
    # Load config
    config_path = "assets/config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)

    if not config or "channels" not in config:
        print("Error: 'channels' not found in config or config is empty.")
        sys.exit(1)

    channels_to_process = config.get("channels", {})
    global_settings = {
        "number_of_clips": config.get("number_of_clips"),
        "whisper": config.get("whisper")
    }

    # Create processes for each channel
    processes = []
    for channel_name, channel_data in channels_to_process.items():
        if isinstance(channel_data, dict) and "path" in channel_data:
            p = multiprocessing.Process(target=process_channel, args=(channel_name, channel_data, global_settings))
            processes.append(p)
            p.start()
        else:
            print(f"Skipping channel '{channel_name}': Invalid configuration data.")

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("All channels processed.")

if __name__ == "__main__":
    main() 