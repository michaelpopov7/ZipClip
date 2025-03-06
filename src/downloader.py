import argparse
import os
import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
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

def download_youtube_video(url, output_path="downloads", resolution="best"):
    """
    Download a YouTube video using yt-dlp.
    
    Args:
        url (str): YouTube video URL
        output_path (str): Directory to save the video
        resolution (str): Video quality option (best, worst, or specific like 720)
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Map resolution options
    if resolution == "highest":
        resolution = "best"
    elif resolution == "lowest":
        resolution = "worst"
    
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
    
    # Build the command
    output_template = os.path.join(output_path, "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        url,
        "-f", format_option,
        "-o", output_template,
        "--no-playlist",  # Don't download playlists
        "--progress"      # Show progress
    ]
    
    # Run the download command
    try:
        logger.info(f"Starting download from: {url}")
        logger.info(f"Quality setting: {resolution}")
        
        process = subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        # Extract the output filename from the process output
        output = process.stdout
        for line in output.split('\n'):
            if "Destination" in line and ":" in line:
                file_path = line.split("Destination: ", 1)[1].strip()
                logger.info(f"Download complete! File saved to: {file_path}")
                return True
        
        logger.info("Download complete!")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        if e.stderr:
            logger.error(f"Error details: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download YouTube videos using yt-dlp")
    parser.add_argument("--url", type=str, help="YouTube video URL")
    parser.add_argument("--output", type=str, default="downloads", help="Output directory")
    parser.add_argument("--resolution", type=str, default="best", 
                       help="Video resolution (highest/best, lowest/worst, or specific like 720)")
    
    args = parser.parse_args()
    
    # Check if yt-dlp is installed
    if not check_dependencies():
        return
    
    # If URL not provided as argument, prompt for it
    url = args.url
    if not url:
        url = input("Enter YouTube video URL: ")
    
    # Download the video
    download_youtube_video(url, args.output, args.resolution)

if __name__ == "__main__":
    main()