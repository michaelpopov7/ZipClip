#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse
from datetime import datetime

def list_formats(video_url):
    """List available formats for a video"""
    print(f"Listing available formats for: {video_url}")
    
    try:
        subprocess.run(["yt-dlp", "--list-formats", video_url])
        return True
    except Exception as e:
        print(f"Error listing formats: {e}")
        return False

def download_video(video_url, output_dir="./downloads", format_code=None):
    """Download a specific video using yt-dlp"""
    print(f"\nDownloading: {video_url}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the download command
    download_command = [
        "yt-dlp",
        video_url,
        "-o", f"{output_dir}/%(title)s-%(id)s.%(ext)s",
        "--write-info-json",
        "--write-thumbnail"
    ]
    
    # Add format specification only if explicitly provided
    if format_code:
        download_command.extend(["-f", format_code])
    
    # Execute the download
    try:
        result = subprocess.run(download_command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            print("You may want to try listing available formats with --list-formats")
            return False
            
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download a YouTube video")
    parser.add_argument("url", nargs="?", help="YouTube video URL")
    parser.add_argument("--list-formats", action="store_true", help="List available formats instead of downloading")
    parser.add_argument("--format", "-f", help="Specify format code to download (use --list-formats to see options)")
    
    args = parser.parse_args()
    
    # Get video URL from args or input
    video_url = args.url
    if not video_url:
        video_url = input("Enter YouTube video URL: ")
    
    if not video_url:
        print("No URL provided. Exiting.")
        return
    
    # Just list formats if requested
    if args.list_formats:
        list_formats(video_url)
        return
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./downloads/{timestamp}"
    
    # Download the video
    success = download_video(video_url, output_dir, args.format)
    
    if success:
        print(f"\nCompleted! Downloaded video to {output_dir}")
    else:
        print("\nDownload failed.")

if __name__ == "__main__":
    main() 