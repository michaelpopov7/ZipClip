#!/usr/bin/env python3
import subprocess
import json
import random
import os
import argparse
from datetime import datetime

def get_trending_videos():
    """Get a list of trending videos from YouTube"""
    print("Fetching trending videos...")
    
    try:
        # Get trending videos
        result = subprocess.run(
            ["yt-dlp", "--flat-playlist", "--dump-single-json", "https://www.youtube.com/feed/trending"],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print(f"Error fetching trending videos: {result.stderr}")
            return []
        
        try:
            data = json.loads(result.stdout)
            if 'entries' in data:
                return [entry['url'] for entry in data['entries']]
            return []
        except json.JSONDecodeError:
            print("Failed to parse YouTube response")
            return []
            
    except Exception as e:
        print(f"Error: {e}")
        return []

def check_video_duration(video_url):
    """Check if a video's duration is between 1-2 hours"""
    try:
        result = subprocess.run(
            ["yt-dlp", "--skip-download", "--print", "duration", video_url],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return None
        
        # Get duration in seconds
        duration = int(result.stdout.strip())
        return duration
        
    except Exception:
        return None

def get_video_info(video_url):
    """Get detailed info about a video"""
    try:
        result = subprocess.run(
            ["yt-dlp", "--skip-download", "--dump-json", video_url],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return None
        
        return json.loads(result.stdout)
        
    except Exception:
        return None

def find_videos_in_range(min_duration=3600, max_duration=7200, limit=30):
    """Find videos that match the duration criteria"""
    print(f"Looking for videos between {min_duration/3600:.1f}-{max_duration/3600:.1f} hours...")
    
    # Start with trending videos and some popular channels/playlists
    sources = [
        "https://www.youtube.com/feed/trending",
        "https://www.youtube.com/playlist?list=PLbpi6ZahtOH7vgyGImZ4P-olTT11WLkLk",  # Popular uploads
        "https://www.youtube.com/watch?v=jfKfPfyJRdk",  # Popular lofi stream (likely to have recommendations)
        "https://www.youtube.com/results?search_query=documentary+full+length",
        "https://www.youtube.com/results?search_query=podcast+full+episode"
    ]
    
    # Get videos from each source
    all_video_ids = set()
    matching_videos = []
    
    for source in sources:
        try:
            print(f"Checking source: {source}")
            # Get video IDs from the source
            result = subprocess.run(
                ["yt-dlp", "--flat-playlist", "--get-id", source],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                video_ids = result.stdout.strip().split('\n')
                for vid in video_ids:
                    if vid and vid not in all_video_ids:
                        all_video_ids.add(vid)
                        
                        # Check duration for each video
                        video_url = f"https://www.youtube.com/watch?v={vid}"
                        duration = check_video_duration(video_url)
                        
                        if duration and min_duration <= duration <= max_duration:
                            info = get_video_info(video_url)
                            if info:
                                matching_videos.append(info)
                                print(f"Found matching video: {info['title']} ({duration/60:.1f} minutes)")
                        
                        # Stop if we've found enough videos
                        if len(matching_videos) >= limit:
                            break
            
            # Stop if we've found enough videos
            if len(matching_videos) >= limit:
                break
                
        except Exception as e:
            print(f"Error processing source {source}: {e}")
    
    print(f"Found {len(matching_videos)} videos matching duration criteria")
    return matching_videos

def download_video(video_info, output_dir="."):
    """Download a specific video using yt-dlp"""
    video_id = video_info['id']
    title = video_info['title']
    duration_mins = video_info['duration'] / 60
    
    print(f"\nDownloading: {title}")
    print(f"Duration: {duration_mins:.1f} minutes")
    print(f"Video ID: {video_id}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the download command
    download_command = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", f"{output_dir}/%(title)s-%(id)s.%(ext)s",
        "--write-info-json",
        "--write-thumbnail"
    ]
    
    # Execute the download
    try:
        subprocess.run(download_command)
        return True
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download random YouTube videos between 1-2 hours in length")
    parser.add_argument("--count", type=int, default=1, help="Number of videos to download")
    parser.add_argument("--output", type=str, default="./downloads", help="Output directory")
    parser.add_argument("--min-hours", type=float, default=1.0, help="Minimum video length in hours")
    parser.add_argument("--max-hours", type=float, default=2.0, help="Maximum video length in hours")
    
    args = parser.parse_args()
    
    # Convert hours to seconds
    min_duration = int(args.min_hours * 3600)
    max_duration = int(args.max_hours * 3600)
    
    # Find videos in the specified range
    videos = find_videos_in_range(min_duration, max_duration)
    
    if not videos:
        print("No matching videos found. Try again later or modify the duration range.")
        return
    
    # Pick random videos to download
    to_download = min(args.count, len(videos))
    selected_videos = random.sample(videos, to_download)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}/{timestamp}"
    
    # Download each selected video
    for i, video in enumerate(selected_videos, 1):
        print(f"\n[{i}/{to_download}] Downloading video...")
        download_video(video, output_dir)
    
    print(f"\nCompleted! Downloaded {to_download} videos to {output_dir}")

if __name__ == "__main__":
    main()