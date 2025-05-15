#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse
from datetime import datetime

def transform_to_vertical(input_file, output_dir="./transformed", width=1080, height=1920, background_color="black", blur_background=False, crop_mode=True):
    """
    Transform a video to vertical format with options for background handling.
    
    Args:
        input_file: Path to the input video file
        output_dir: Directory to save the output
        width: Width of the output video (default: 1080px)
        height: Height of the output video (default: 1920px)
        background_color: Color to use for padding (default: black)
        blur_background: Whether to use a blurred version of the video as background
        crop_mode: Whether to crop the video instead of scaling with padding
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    filename = os.path.basename(input_file)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/{name}_vertical_{timestamp}{ext}"
    
    print(f"Transforming '{input_file}' to vertical format...")
    
    # Get video information
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        input_file
    ]
    
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error getting video dimensions: {result.stderr}")
            return False
        
        original_width, original_height = map(int, result.stdout.strip().split(','))
        print(f"Original dimensions: {original_width}x{original_height}")
        
        # Build FFmpeg command based on the transformation mode
        if crop_mode:
            # Center crop the video
            # Calculate target aspect ratio
            target_aspect_ratio = width / height
            original_aspect_ratio = original_width / original_height
            
            if original_aspect_ratio > target_aspect_ratio:
                # Video is wider than target, need to crop width
                # Scale height to match target and crop sides
                crop_height = original_height
                crop_width = int(original_height * target_aspect_ratio)
                # Center the crop
                x_offset = (original_width - crop_width) // 2
                y_offset = 0
                
                print(f"Cropping sides: {original_width}x{original_height} -> {crop_width}x{crop_height}")
                
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i", input_file,
                    "-vf", f"crop={crop_width}:{crop_height}:{x_offset}:{y_offset},scale={width}:{height}",
                    "-c:a", "copy",
                    output_file
                ]
                
            else:
                # Video is taller than target or equal, need to crop height
                # Scale width to match target and crop top/bottom
                crop_width = original_width
                crop_height = int(original_width / target_aspect_ratio)
                # Center the crop
                x_offset = 0
                y_offset = (original_height - crop_height) // 2
                
                print(f"Cropping top/bottom: {original_width}x{original_height} -> {crop_width}x{crop_height}")
                
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i", input_file,
                    "-vf", f"crop={crop_width}:{crop_height}:{x_offset}:{y_offset},scale={width}:{height}",
                    "-c:a", "copy",
                    output_file
                ]
        
        else:
            # Scale with padding (original behavior)
            # Calculate scaling factors to fit within the target dimensions
            # while maintaining aspect ratio
            scale_factor = min(width / original_width, height / original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            print(f"Scaled dimensions: {new_width}x{new_height}")
            
            # Calculate padding
            x_offset = (width - new_width) // 2
            y_offset = (height - new_height) // 2
            
            # Build FFmpeg command based on background option
            if blur_background:
                # Version with blurred background
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i", input_file,
                    "-filter_complex", 
                    f"[0:v]split=2[original][bg];"
                    f"[bg]scale={width}:{height},boxblur=20:5[blurred];"
                    f"[original]scale={new_width}:{new_height}[scaled];"
                    f"[blurred][scaled]overlay=x={x_offset}:y={y_offset}[outv]",
                    "-map", "[outv]",
                    "-map", "0:a?",
                    "-c:a", "copy",
                    output_file
                ]
            else:
                # Version with solid color background
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i", input_file,
                    "-vf", f"scale={new_width}:{new_height},pad={width}:{height}:{x_offset}:{y_offset}:{background_color}",
                    "-c:a", "copy",
                    output_file
                ]
        
        print("Running FFmpeg command...")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        
        print(f"Successfully transformed video to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error transforming video: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Transform video to vertical format for social media")
    parser.add_argument("input", nargs="?", help="Input video file")
    parser.add_argument("--output-dir", "-o", default="./transformed", help="Output directory")
    parser.add_argument("--width", "-w", type=int, default=1080, help="Output width (default: 1080px)")
    parser.add_argument("--height", "-H", type=int, default=1920, help="Output height (default: 1920px)")
    parser.add_argument("--background", "-b", default="black", help="Background color (default: black)")
    parser.add_argument("--blur", action="store_true", help="Use blurred video as background")
    parser.add_argument("--scale", action="store_true", help="Scale with padding instead of cropping")
    
    args = parser.parse_args()
    
    # Get input file from args or input
    input_file = args.input
    if not input_file:
        input_file = input("Enter path to input video file: ")
    
    if not input_file:
        print("No input file provided. Exiting.")
        return
    
    # Transform the video
    success = transform_to_vertical(
        input_file, 
        args.output_dir, 
        args.width, 
        args.height, 
        args.background, 
        args.blur,
        not args.scale  # Default to crop mode unless --scale is specified
    )
    
    if not success:
        print("\nTransformation failed.")

if __name__ == "__main__":
    main() 