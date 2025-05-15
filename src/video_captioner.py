#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
from transformers import BlipForConditionalGeneration

def extract_frames(video_path, num_frames=8):
    """Extract evenly spaced frames from a video"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
    
    cap.release()
    return frames

def generate_caption(frames):
    """Generate a caption for the video frames using BLIP model"""
    # Load BLIP model for image captioning
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Process frames and generate captions
    captions = []
    for frame in frames:
        inputs = processor(frame, return_tensors="pt")
        output = model.generate(**inputs, max_length=50)
        caption = processor.decode(output[0], skip_special_tokens=True)
        captions.append(caption)
    
    # Combine captions into a 2-3 sentence summary
    if len(captions) > 0:
        # Use the two longest captions as they might be the most informative
        sorted_captions = sorted(captions, key=len, reverse=True)[:2]
        summary = " ".join(sorted_captions)
        return summary
    else:
        return "Could not generate a caption for this video."

def generate_hashtags(caption, num_tags=15):
    """Generate relevant hashtags based on the caption"""
    # Use a text classification pipeline for zero-shot classification
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    # Potential categories/topics for hashtags
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
    
    # Combine and ensure we have 15 hashtags
    all_hashtags = selected_hashtags + common_hashtags
    final_hashtags = list(dict.fromkeys(all_hashtags))[:num_tags]  # Remove duplicates and limit to 15
    
    return final_hashtags

def main():
    parser = argparse.ArgumentParser(description="Generate AI caption and hashtags for a video")
    parser.add_argument("video_path", help="Path to the video file")
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        sys.exit(1)
    
    print(f"Processing video: {args.video_path}")
    
    # Extract frames from the video
    frames = extract_frames(args.video_path)
    
    # Generate caption
    caption = generate_caption(frames)
    
    # Generate hashtags
    hashtags = generate_hashtags(caption)
    hashtag_str = " ".join([f"#{tag}" for tag in hashtags])
    
    # Print the result
    print("\n--- Generated Caption ---")
    print(caption)
    print("\n--- Generated Hashtags ---")
    print(hashtag_str)
    print("\n--- Combined Output ---")
    print(f"{caption}\n\n{hashtag_str}")

if __name__ == "__main__":
    main() 