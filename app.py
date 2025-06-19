#!/usr/bin/env python3
"""
ZipClip Web Interface
Beautiful, modern web interface for AI-powered video clip extraction
"""

import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import threading
import time

from src.main_pipeline import ZipClipPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'zipclip-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v'}

# Ensure directories exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, 'static', 'templates']:
    os.makedirs(folder, exist_ok=True)

# Global storage for job status
job_status = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class JobManager:
    """Manages background video processing jobs"""
    
    def __init__(self):
        self.jobs = {}
    
    def create_job(self, job_id: str, video_path: str, prompt: str, num_clips: int = 5):
        """Create a new processing job"""
        self.jobs[job_id] = {
            'id': job_id,
            'status': 'queued',
            'progress': 0,
            'message': 'Job queued for processing',
            'video_path': video_path,
            'prompt': prompt,
            'num_clips': num_clips,
            'created_at': datetime.now().isoformat(),
            'clips': [],
            'error': None
        }
        
        # Start processing in background thread
        thread = threading.Thread(target=self._process_job, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        return job_id
    
    def _process_job(self, job_id: str):
        """Process a video job in background"""
        try:
            job = self.jobs[job_id]
            
            # Update status
            job['status'] = 'processing'
            job['progress'] = 10
            job['message'] = 'Initializing pipeline...'
            
            # Initialize pipeline
            pipeline = ZipClipPipeline()
            
            # Extract transcript
            job['progress'] = 25
            job['message'] = 'Extracting transcript...'
            
            transcript_result = pipeline.transcript_extractor.extract_transcript(job['video_path'])
            if not transcript_result:
                job['status'] = 'failed'
                job['error'] = 'Failed to extract transcript'
                return
            
            # Analyze with AI
            job['progress'] = 50
            job['message'] = 'Analyzing video with AI...'
            
            clip_info_list = pipeline.clip_analyzer.analyze_transcript(
                transcript_result, job['prompt'], job['num_clips']
            )
            
            if not clip_info_list:
                job['status'] = 'failed'
                job['error'] = 'Failed to identify interesting moments'
                return
            
            # Extract clips
            job['progress'] = 75
            job['message'] = 'Extracting video clips...'
            
            # Create job-specific output directory
            output_dir = os.path.join(OUTPUT_FOLDER, job_id)
            extracted_clips = pipeline.clip_extractor.extract_clips(
                job['video_path'], clip_info_list, output_dir
            )
            
            if not extracted_clips:
                job['status'] = 'failed'
                job['error'] = 'Failed to extract clips'
                return
            
            # Add captions
            job['progress'] = 90
            job['message'] = 'Adding captions...'
            
            processed_clips = []
            for i, clip_path in enumerate(extracted_clips):
                clip_dir = os.path.dirname(clip_path)
                captioned_path = os.path.join(clip_dir, "final_clip.mp4")
                
                if pipeline.captioner.add_captions(clip_path, captioned_path):
                    # Remove original uncaptioned clip
                    try:
                        os.remove(clip_path)
                    except:
                        pass
                    
                    # Store clip info
                    clip_info = {
                        'id': i + 1,
                        'path': captioned_path,
                        'reason': clip_info_list[i].get('reason', f'Clip {i+1}'),
                        'start_time': clip_info_list[i]['start_time'],
                        'end_time': clip_info_list[i]['end_time'],
                        'duration': clip_info_list[i]['end_time'] - clip_info_list[i]['start_time']
                    }
                    processed_clips.append(clip_info)
                else:
                    # Keep original clip if captioning failed
                    clip_info = {
                        'id': i + 1,
                        'path': clip_path,
                        'reason': clip_info_list[i].get('reason', f'Clip {i+1}'),
                        'start_time': clip_info_list[i]['start_time'],
                        'end_time': clip_info_list[i]['end_time'],
                        'duration': clip_info_list[i]['end_time'] - clip_info_list[i]['start_time']
                    }
                    processed_clips.append(clip_info)
            
            # Job completed successfully
            job['status'] = 'completed'
            job['progress'] = 100
            job['message'] = f'Successfully extracted {len(processed_clips)} clips!'
            job['clips'] = processed_clips
            job['completed_at'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            job['status'] = 'failed'
            job['error'] = str(e)
    
    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        return self.jobs.get(job_id, {})
    
    def list_jobs(self) -> Dict[str, Any]:
        """List all jobs"""
        return self.jobs

# Initialize job manager
job_manager = JobManager()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        prompt = request.form.get('prompt', '').strip()
        num_clips = int(request.form.get('num_clips', 5))
        
        if not prompt:
            return jsonify({'error': 'Please provide a prompt describing what makes clips interesting'}), 400
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported. Please upload MP4, AVI, MOV, MKV, WebM, or M4V files.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
        file.save(video_path)
        
        # Create processing job
        job_manager.create_job(job_id, video_path, prompt, num_clips)
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Video uploaded successfully. Processing started...'
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Failed to upload video'}), 500

@app.route('/status/<job_id>')
def job_status_endpoint(job_id):
    """Get job status"""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(job)

@app.route('/download/<job_id>/<int:clip_id>')
def download_clip(job_id, clip_id):
    """Download individual clip"""
    job = job_manager.get_job(job_id)
    if not job or job['status'] != 'completed':
        return jsonify({'error': 'Job not found or not completed'}), 404
    
    # Find the clip
    clip = None
    for c in job['clips']:
        if c['id'] == clip_id:
            clip = c
            break
    
    if not clip:
        return jsonify({'error': 'Clip not found'}), 404
    
    if not os.path.exists(clip['path']):
        return jsonify({'error': 'Clip file not found'}), 404
    
    return send_file(clip['path'], as_attachment=True, download_name=f'clip_{clip_id}.mp4')

@app.route('/preview/<job_id>/<int:clip_id>')
def preview_clip(job_id, clip_id):
    """Stream clip for preview"""
    job = job_manager.get_job(job_id)
    if not job or job['status'] != 'completed':
        return jsonify({'error': 'Job not found or not completed'}), 404
    
    # Find the clip
    clip = None
    for c in job['clips']:
        if c['id'] == clip_id:
            clip = c
            break
    
    if not clip:
        return jsonify({'error': 'Clip not found'}), 404
    
    if not os.path.exists(clip['path']):
        return jsonify({'error': 'Clip file not found'}), 404
    
    return send_file(clip['path'])

@app.route('/jobs')
def list_jobs():
    """List all jobs"""
    jobs = job_manager.list_jobs()
    return jsonify(jobs)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)