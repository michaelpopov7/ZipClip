# ZipClip 🎬

**AI-Powered Video Clip Extraction with Beautiful Web Interface**

ZipClip transforms long videos into engaging short clips using AI. Simply upload a video and describe what makes clips interesting - ZipClip will analyze the transcript with LLM intelligence and extract the best moments, complete with captions.

## ✨ Features

- **🎨 Beautiful Web Interface** - Modern, responsive design with drag & drop upload
- **🤖 AI-Powered Analysis** - Uses OpenAI or local LLMs to identify interesting moments
- **🎯 Custom Prompts** - Describe exactly what you want: "funny moments", "key insights", etc.
- **📱 Real-time Processing** - Live progress updates and status tracking
- **🎬 Professional Output** - Auto-generated captions and multiple export formats
- **🔒 Privacy Options** - Use local LLMs with Ollama for complete privacy
- **🐳 Dockerized** - One-command deployment with Docker

## 🚀 Quick Start (Docker - Recommended)

### Option 1: Web Interface Only
```bash
# Clone the repository
git clone <your-repo-url>
cd ZipClip

# Start with web interface (uses OpenAI API)
./start.sh up
```

### Option 2: Full Setup with Local LLM
```bash
# Start with local LLM support (complete privacy)
./start.sh up-full

# Download a local model (optional)
./start.sh ollama pull llama3.2
```

### 🌐 Access the Web Interface
Open http://localhost:5000 in your browser and start creating clips!

## 🐳 Docker Commands

```bash
./start.sh up          # Start web interface only
./start.sh up-full     # Start with local LLM support  
./start.sh down        # Stop all services
./start.sh logs        # View application logs
./start.sh build       # Rebuild Docker image
./start.sh clean       # Clean up containers and images

# Manage local LLM models
./start.sh ollama pull llama3.2    # Download model
./start.sh ollama list             # List models
./start.sh ollama rm <model>       # Remove model
```

## 💻 Manual Installation (Advanced)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg
```bash
# On macOS
brew install ffmpeg

# On Ubuntu/Debian  
sudo apt install ffmpeg

# On Windows
# Download from https://ffmpeg.org/download.html
```

### 3. Configure LLM Options

**Option A: OpenAI API (Cloud)**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Option B: Local LLM (Privacy-focused)**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download a model
ollama pull llama3.2

# Enable in config.yaml
local_llm:
  enabled: true
  model: "llama3.2"
```

### 4. Run the Application
```bash
# Web interface
python app.py

# Command line
python src/main_pipeline.py video.mp4 --prompt "funny moments"
```

## Usage

### Basic Usage
```bash
python src/main_pipeline.py path/to/your/video.mp4 --prompt "funny moments and highlights"
```

### Advanced Usage
```bash
python src/main_pipeline.py video.mp4 \
  --prompt "educational explanations and key insights" \
  --clips 8 \
  --output my_clips
```

### Parameters
- `video_path` - Path to your input video file
- `--prompt` - Describe what makes clips interesting (e.g., "funny moments", "key insights", "action scenes")
- `--clips` - Number of clips to extract (default: 5)
- `--output` - Output directory (default: "output")
- `--config` - Config file path (default: "assets/config.yaml")

## Example Prompts

- **Educational Content**: `"key explanations, important concepts, and learning moments"`
- **Entertainment**: `"funny moments, highlights, and engaging scenes"`  
- **Sports**: `"goals, great plays, and exciting moments"`
- **Gaming**: `"epic wins, funny fails, and highlight plays"`
- **Tutorials**: `"step-by-step instructions and important tips"`

## Output Structure

ZipClip creates organized output with timestamp information:

```
output/
  └── your_video_20240119_143022/
      ├── transcript.json     # Full transcript with timestamps
      ├── analysis.json       # AI analysis results
      ├── clip_1/
      │   ├── final_clip.mp4  # Finished clip with captions
      │   └── metadata.json   # Clip timing and reason
      ├── clip_2/
      │   ├── final_clip.mp4
      │   └── metadata.json
      └── ...
```

## ⚙️ Configuration

Edit `assets/config.yaml` to customize:

```yaml
whisper:
  model: base  # tiny, base, small, medium, large (larger = more accurate)

# LLM Configuration
# Option 1: OpenAI API (requires API key)
openai_api_key: ""
llm_model: "gpt-3.5-turbo"  # or "gpt-4" for better analysis

# Option 2: Local LLM via Ollama (privacy-focused, no API costs)
local_llm:
  enabled: false  # Set to true to use local LLM instead of OpenAI
  model: "llama3.2"  # Popular options: llama3.2, llama3.1, mistral, codellama

default_num_clips: 5
default_output_dir: "output"
```

## 🎯 Example Prompts

- **Educational Content**: `"key explanations, important concepts, and learning moments"`
- **Entertainment**: `"funny moments, highlights, and engaging scenes"`  
- **Sports**: `"goals, great plays, and exciting moments"`
- **Gaming**: `"epic wins, funny fails, and highlight plays"`
- **Business**: `"key insights, important decisions, and strategic moments"`
- **Tutorials**: `"step-by-step instructions and important tips"`

## 📁 Output Structure

ZipClip creates organized output with detailed metadata:

```
output/
  └── your_video_20240119_143022/
      ├── transcript.json     # Full transcript with timestamps
      ├── analysis.json       # AI analysis results
      ├── clip_1/
      │   ├── final_clip.mp4  # Finished clip with captions
      │   └── metadata.json   # Clip timing and reason
      ├── clip_2/
      │   ├── final_clip.mp4
      │   └── metadata.json
      └── ...
```

## 🔧 System Requirements

### Docker (Recommended)
- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM
- 10GB+ disk space

### Manual Installation
- Python 3.8+
- FFmpeg
- 4GB+ RAM
- CUDA (optional, for GPU acceleration)

## 🚧 Troubleshooting

### Docker Issues
```bash
# Check Docker status
docker --version
docker compose --version

# View logs
./start.sh logs

# Rebuild if needed
./start.sh build
```

### Local LLM Issues
```bash
# Check Ollama status
docker compose exec ollama ollama list

# Pull a model if none available
./start.sh ollama pull llama3.2
```

### Memory Issues
- Reduce Whisper model size in config (use 'tiny' or 'base')
- Limit video length to < 1 hour for processing
- Ensure 4GB+ RAM available

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🆘 Support

- Create an issue on GitHub
- Check the troubleshooting section
- Review Docker logs with `./start.sh logs`
