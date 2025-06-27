#!/bin/bash

# ZipClip Docker Startup Script
# This script provides easy commands to run ZipClip with Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Banner
echo -e "${PURPLE}"
echo "  ███████╗██╗██████╗  ██████╗██╗     ██╗██████╗ "
echo "  ╚══███╔╝██║██╔══██╗██╔════╝██║     ██║██╔══██╗"
echo "    ███╔╝ ██║██████╔╝██║     ██║     ██║██████╔╝"
echo "   ███╔╝  ██║██╔═══╝ ██║     ██║     ██║██╔═══╝ "
echo "  ███████╗██║██║     ╚██████╗███████╗██║██║     "
echo "  ╚══════╝╚═╝╚═╝      ╚═════╝╚══════╝╚═╝╚═╝     "
echo -e "${NC}"
echo -e "${BLUE}AI-Powered Video Clip Extraction${NC}"
echo ""

# Function definitions
show_help() {
    echo -e "${GREEN}Usage: ./start.sh [COMMAND]${NC}"
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo "  up          Start ZipClip (web interface only)"
    echo "  up-full     Start ZipClip with local LLM support (Ollama)"
    echo "  down        Stop ZipClip"
    echo "  logs        Show application logs"
    echo "  build       Build/rebuild the Docker image"
    echo "  clean       Clean up containers and images"
    echo "  ollama      Manage Ollama models"
    echo "  help        Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./start.sh up              # Start web interface"
    echo "  ./start.sh up-full         # Start with local LLM"
    echo "  ./start.sh ollama pull llama3.2  # Download a model"
    echo ""
}

start_basic() {
    echo -e "${GREEN}🚀 Starting ZipClip (web interface only)...${NC}"
    docker compose up -d zipclip
    echo ""
    echo -e "${GREEN}✅ ZipClip is running!${NC}"
    echo -e "${BLUE}🌐 Web interface: http://localhost:5000${NC}"
    echo -e "${YELLOW}💡 Use OpenAI API key for AI analysis${NC}"
    echo ""
}

start_full() {
    echo -e "${GREEN}🚀 Starting ZipClip with local LLM support...${NC}"
    docker compose up -d
    echo ""
    echo -e "${GREEN}✅ ZipClip is running with Ollama!${NC}"
    echo -e "${BLUE}🌐 Web interface: http://localhost:5000${NC}"
    echo -e "${BLUE}🤖 Ollama API: http://localhost:11434${NC}"
    echo ""
    echo -e "${YELLOW}💡 To use local LLM:${NC}"
    echo "1. Download a model: ./start.sh ollama pull llama3.2"
    echo "2. Enable in config: assets/config.yaml (set local_llm.enabled: true)"
    echo ""
}

stop_services() {
    echo -e "${YELLOW}🛑 Stopping ZipClip...${NC}"
    docker compose down
    echo -e "${GREEN}✅ ZipClip stopped${NC}"
}

show_logs() {
    echo -e "${BLUE}📜 Showing ZipClip logs...${NC}"
    docker compose logs -f zipclip
}

build_image() {
    echo -e "${BLUE}🔨 Building ZipClip Docker image...${NC}"
    docker compose build --no-cache
    echo -e "${GREEN}✅ Build complete${NC}"
}

clean_up() {
    echo -e "${YELLOW}🧹 Cleaning up Docker resources...${NC}"
    docker compose down -v --rmi all --remove-orphans
    docker system prune -f
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

manage_ollama() {
    case "$2" in
        "pull")
            if [ -z "$3" ]; then
                echo -e "${RED}❌ Please specify a model to pull${NC}"
                echo "Example: ./start.sh ollama pull llama3.2"
                echo ""
                echo -e "${YELLOW}Popular models:${NC}"
                echo "  llama3.2    - Latest Llama model (recommended)"
                echo "  llama3.1    - Previous Llama version"
                echo "  mistral     - Mistral 7B model"
                echo "  codellama   - Code-focused model"
                exit 1
            fi
            echo -e "${BLUE}📥 Downloading model: $3${NC}"
            docker compose exec ollama ollama pull "$3"
            ;;
        "list")
            echo -e "${BLUE}📋 Available models:${NC}"
            docker compose exec ollama ollama list
            ;;
        "rm")
            if [ -z "$3" ]; then
                echo -e "${RED}❌ Please specify a model to remove${NC}"
                exit 1
            fi
            echo -e "${YELLOW}🗑️  Removing model: $3${NC}"
            docker compose exec ollama ollama rm "$3"
            ;;
        *)
            echo -e "${GREEN}Ollama Management Commands:${NC}"
            echo "  ./start.sh ollama pull <model>  # Download a model"
            echo "  ./start.sh ollama list          # List downloaded models"
            echo "  ./start.sh ollama rm <model>    # Remove a model"
            echo ""
            echo -e "${YELLOW}Popular models:${NC}"
            echo "  llama3.2, llama3.1, mistral, codellama"
            ;;
    esac
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Parse command
case "$1" in
    "up")
        start_basic
        ;;
    "up-full")
        start_full
        ;;
    "down")
        stop_services
        ;;
    "logs")
        show_logs
        ;;
    "build")
        build_image
        ;;
    "clean")
        clean_up
        ;;
    "ollama")
        manage_ollama "$@"
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    "")
        show_help
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
