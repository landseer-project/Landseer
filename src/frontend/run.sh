#!/bin/bash
# Landseer Frontend Runner
# Uses local Node.js installation (no sudo required)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NODE_DIR="$PROJECT_ROOT/.node"

# Add local Node.js to PATH
export PATH="$NODE_DIR/bin:$PATH"

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "Error: Node.js not found in $NODE_DIR"
    echo "Run: cd $PROJECT_ROOT && mkdir -p .node && cd .node && curl -L https://nodejs.org/dist/v20.18.0/node-v20.18.0-linux-x64.tar.xz | tar -xJ --strip-components=1"
    exit 1
fi

cd "$SCRIPT_DIR"

case "${1:-dev}" in
    install)
        echo "Installing dependencies..."
        npm install
        ;;
    dev)
        echo "Starting development server at http://localhost:3000"
        echo "Make sure the backend API is running at http://localhost:8000"
        npm run dev
        ;;
    build)
        echo "Building for production..."
        npm run build
        ;;
    preview)
        echo "Starting preview server..."
        npm run preview
        ;;
    *)
        echo "Landseer Frontend"
        echo ""
        echo "Usage: ./run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  install   Install dependencies"
        echo "  dev       Start development server (default)"
        echo "  build     Build for production"
        echo "  preview   Preview production build"
        ;;
esac
