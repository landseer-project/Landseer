#!/bin/bash
# Landseer Pre-Run Cleanup Script
# This script stops all Landseer-related Docker containers and ensures GPUs are free

set -e  # Exit on error

echo "================================================"
echo "🧹 Landseer Pre-Run Cleanup"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
if ! command_exists docker; then
    echo -e "${RED}❌ Docker not found. Please install Docker.${NC}"
    exit 1
fi

if ! command_exists nvidia-smi; then
    echo -e "${YELLOW}⚠️  nvidia-smi not found. Skipping GPU checks.${NC}"
    GPU_AVAILABLE=false
else
    GPU_AVAILABLE=true
fi

echo -e "${BLUE}Step 1: Checking for running Docker containers${NC}"
echo "----------------------------------------------"

# Get all running containers
RUNNING_CONTAINERS=$(docker ps -q 2>/dev/null || true)

if [ -z "$RUNNING_CONTAINERS" ]; then
    echo -e "${GREEN}✓ No running containers found${NC}"
else
    CONTAINER_COUNT=$(echo "$RUNNING_CONTAINERS" | wc -l)
    echo -e "${YELLOW}Found $CONTAINER_COUNT running container(s)${NC}"
    
    # Show container details
    echo ""
    echo "Container details:"
    docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}"
    echo ""
    
    # Filter Landseer-related containers (by common image patterns)
    LANDSEER_CONTAINERS=$(docker ps --filter "ancestor=ghcr.io/landseer-project/*" -q 2>/dev/null || true)
    
    # Also check for containers with landseer in the name
    LANDSEER_NAMED=$(docker ps --filter "name=landseer" -q 2>/dev/null || true)
    
    # Combine and deduplicate
    ALL_LANDSEER=$(echo -e "$LANDSEER_CONTAINERS\n$LANDSEER_NAMED" | sort -u | grep -v '^$' || true)
    
    if [ -n "$ALL_LANDSEER" ]; then
        LANDSEER_COUNT=$(echo "$ALL_LANDSEER" | wc -l)
        echo -e "${YELLOW}📦 Found $LANDSEER_COUNT Landseer-related container(s)${NC}"
        
        # Prompt for confirmation
        read -p "Stop these containers? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping Landseer containers..."
            for container_id in $ALL_LANDSEER; do
                CONTAINER_NAME=$(docker ps --filter "id=$container_id" --format "{{.Names}}")
                echo -e "  Stopping ${BLUE}$container_id${NC} ($CONTAINER_NAME)..."
                docker stop "$container_id" 2>/dev/null || true
                docker rm "$container_id" 2>/dev/null || true
            done
            echo -e "${GREEN}✓ Stopped Landseer containers${NC}"
        else
            echo -e "${YELLOW}⚠️  Skipped stopping containers${NC}"
        fi
    fi
    
    # Ask about stopping ALL containers
    REMAINING=$(docker ps -q 2>/dev/null | wc -l)
    if [ "$REMAINING" -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}⚠️  $REMAINING container(s) still running${NC}"
        read -p "Stop ALL running containers? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping all containers..."
            docker stop $(docker ps -q) 2>/dev/null || true
            docker rm $(docker ps -aq) 2>/dev/null || true
            echo -e "${GREEN}✓ Stopped all containers${NC}"
        fi
    fi
fi

echo ""
echo -e "${BLUE}Step 2: Checking GPU status${NC}"
echo "----------------------------------------------"

if [ "$GPU_AVAILABLE" = true ]; then
    # Check for processes using GPU
    GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
    
    if [ -z "$GPU_PROCESSES" ]; then
        echo -e "${GREEN}✓ No processes using GPU${NC}"
    else
        echo -e "${YELLOW}⚠️  Found processes using GPU:${NC}"
        echo ""
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=table
        echo ""
        
        # Try to identify if they're Docker-related
        DOCKER_PIDS=$(pgrep -f "docker" || true)
        
        read -p "Kill GPU processes? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            while IFS= read -r line; do
                PID=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
                PROCESS=$(echo "$line" | cut -d',' -f2)
                if [ -n "$PID" ] && [ "$PID" != "PID" ]; then
                    echo -e "  Killing PID ${BLUE}$PID${NC} ($PROCESS)..."
                    kill -9 "$PID" 2>/dev/null || true
                fi
            done <<< "$GPU_PROCESSES"
            
            echo -e "${GREEN}✓ Killed GPU processes${NC}"
            
            # Wait a moment for cleanup
            sleep 2
            
            # Verify
            REMAINING_GPU=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
            if [ "$REMAINING_GPU" -eq 0 ]; then
                echo -e "${GREEN}✓ GPUs are now free${NC}"
            else
                echo -e "${YELLOW}⚠️  Some GPU processes still remain${NC}"
            fi
        fi
    fi
    
    # Show GPU status
    echo ""
    echo "Current GPU status:"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=table
    
else
    echo -e "${YELLOW}⚠️  GPU tools not available, skipping GPU checks${NC}"
fi

echo ""
echo -e "${BLUE}Step 3: Cleanup temporary files${NC}"
echo "----------------------------------------------"

# Clean up common Landseer temp directories
CLEANED=0

if [ -d "results" ]; then
    # Find and optionally clean old staged_inputs
    OLD_STAGED=$(find results -type d -name "staged_inputs" 2>/dev/null | wc -l)
    if [ "$OLD_STAGED" -gt 0 ]; then
        echo -e "${YELLOW}Found $OLD_STAGED staged_inputs directories${NC}"
        read -p "Clean up old staged_inputs? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            find results -type d -name "staged_inputs" -exec rm -rf {} + 2>/dev/null || true
            CLEANED=$((CLEANED + OLD_STAGED))
            echo -e "${GREEN}✓ Cleaned staged_inputs${NC}"
        fi
    fi
fi

if [ -d "/tmp" ]; then
    TEMP_LANDSEER=$(find /tmp -maxdepth 1 -name "*landseer*" -o -name "*defense*" 2>/dev/null | wc -l)
    if [ "$TEMP_LANDSEER" -gt 0 ]; then
        echo -e "${YELLOW}Found $TEMP_LANDSEER temporary Landseer files in /tmp${NC}"
        read -p "Clean up /tmp files? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            find /tmp -maxdepth 1 -name "*landseer*" -exec rm -rf {} + 2>/dev/null || true
            find /tmp -maxdepth 1 -name "*defense*" -exec rm -rf {} + 2>/dev/null || true
            CLEANED=$((CLEANED + TEMP_LANDSEER))
            echo -e "${GREEN}✓ Cleaned /tmp files${NC}"
        fi
    fi
fi

if [ "$CLEANED" -eq 0 ]; then
    echo -e "${GREEN}✓ No temporary files to clean${NC}"
fi

echo ""
echo -e "${BLUE}Step 4: System resource check${NC}"
echo "----------------------------------------------"

# Check disk space
DISK_USAGE=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo -e "${RED}⚠️  Disk usage: ${DISK_USAGE}% - Consider freeing space${NC}"
elif [ "$DISK_USAGE" -gt 80 ]; then
    echo -e "${YELLOW}⚠️  Disk usage: ${DISK_USAGE}%${NC}"
else
    echo -e "${GREEN}✓ Disk usage: ${DISK_USAGE}%${NC}"
fi

# Check memory
MEMORY_USAGE=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
if [ "$MEMORY_USAGE" -gt 90 ]; then
    echo -e "${YELLOW}⚠️  Memory usage: ${MEMORY_USAGE}%${NC}"
else
    echo -e "${GREEN}✓ Memory usage: ${MEMORY_USAGE}%${NC}"
fi

# Check cache size
if [ -d "cache/artifact_store" ]; then
    CACHE_SIZE=$(du -sh cache/artifact_store 2>/dev/null | cut -f1)
    CACHE_COUNT=$(find cache/artifact_store -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo -e "${BLUE}ℹ  Artifact cache: $CACHE_SIZE ($CACHE_COUNT entries)${NC}"
    
    if [ "$CACHE_COUNT" -gt 100 ]; then
        echo -e "${YELLOW}⚠️  Large cache - consider running with --no-cache if testing${NC}"
    fi
fi

echo ""
echo "================================================"
echo -e "${GREEN}✅ Cleanup Complete!${NC}"
echo "================================================"
echo ""
echo "You can now safely run Landseer:"
echo -e "${BLUE}  poetry run python -m landseer_pipeline.main -c <config> -a <attack>${NC}"
echo ""

# Optional: Auto-run check
if [ "$1" == "--auto-run" ] && [ -n "$2" ] && [ -n "$3" ]; then
    echo "Auto-running Landseer with provided config..."
    echo ""
    poetry run python -m landseer_pipeline.main -c "$2" -a "$3"
fi
