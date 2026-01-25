#!/bin/bash
# Clean up Docker resources
# Run this to free disk space from unused containers and images

echo "=== Docker Cleanup ==="
echo ""

# Show Docker disk usage
echo "Current Docker disk usage:"
docker system df 2>/dev/null || echo "  Cannot get Docker disk usage"
echo ""

# Check for Landseer containers
echo "Landseer containers:"
docker ps -a --filter "ancestor=ghcr.io/landseer-project" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}" 2>/dev/null || true
echo ""

# Stop running Landseer containers
RUNNING=$(docker ps -q --filter "ancestor=ghcr.io/landseer-project" 2>/dev/null)
if [ -n "$RUNNING" ]; then
    echo "Stopping running Landseer containers..."
    docker stop $RUNNING 2>/dev/null
fi

# Remove stopped containers
STOPPED=$(docker ps -aq --filter "status=exited" 2>/dev/null)
if [ -n "$STOPPED" ]; then
    STOPPED_COUNT=$(echo "$STOPPED" | wc -l)
    echo "Found $STOPPED_COUNT stopped container(s)"
    
    if [ "$1" == "-y" ] || [ "$1" == "--yes" ]; then
        REPLY="y"
    else
        read -p "Remove stopped containers? (y/n): " -n 1 -r
        echo
    fi
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rm $STOPPED 2>/dev/null
        echo "✓ Removed stopped containers"
    fi
fi

# Offer to prune dangling images
DANGLING=$(docker images -q --filter "dangling=true" 2>/dev/null)
if [ -n "$DANGLING" ]; then
    DANGLING_COUNT=$(echo "$DANGLING" | wc -l)
    echo ""
    echo "Found $DANGLING_COUNT dangling image(s)"
    
    if [ "$1" == "-y" ] || [ "$1" == "--yes" ]; then
        REPLY="y"
    else
        read -p "Remove dangling images? (y/n): " -n 1 -r
        echo
    fi
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker image prune -f
        echo "✓ Removed dangling images"
    fi
fi

# Offer full prune
echo ""
if [ "$1" == "--full" ]; then
    echo "Running full Docker prune..."
    docker system prune -f
    echo "✓ Docker system pruned"
else
    echo "Tip: Run with --full for complete Docker cleanup (docker system prune)"
fi

echo ""
echo "Current Docker disk usage:"
docker system df 2>/dev/null
