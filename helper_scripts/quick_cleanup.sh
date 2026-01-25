#!/bin/bash
# Quick non-interactive cleanup - clean workspaces and cache without prompts
# Use this for fast cleanup between test runs

echo "=== Quick Cleanup ==="

# Clean workspaces
echo -n "Cleaning workspaces... "
rm -rf /tmp/landseer_worker_* 2>/dev/null
echo "done"

# Clean cache (keep the directory)
echo -n "Cleaning cache... "
rm -rf /tmp/landseer_cache/* 2>/dev/null
echo "done"

# Remove old stopped containers
echo -n "Cleaning Docker containers... "
docker rm $(docker ps -aq --filter "status=exited" 2>/dev/null) 2>/dev/null || true
echo "done"

echo ""
echo "✓ Quick cleanup complete"

# Show remaining disk usage
df -h /tmp | head -2
