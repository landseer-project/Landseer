#!/bin/bash
# Quick disk usage check for Landseer-related files

echo "=== Landseer Disk Usage ==="
echo ""

# Project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Project directory: $PROJECT_DIR"
du -sh "$PROJECT_DIR" 2>/dev/null
echo ""

# Worker workspaces
echo "Worker workspaces (/tmp/landseer_worker_*):"
if ls /tmp/landseer_worker_* 1>/dev/null 2>&1; then
    du -sh /tmp/landseer_worker_* 2>/dev/null
    TOTAL_WORKSPACE=$(du -sc /tmp/landseer_worker_* 2>/dev/null | tail -1 | cut -f1)
    echo "  Total: $(numfmt --to=iec $((TOTAL_WORKSPACE * 1024)) 2>/dev/null || echo "${TOTAL_WORKSPACE}K")"
else
    echo "  (none)"
fi
echo ""

# Cache
CACHE_DIR="${LANDSEER_CACHE_DIR:-/tmp/landseer_cache}"
echo "Cache ($CACHE_DIR):"
if [ -d "$CACHE_DIR" ]; then
    du -sh "$CACHE_DIR" 2>/dev/null
else
    echo "  (none)"
fi
echo ""

# Database
echo "Database:"
if [ -f "$PROJECT_DIR/landseer.db" ]; then
    du -sh "$PROJECT_DIR/landseer.db"
else
    echo "  (none)"
fi
echo ""

# MinIO data
if [ -d "$PROJECT_DIR/data/minio" ]; then
    echo "MinIO data ($PROJECT_DIR/data/minio):"
    du -sh "$PROJECT_DIR/data/minio" 2>/dev/null
    echo ""
fi

# Docker
echo "Docker:"
docker system df 2>/dev/null || echo "  (unable to check)"
echo ""

# Disk space summary
echo "=== Disk Space Summary ==="
df -h "$PROJECT_DIR" | head -2
echo ""

# Recommendations
DISK_PCT=$(df "$PROJECT_DIR" | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_PCT" -gt 90 ]; then
    echo "⚠️  WARNING: Disk usage is at ${DISK_PCT}%"
    echo "Run: ./cleanup_workspaces.sh -y && ./cleanup_cache.sh -y"
elif [ "$DISK_PCT" -gt 80 ]; then
    echo "⚠️  Note: Disk usage is at ${DISK_PCT}%"
fi
