#!/bin/bash
# Reset everything for a fresh pipeline run
# WARNING: This deletes all state - use carefully!

echo "=== Landseer Full Reset ==="
echo ""
echo "This will delete:"
echo "  - Worker workspaces (/tmp/landseer_worker_*)"
echo "  - Artifact cache (/tmp/landseer_cache)"
echo "  - SQLite database (landseer.db)"
echo "  - Stopped Docker containers"
echo ""

if [ "$1" != "-y" ] && [ "$1" != "--yes" ]; then
    read -p "Are you sure you want to reset everything? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 0
    fi
fi

echo ""
echo "Cleaning worker workspaces..."
rm -rf /tmp/landseer_worker_* 2>/dev/null && echo "  ✓ Workspaces cleaned" || echo "  - No workspaces found"

echo "Cleaning cache..."
rm -rf /tmp/landseer_cache/* 2>/dev/null && echo "  ✓ Cache cleaned" || echo "  - No cache found"

echo "Removing database..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJECT_DIR/landseer.db" ]; then
    rm -f "$PROJECT_DIR/landseer.db"
    echo "  ✓ Database removed"
else
    echo "  - No database found"
fi

echo "Cleaning Docker..."
docker stop $(docker ps -q --filter "ancestor=ghcr.io/landseer-project" 2>/dev/null) 2>/dev/null || true
docker rm $(docker ps -aq --filter "status=exited" 2>/dev/null) 2>/dev/null || true
echo "  ✓ Docker cleaned"

echo ""
echo "=== Reset Complete ==="
echo ""
echo "You can now start fresh:"
echo "  1. Start backend:  poetry run landseer-backend"
echo "  2. Start workers:  poetry run landseer-worker --gpu 0"
