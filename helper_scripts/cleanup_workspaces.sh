#!/bin/bash
# Quick cleanup of worker workspaces in /tmp
# Run this to free disk space from task execution directories

echo "=== Landseer Worker Workspace Cleanup ==="
echo ""

# Show current usage
echo "Current worker workspace usage:"
du -sh /tmp/landseer_worker_* 2>/dev/null || echo "  No worker workspaces found"
echo ""

# Count directories
WORKSPACE_COUNT=$(ls -d /tmp/landseer_worker_* 2>/dev/null | wc -l)

if [ "$WORKSPACE_COUNT" -eq 0 ]; then
    echo "✓ No worker workspaces to clean"
    exit 0
fi

echo "Found $WORKSPACE_COUNT worker workspace(s)"

if [ "$1" == "-y" ] || [ "$1" == "--yes" ]; then
    REPLY="y"
else
    read -p "Delete all worker workspaces? (y/n): " -n 1 -r
    echo
fi

if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf /tmp/landseer_worker_*
    echo "✓ Cleaned all worker workspaces"
else
    echo "Skipped cleanup"
fi
