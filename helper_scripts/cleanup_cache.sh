#!/bin/bash
# Clear the artifact cache
# Run this to free disk space from cached task outputs

echo "=== Landseer Cache Cleanup ==="
echo ""

CACHE_DIR="${LANDSEER_CACHE_DIR:-/tmp/landseer_cache}"

# Show current usage
echo "Cache directory: $CACHE_DIR"
if [ -d "$CACHE_DIR" ]; then
    du -sh "$CACHE_DIR" 2>/dev/null
    ITEM_COUNT=$(find "$CACHE_DIR" -mindepth 1 -maxdepth 2 -type d 2>/dev/null | wc -l)
    echo "Cache entries: $ITEM_COUNT"
else
    echo "  Cache directory does not exist"
    exit 0
fi
echo ""

if [ "$1" == "-y" ] || [ "$1" == "--yes" ]; then
    REPLY="y"
else
    read -p "Clear all cache? (y/n): " -n 1 -r
    echo
fi

if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$CACHE_DIR"/*
    echo "✓ Cache cleared"
else
    echo "Skipped cleanup"
fi
