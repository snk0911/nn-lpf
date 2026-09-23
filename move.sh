#!/usr/bin/env bash

SOURCE="out"
TARGET="best_checkpoints"

# Create target directory if it does not exist
mkdir -p "$TARGET"

# Move all *_best.pth.tar files recursively from out/ to best_checkpoints/
find "$SOURCE" -type f -name '*_best.pth.tar' \
    -exec mv -v -- {} "$TARGET/" \;

echo "All best checkpoints have been moved to $TARGET."