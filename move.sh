#!/bin/bash
# Moves files with "best" in their name from out/*/ to saved_checkpoints/
 
mkdir -p saved_checkpoints
 
find out -mindepth 2 -type f -name "*best*" -exec mv -v {} saved_checkpoints/ \;