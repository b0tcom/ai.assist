#!/bin/bash

set -e

# Generate a timestamp
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Compose the update message
MESSAGE="Update: $TIMESTAMP"

echo "Staging all changes..."
git add -A

echo "Committing with message: \"$MESSAGE\""
git commit -m "$MESSAGE"

echo "Pushing to current branch..."
git push

echo "Done. [$MESSAGE]"
