#!/bin/bash
echo "Starting FreshTrack backend build..."

# Clean install
echo "Cleaning and installing dependencies..."
rm -rf node_modules package-lock.json
npm cache clean --force
npm install --omit=dev

echo "Build completed successfully!"
