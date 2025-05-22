#!/bin/bash

# The bucket name
BUCKET="videous-media-files"

# Create a uploads directory for tracking
mkdir -p .s3_uploads

# Function to upload a file with progress
upload_file() {
    local file=$1
    local filename=$(basename "$file")
    local upload_tracker=".s3_uploads/${filename}.uploaded"
    
    if [ -f "$upload_tracker" ]; then
        echo "Skipping $filename (already uploaded)"
        return
    fi
    
    echo "Uploading $filename to S3..."
    aws s3 cp "$file" "s3://$BUCKET/videos/$filename" \
        --storage-class STANDARD \
        --metadata "ContentType=video/mp4"
    
    if [ $? -eq 0 ]; then
        touch "$upload_tracker"
        echo "✓ Successfully uploaded $filename"
    else
        echo "✗ Failed to upload $filename"
    fi
}

# Upload all MP4 files
for file in $(find . -name "*.mp4" -type f); do
    upload_file "$file"
done

# Upload all MOV files
for file in $(find . -name "*.mov" -type f); do
    upload_file "$file"
done

echo "All uploads complete! Check .s3_uploads directory for status."
