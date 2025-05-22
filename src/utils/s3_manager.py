"""
S3 Media Manager for Videous & Friends

This module provides utilities for managing media files in Amazon S3.
"""

import os
import boto3
from botocore.exceptions import ClientError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3MediaManager:
    """Manages media files in Amazon S3."""
    
    def __init__(self, bucket_name="videous-media-files", prefix="videos/"):
        """Initialize the S3 media manager.
        
        Args:
            bucket_name (str): The S3 bucket name
            prefix (str): The prefix (folder) within the bucket
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3_client = boto3.client('s3')
    
    def list_media_files(self):
        """List all media files in the S3 bucket.
        
        Returns:
            list: List of media file names
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.prefix
            )
            
            if 'Contents' not in response:
                return []
                
            return [item['Key'].replace(self.prefix, '') for item in response['Contents']]
        except ClientError as e:
            logger.error(f"Error listing S3 files: {e}")
            return []
    
    def download_file(self, filename, local_path=None):
        """Download a file from S3.
        
        Args:
            filename (str): The name of the file in S3
            local_path (str, optional): Local path to save the file.
                                       If None, saves to current directory.
        
        Returns:
            str: Path to the downloaded file or None if failed
        """
        if local_path is None:
            local_path = os.path.basename(filename)
            
        s3_key = f"{self.prefix}{filename}"
        
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded {filename} to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Error downloading {filename}: {e}")
            return None
    
    def upload_file(self, local_path, s3_filename=None):
        """Upload a file to S3.
        
        Args:
            local_path (str): Path to the local file
            s3_filename (str, optional): Name to use in S3.
                                        If None, uses the basename of local_path.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(local_path):
            logger.error(f"File not found: {local_path}")
            return False
            
        if s3_filename is None:
            s3_filename = os.path.basename(local_path)
            
        s3_key = f"{self.prefix}{s3_filename}"
        
        try:
            # Determine content type based on file extension
            content_type = "video/mp4"  # Default
            if local_path.lower().endswith('.mov'):
                content_type = "video/quicktime"
            elif local_path.lower().endswith('.wav'):
                content_type = "audio/wav"
                
            self.s3_client.upload_file(
                local_path, 
                self.bucket_name, 
                s3_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'StorageClass': 'STANDARD'
                }
            )
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Error uploading {local_path}: {e}")
            return False
    
    def delete_file(self, filename):
        """Delete a file from S3.
        
        Args:
            filename (str): Name of the file to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        s3_key = f"{self.prefix}{filename}"
        
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting {filename}: {e}")
            return False


# Example usage
if __name__ == "__main__":
    s3_manager = S3MediaManager()
    
    # List files
    files = s3_manager.list_media_files()
    print(f"Found {len(files)} files in S3")
