# File: s3_media_manager.py
# This script helps manage your media files in S3

import os
import boto3
import argparse
from botocore.exceptions import ClientError
from tqdm import tqdm  # For progress bars

class S3MediaManager:
    def __init__(self, bucket_name="videous-media-files"):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')
        self.bucket = self.s3_resource.Bucket(bucket_name)
    
    def list_files(self, prefix="videos/", limit=None):
        """List all files in the S3 bucket with the given prefix"""
        print(f"Listing files in s3://{self.bucket_name}/{prefix}")
        
        count = 0
        for obj in self.bucket.objects.filter(Prefix=prefix):
            print(f"- {obj.key} ({self.format_size(obj.size)})")
            count += 1
            if limit and count >= limit:
                print(f"\n[Showing {limit} of {count} files. Use --limit 0 to see all.]")
                break
        
        if count == 0:
            print("No files found.")
        else:
            print(f"\nTotal: {count} files")
    
    def download_file(self, s3_key, local_path=None):
        """Download a file from S3"""
        if local_path is None:
            local_path = os.path.basename(s3_key)
        
        try:
            file_size = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)['ContentLength']
            
            # Create progress callback
            progress = tqdm(total=file_size, unit='B', unit_scale=True, 
                            desc=f"Downloading {os.path.basename(s3_key)}")
            
            def callback(bytes_transferred):
                progress.update(bytes_transferred)
            
            self.s3_client.download_file(
                self.bucket_name, 
                s3_key, 
                local_path, 
                Callback=callback
            )
            progress.close()
            print(f"Successfully downloaded to {local_path}")
            return True
        except ClientError as e:
            print(f"Error downloading file: {e}")
            return False
    
    def upload_file(self, local_path, s3_key=None):
        """Upload a file to S3"""
        if not os.path.exists(local_path):
            print(f"Error: File {local_path} does not exist")
            return False
        
        if s3_key is None:
            s3_key = f"videos/{os.path.basename(local_path)}"
        
        try:
            file_size = os.path.getsize(local_path)
            
            # Create progress callback
            progress = tqdm(total=file_size, unit='B', unit_scale=True, 
                            desc=f"Uploading {os.path.basename(local_path)}")
            
            def callback(bytes_transferred):
                progress.update(bytes_transferred)
            
            self.s3_client.upload_file(
                local_path, 
                self.bucket_name, 
                s3_key, 
                Callback=callback,
                ExtraArgs={'ContentType': self.get_content_type(local_path)}
            )
            progress.close()
            print(f"Successfully uploaded to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            print(f"Error uploading file: {e}")
            return False
    
    def delete_file(self, s3_key):
        """Delete a file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            print(f"Successfully deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            print(f"Error deleting file: {e}")
            return False
    
    def batch_upload(self, directory, extensions=['.mp4', '.mov'], recursive=False):
        """Upload multiple files with specific extensions from a directory"""
        file_count = 0
        success_count = 0
        
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, directory)
                    s3_key = f"videos/{relative_path.replace(os.sep, '/')}"
                    
                    file_count += 1
                    if self.upload_file(file_path, s3_key):
                        success_count += 1
            
            if not recursive:
                break
        
        print(f"\nUpload complete: {success_count}/{file_count} files successfully uploaded")
    
    @staticmethod
    def format_size(size_bytes):
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024 or unit == 'TB':
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
    
    @staticmethod
    def get_content_type(filename):
        """Get content type based on file extension"""
        extension = os.path.splitext(filename.lower())[1]
        content_types = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.wav': 'audio/wav',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png'
        }
        return content_types.get(extension, 'application/octet-stream')


def main():
    parser = argparse.ArgumentParser(description='Manage media files in Amazon S3')
    parser.add_argument('--bucket', default='videous-media-files', help='S3 bucket name')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List files in S3')
    list_parser.add_argument('--prefix', default='videos/', help='Prefix for listing')
    list_parser.add_argument('--limit', type=int, default=20, help='Limit number of files to show (0 for all)')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download a file from S3')
    download_parser.add_argument('key', help='S3 key of the file to download')
    download_parser.add_argument('--output', help='Local path to save the file')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload a file to S3')
    upload_parser.add_argument('file', help='Local path of the file to upload')
    upload_parser.add_argument('--key', help='S3 key to use (default: videos/filename)')
    
    # Batch upload command
    batch_parser = subparsers.add_parser('batch-upload', help='Upload multiple files to S3')
    batch_parser.add_argument('directory', help='Directory containing files to upload')
    batch_parser.add_argument('--recursive', action='store_true', help='Recursively upload files')
    batch_parser.add_argument('--extensions', default='.mp4,.mov', 
                              help='Comma-separated list of file extensions to upload')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a file from S3')
    delete_parser.add_argument('key', help='S3 key of the file to delete')
    
    args = parser.parse_args()
    manager = S3MediaManager(args.bucket)
    
    if args.command == 'list':
        limit = None if args.limit == 0 else args.limit
        manager.list_files(args.prefix, limit)
    
    elif args.command == 'download':
        manager.download_file(args.key, args.output)
    
    elif args.command == 'upload':
        manager.upload_file(args.file, args.key)
    
    elif args.command == 'batch-upload':
        extensions = args.extensions.split(',')
        manager.batch_upload(args.directory, extensions, args.recursive)
    
    elif args.command == 'delete':
        manager.delete_file(args.key)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
