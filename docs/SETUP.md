# Setup Guide for Videous & Friends

This guide provides instructions for setting up the Videous & Friends platform for development and production use.

## Prerequisites

- Python 3.10 or higher
- FFmpeg 4.4 or higher
- Git LFS (for cloning the repository with large files)
- AWS CLI (for S3 integration)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/BlackBoyZeus/videous-friends.git
cd videous-friends
```

### 2. Set Up Virtual Environment

```bash
# Create a virtual environment
python -m venv pegasus_env_py310

# Activate the virtual environment
# On Windows:
pegasus_env_py310\Scripts\activate
# On macOS/Linux:
source pegasus_env_py310/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure AWS Credentials

For S3 integration, configure your AWS credentials:

```bash
aws configure
```

Enter your AWS Access Key ID, Secret Access Key, default region, and output format.

### 5. Configure the Application

Copy the example configuration files:

```bash
cp configs/config.ini.example configs/config.ini
cp configs/chef_config.json.example configs/chef_config.json
```

Edit the configuration files to match your environment.

## Running the Application

### Pegasus Editor

```bash
python -m src.pegasus.app
```

### Videous CHEF

```bash
python -m src.chef.main
```

### StarLeague

```bash
python -m src.starleague.main
```

## Development Setup

### Setting Up for Development

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
python -m tests.run_tests
```

## Troubleshooting

### Common Issues

1. **Missing FFmpeg**: Ensure FFmpeg is installed and in your PATH
2. **S3 Access Denied**: Check your AWS credentials and bucket permissions
3. **Missing Models**: Download required models from S3 to the `models/` directory

### Getting Help

If you encounter issues, please:

1. Check the documentation in the `docs/` directory
2. Review the logs in the `logs/` directory
3. Contact the development team

## Deployment

### Production Deployment

1. Build the application:
   ```bash
   python setup.py build
   ```

2. Deploy to your production environment:
   ```bash
   python setup.py install
   ```

### Docker Deployment

A Dockerfile is provided for containerized deployment:

```bash
docker build -t videous-friends .
docker run -p 8080:8080 videous-friends
```
