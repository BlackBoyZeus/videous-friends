# Videous & Friends Architecture

This document outlines the architecture of the Videous & Friends platform, explaining how the different components interact.

## System Overview

Videous & Friends is a comprehensive video editing and processing platform with three main components:

1. **Pegasus Editor**: Front-end video editing interface
2. **Videous CHEF**: Back-end video processing engine
3. **StarLeague**: Video analysis and enhancement system

## Component Interactions

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Pegasus Editor │────▶│  Videous CHEF   │────▶│   StarLeague    │
│  (Front-end)    │◀────│  (Processing)   │◀────│   (Analysis)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Amazon S3 Storage                         │
│                                                                 │
│  - Input Videos                                                 │
│  - Processed Videos                                             │
│  - Model Files                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Pegasus Editor

The Pegasus Editor is the main user interface for video editing:

- **Highlight Detection**: Automatically identifies key moments in videos
- **Timeline Editing**: Allows manual adjustment of detected highlights
- **Export Options**: Provides various export formats and quality settings

Key files:
- `src/pegasus/app.py`: Main application entry point
- `src/pegasus/sport.py`: Sports-specific functionality

## Videous CHEF

Videous CHEF (Comprehensive Highlight Enhancement Framework) is the core processing engine:

- **Video Effects**: Applies professional-grade effects to videos
- **Video Matting**: Separates foreground and background elements
- **Background Replacement**: Replaces video backgrounds with custom images or videos
- **Depth Estimation**: Analyzes depth information for 3D effects

Key modules:
- `src/chef/effects/`: Video effects processing
- `src/chef/matting/`: Video matting and background separation
- `src/chef/api/`: API interface for video processing

## StarLeague

StarLeague provides advanced video analysis and enhancement:

- **Content Analysis**: Analyzes video content for specific features
- **Enhancement**: Enhances video quality based on analysis
- **Custom Processing**: Applies domain-specific processing for sports and gaming

Key files:
- `src/starleague/StarLeague.py`: Main StarLeague application
- `src/starleague/StarLeaguecustom.py`: Custom implementation

## Data Flow

1. **Input**: Videos are uploaded to Amazon S3 or processed locally
2. **Processing**: Videous CHEF processes videos with requested effects
3. **Analysis**: StarLeague analyzes video content for specific features
4. **Editing**: Pegasus Editor provides interface for final adjustments
5. **Output**: Processed videos are saved locally or to Amazon S3

## Technology Stack

- **Programming Languages**: Python, Swift
- **Machine Learning**: PyTorch, TensorFlow
- **Video Processing**: OpenCV, FFmpeg
- **Cloud Storage**: Amazon S3
- **UI Frameworks**: Qt (Pegasus), SwiftUI (StarLeague iOS)

## Development Workflow

1. **Local Development**: Develop and test components locally
2. **Integration Testing**: Test component interactions
3. **Deployment**: Deploy to production environment
4. **Monitoring**: Monitor system performance and user feedback
