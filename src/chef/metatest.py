import essentia
import essentia.standard as es
import numpy as np
import json
import os
import sys

print(dir(es.HPCP))

def extract_chord_progression(audio, sr=44100):
    """
    Extract chord progression using Essentia's ChordDetection.
    """
    try:
        print(f"Essentia version: {essentia.__version__}")
        #print("HPCP algorithm info:", es.HPCP.algoInfo())
        print("Starting chord progression extraction...")
        frame_size = 4096
        hop_size = 2048
        
        # Minimal HPCP configuration
        hpcp = es.HPCP(
            size=12,
            referenceFrequency=440.0
        )
        chord_detection = es.ChordDetection(
            sampleRate=sr,
            hopSize=hop_size
        )

        print("Processing audio frames...")
        frames = es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size)
        hpcp_frames = []
        for i, frame in enumerate(frames):
            spectrum = es.Spectrum()(frame)
            freqs, mags = es.SpectralPeaks()(spectrum)
            hpcp_frame = hpcp(freqs, mags)
            hpcp_frames.append(hpcp_frame)
            if i % 100 == 0:
                print(f"Processed {i} frames")

        hpcp_frames = np.array(hpcp_frames)
        print(f"Computing chords for {len(hpcp_frames)} HPCP frames...")
        chords, strengths = chord_detection(hpcp_frames)

        print("Converting chords to timed segments...")
        chord_progression = []
        current_chord = chords[0]
        start_time = 0.0
        for i, chord in enumerate(chords[1:], 1):
            time = i * hop_size / sr
            if chord != current_chord:
                chord_progression.append({
                    "chord": current_chord,
                    "start_time": start_time,
                    "end_time": time
                })
                current_chord = chord
                start_time = time
        chord_progression.append({
            "chord": current_chord,
            "start_time": start_time,
            "end_time": len(audio) / sr
        })

        print(f"Extracted {len(chord_progression)} chord segments")
        return chord_progression
    except Exception as e:
        print(f"Error in extract_chord_progression: {str(e)}")
        return []

def process_audio_file(input_file, output_dir="output"):
    """
    Process an audio file and extract features using Essentia.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Loading audio: {input_file}")
        audio = es.MonoLoader(filename=input_file, sampleRate=44100)()
        print(f"Audio loaded, length: {len(audio)} samples, duration: {len(audio)/44100:.2f} seconds")
        
        features = {}
        features["chord_progression"] = extract_chord_progression(audio)

        output_json = os.path.join(output_dir, "features.json")
        print(f"Saving features to: {output_json}")
        with open(output_json, "w") as f:
            json.dump(features, f, indent=2)
        
        return features
    except Exception as e:
        print(f"Error in process_audio_file: {str(e)}")
        return {}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process audio file with Essentia")
    parser.add_argument("--input", required=True, help="Input audio file")
    args = parser.parse_args()
    
    process_audio_file(args.input)