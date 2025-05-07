# Audio Transcriber

A Python script that uses OpenAI's Whisper model to transcribe audio files.

## Setup

1. Create a virtual environment using `uv`:
```bash
uv venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage

Basic command:
```bash
python notebooks/transcriber.py <path_to_audio_file> --model <model_size> --language <language_code> --output-dir <output_directory>
```

### Options

- `--model` or `-m`: Whisper model to use
  - Available models: tiny, base, small, medium, large
  - Default: base
  - Example: `--model small`

- `--language` or `-l`: Language code for transcription
  - Examples: en (English), fr (French), es (Spanish)
  - Default: auto-detect
  - Example: `--language en`

- `--output-dir` or `-o`: Directory to save transcription results
  - Creates the directory if it doesn't exist
  - Example: `--output-dir data`

- `--format` or `-f`: Output format
  - Options: txt, json
  - Default: txt
  - Example: `--format json`

### Examples

1. Transcribe in English using small model:
```bash
python notebooks/transcriber.py "path/to/audio.m4a" --model small --language en --output-dir data
```

2. Transcribe in French using base model:
```bash
python notebooks/transcriber.py "path/to/audio.m4a" --model base --language fr --output-dir data
```

3. Save as JSON with all metadata:
```bash
python notebooks/transcriber.py "path/to/audio.m4a" --model small --language en --output-dir data --format json
```

## Output

The script will create timestamped output files in the specified output directory:
- For txt format: `filename_YYYYMMDD_HHMMSS.txt`
- For json format: `filename_YYYYMMDD_HHMMSS.json`

## Notes

- The script supports various audio formats (m4a, mp3, wav, etc.)
- Larger models (medium, large) provide better accuracy but require more memory and processing time
- The first run will download the selected model, which may take some time depending on your internet connection 