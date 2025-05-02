import whisper
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

def get_available_models() -> list[str]:
    """Return list of available Whisper models."""
    return ["tiny", "base", "small", "medium", "large"]

def transcribe_audio(
    file_path: str,
    model_name: str = "base",
    output_dir: Optional[str] = None,
    language: Optional[str] = None,
    save_format: str = "txt"
) -> Dict[str, Any]:
    """
    Transcribe audio file using Whisper model.
    
    Args:
        file_path: Path to audio file
        model_name: Whisper model to use (tiny, base, small, medium, large)
        output_dir: Directory to save transcription results
        language: Language code (e.g., 'en', 'fr', 'es')
        save_format: Output format ('txt' or 'json')
    
    Returns:
        Dict containing transcription results
    """
    # Validate file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")
    
    # Validate model
    if model_name not in get_available_models():
        raise ValueError(f"Invalid model name. Choose from: {', '.join(get_available_models())}")

    # Load model
    print(f"Loading Whisper {model_name} model...")
    model = whisper.load_model(model_name)

    # Transcribe
    print(f"Transcribing '{file_path}'...")
    result = model.transcribe(
        file_path,
        language=language,
        verbose=True
    )

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(file_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if save_format == "json":
            output_path = os.path.join(output_dir, f"{base_name}_{timestamp}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        else:  # txt format
            output_path = os.path.join(output_dir, f"{base_name}_{timestamp}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
        
        print(f"Transcription saved to: {output_path}")

    return result

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using OpenAI's Whisper")
    parser.add_argument("file_path", help="Path to audio file")
    parser.add_argument("--model", "-m", default="base", choices=get_available_models(),
                      help="Whisper model to use")
    parser.add_argument("--output-dir", "-o", help="Directory to save transcription results")
    parser.add_argument("--language", "-l", help="Language code (e.g., 'en', 'fr', 'es')")
    parser.add_argument("--format", "-f", choices=["txt", "json"], default="txt",
                      help="Output format (txt or json)")
    
    args = parser.parse_args()
    
    try:
        result = transcribe_audio(
            args.file_path,
            model_name=args.model,
            output_dir=args.output_dir,
            language=args.language,
            save_format=args.format
        )
        
        if not args.output_dir:
            print("\nTranscription:")
            print(result["text"])
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
