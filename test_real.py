import os
import wave
import struct
from dotenv import load_dotenv

if __name__ != "__main__":
    import pytest

    pytest.skip("manual Gemini batch smoke script", allow_module_level=True)

# Load real environment variables
load_dotenv()

from calls_analyser.services.gemini_batch import GeminiBatchRunner, BatchTask

def create_dummy_wav(path="dummy.wav"):
    with wave.open(path, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        # 1 second of silence
        data = struct.pack('<h', 0) * 8000
        f.writeframes(data)

def test_real_batch():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found in .env")
        return
        
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "canvas-genius-492412-c3")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    # Usually we use a model like gemini-2.5-flash-lite
    model = "models/gemini-2.5-flash-lite"
    
    print("Initializing GeminiBatchRunner...")
    runner = GeminiBatchRunner(
        api_key=api_key,
        model=model,
        project=project,
        location=location
    )
    
    print("Creating dummy wav file...")
    create_dummy_wav("dummy.wav")
    
    tasks = [
        BatchTask(key="test-1", path="dummy.wav", mime_type="audio/wav")
    ]
    
    print("Running batch for 1 file...")
    results = runner.run_batch(tasks, prompt_text="Please describe this audio.", chunk_size=1)
    
    print("Results:")
    print(results)
    
if __name__ == "__main__":
    test_real_batch()
