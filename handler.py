import base64
import tempfile
import os
import subprocess
import soundfile as sf
import librosa
from pathlib import Path
import runpod

def generate_speech(
    text: str,
    speaker_wav_base64: str,
    reference_text: str = "",
    language: str = 'en',
    stability: float = 0.75,
    similarity: float = 0.85,
    speed: float = 1.0,
    repetition_penalty: float = 1.2
) -> bytes:
    temperature = max(0.01, 1.0 - stability)
    top_p = similarity

    speaker_bytes = base64.b64decode(speaker_wav_base64)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as ref_file:
        ref_path = ref_file.name
        ref_file.write(speaker_bytes)

    prompt_tokens_npy = 'fake.npy'   # Latest DAC output file name
    output_codes_npy = 'codes_0.npy' # text2semantic output
    output_wav = 'output.wav'

    try:
        # Step 1: DAC encode reference audio → prompt tokens (fake.npy)
        subprocess.run([
            'python', '-m', 'fish_speech.models.dac.inference',
            '-i', ref_path,
            '--checkpoint-path', 'checkpoints/openaudio-s1-mini/codec.pth'
        ], check=True, capture_output=True)

        # Step 2: Text to semantic tokens
        cmd = [
            'python', '-m', 'fish_speech.models.text2semantic.inference',
            '--text', text,
            '--prompt-text', reference_text,
            '--prompt-tokens', prompt_tokens_npy,
            '--top-p', str(top_p),
            '--temperature', str(temperature),
            '--repetition-penalty', str(repetition_penalty),
            '--checkpoint-path', 'checkpoints/openaudio-s1-mini',
            '--output-dir', '.',
            '--num-samples', '1'
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Step 3: Decode semantic to waveform
        subprocess.run([
            'python', 'tools/vqgan/inference.py',
            '-i', output_codes_npy,
            '--checkpoint-path', 'checkpoints/openaudio-s1-mini/firefly-gan-vq-fsq-8x1024-21hz-generator.pth',
            '--output', output_wav
        ], check=True, capture_output=True)

        # Post-process: speed & resample to 48kHz
        audio, sr = librosa.load(output_wav, sr=None)
        if speed != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=speed)
        if sr != 48000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
            sf.write(tmp_out.name, audio, 48000, format='WAV')
            with open(tmp_out.name, 'rb') as f:
                audio_bytes = f.read()
            os.unlink(tmp_out.name)

        return audio_bytes

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Inference step failed: {e.stderr.decode() if e.stderr else str(e)}")
    finally:
        for p in [ref_path, prompt_tokens_npy, output_codes_npy, output_wav]:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except:
                    pass

# RunPod Serverless Handler
def infinity_handler(job):
    job_input = job['input']
    text = job_input.get('text', 'Hello world')
    speaker_base64 = job_input.get('speaker_wav_base64')
    ref_text = job_input.get('reference_text', '')
    lang = job_input.get('language', 'en')
    stability = job_input.get('stability', 0.75)
    similarity = job_input.get('similarity', 0.85)
    speed = job_input.get('speed', 1.0)

    audio_bytes = generate_speech(text, speaker_base64, ref_text, lang, stability, similarity, speed)

    return {
        "audio_base64": base64.b64encode(audio_bytes).decode('utf-8'),
        "format": "wav",
        "sample_rate": 48000
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": infinity_handler})