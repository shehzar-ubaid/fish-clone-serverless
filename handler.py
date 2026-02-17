import base64
import tempfile
import os
import subprocess
import soundfile as sf
import librosa
from pathlib import Path

def generate_speech(
    text: str,
    speaker_wav_base64: str,
    reference_text: str = "",  # Optional but improves cloning
    language: str = 'en',      # Model auto-detects, but can help
    stability: float = 0.75,
    similarity: float = 0.85,
    speed: float = 1.0,
    repetition_penalty: float = 1.2
) -> bytes:
    """
    InfinityClone: High-fidelity voice cloning using OpenAudio-S1-mini.
    Output: 48kHz WAV audio bytes.
    """
    temperature = max(0.01, 1.0 - stability)  # Avoid zero temp
    top_p = similarity

    # Decode base64 audio
    speaker_bytes = base64.b64decode(speaker_wav_base64)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as ref_file:
        ref_path = ref_file.name
        ref_file.write(speaker_bytes)

    codes_path = Path(ref_path).with_suffix('.npy')  # dac outputs .npy
    output_codes = 'codes_0.npy'                     # default single sample
    output_wav = 'output.wav'

    try:
        # Step 1: DAC encode reference → prompt tokens
        subprocess.run([
            'python', '-m', 'fish_speech.models.dac.inference',
            '--input-path', ref_path,
            '--checkpoint-path', 'checkpoints/s1-mini/codec.pth'
        ], check=True, capture_output=True)

        # Step 2: Text2Semantic – generate semantic tokens
        cmd = [
            'python', '-m', 'fish_speech.models.text2semantic.inference',
            '--text', text,
            '--prompt-text', reference_text,
            '--prompt-tokens', str(codes_path),
            '--top-p', str(top_p),
            '--temperature', str(temperature),
            '--repetition-penalty', str(repetition_penalty),
            '--checkpoint-path', 'checkpoints/s1-mini',
            '--output-dir', '.',
            '--num-samples', '1'
        ]
        if speed != 1.0:
            cmd += ['--speed', str(speed)]  # If supported in latest, else handle in librosa
        subprocess.run(cmd, check=True, capture_output=True)

        # Step 3: Decode to waveform (Firefly GAN / VQ)
        subprocess.run([
            'python', 'tools/vqgan/inference.py',
            '-i', output_codes,
            '--checkpoint-path', 'checkpoints/s1-mini/firefly-gan-vq-fsq-8x1024-21hz-generator.pth',
            '--output', output_wav
        ], check=True, capture_output=True)

        # Load + post-process
        audio, sr = librosa.load(output_wav, sr=None)

        if speed != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=speed)

        if sr != 48000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)

        # To bytes
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
            sf.write(tmp_out.name, audio, 48000, format='WAV')
            with open(tmp_out.name, 'rb') as f:
                audio_bytes = f.read()
            os.unlink(tmp_out.name)

        return audio_bytes

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Inference failed: {e.stderr.decode()}")
    finally:
        # Safe cleanup
        for p in [ref_path, codes_path, output_codes, output_wav]:
            if os.path.exists(p):
                os.unlink(p)

# For local testing:
# if __name__ == "__main__":
#     with open("test_ref.wav", "rb") as f:
#         b64 = base64.b64encode(f.read()).decode()
#     audio = generate_speech("Hello, this is a test in Roman English.", b64, "Hello, this is reference speech.")
#     with open("test_out.wav", "wb") as f:
#         f.write(audio)