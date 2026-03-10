import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import joblib

AUDIO_FILE_PATH = r"C:\Users\Hp\OneDrive\Documents\Dataset\Fake\file10119.mp3.wav_16k.wav_norm.wav_mono.wav_silence.wav"

def extract_features(file_path: str, plot: bool = True) -> np.ndarray:
    temp_wav_path = None
    y_plot = None
    sr_plot = None
    
    try:
        if file_path.lower().endswith('.mp3'):
            audio = AudioSegment.from_mp3(file_path)
            temp_wav_path = os.path.join(os.getcwd(), os.path.basename(file_path).replace('.mp3', '_temp.wav'))
            audio.export(temp_wav_path, format='wav')
            file_path = temp_wav_path

        y, sr = librosa.load(file_path, sr=22050, mono=True)
        y_plot, sr_plot = y.copy(), sr

        if len(y) == 0:
            return None

        min_samples = sr // 10
        if len(y) < min_samples:
            y = np.pad(y, (0, min_samples - len(y)), mode='constant')

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        sc_mean = np.mean(spectral_contrast, axis=1)
        sc_std = np.std(spectral_contrast, axis=1)

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean, zcr_std = np.mean(zcr), np.std(zcr)

        # STFT features (for prediction only)
        stft = np.abs(librosa.stft(y))
        spectral_centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(S=stft)[0]

        stft_features = [
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            np.mean(spectral_rolloff), np.std(spectral_rolloff),
            np.mean(spectral_flatness), np.std(spectral_flatness)
        ]

        # Pitch (F0)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_clean = f0[~np.isnan(f0)]
        f0_mean, f0_std = (np.mean(f0_clean), np.std(f0_clean)) if len(f0_clean) > 0 else (0.0, 0.0)

        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = np.mean(log_mel_spec, axis=1)
        mel_std = np.std(log_mel_spec, axis=1)

        # Plotting - TWO SEPARATE WINDOWS
        if plot and y_plot is not None:
            # WINDOW 1: Time-Frequency Representations
            plt.figure(figsize=(14, 10))
            plt.suptitle('Time-Frequency Representations', fontsize=14, fontweight='bold')
            
            # Waveform
            plt.subplot(2, 2, 1)
            plt.plot(y_plot, color='#1f77b4', linewidth=1.2)
            plt.title('Waveform', fontsize=12, fontweight='bold')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)

            # STFT Magnitude
            plt.subplot(2, 2, 2)
            stft_plot = np.abs(librosa.stft(y_plot))
            librosa.display.specshow(librosa.amplitude_to_db(stft_plot, ref=np.max), 
                                   sr=sr_plot, x_axis='time', y_axis='log', cmap='magma')
            plt.title('STFT Magnitude', fontsize=12, fontweight='bold')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(format='%+2.0f dB')

            # MFCC
            plt.subplot(2, 2, 3)
            librosa.display.specshow(mfcc, sr=sr, x_axis='time', y_axis='mel', cmap='plasma')
            plt.title('MFCC', fontsize=12, fontweight='bold')
            plt.xlabel('Time (s)')
            plt.ylabel('MFCC Coefficient')
            plt.colorbar()

            # Log-Mel Spectrogram
            plt.subplot(2, 2, 4)
            librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
            plt.title('Log-Mel Spectrogram', fontsize=12, fontweight='bold')
            plt.xlabel('Time (s)')
            plt.ylabel('Mel Frequency')
            plt.colorbar(format='%+2.0f dB')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

            # WINDOW 2: Feature Analysis
            plt.figure(figsize=(14, 10))
            plt.suptitle('Feature Analysis', fontsize=14, fontweight='bold')
            
            # Frequency Spectrum (FFT)
            plt.subplot(2, 2, 1)
            n_fft = min(4096, len(y_plot))
            fft_vals = np.fft.fft(y_plot, n=n_fft)
            freqs = np.fft.fftfreq(n_fft, 1/sr_plot)
            idx = freqs >= 0
            plt.plot(freqs[idx], 20*np.log10(np.abs(fft_vals[idx]) + 1e-10), color='#ff7f0e', linewidth=1.2)
            plt.title('Frequency Spectrum (FFT)', fontsize=12, fontweight='bold')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.xlim(0, 8000)
            plt.grid(True, alpha=0.3)

            # Spectral Contrast
            plt.subplot(2, 2, 2)
            librosa.display.specshow(spectral_contrast, sr=sr, x_axis='time', y_axis='chroma', cmap='inferno')
            plt.title('Spectral Contrast', fontsize=12, fontweight='bold')
            plt.xlabel('Time (s)')
            plt.ylabel('Chroma Band')
            plt.colorbar()

            # Zero Crossing Rate
            plt.subplot(2, 2, 3)
            times_zcr = librosa.times_like(zcr)
            plt.plot(times_zcr, zcr, color='#2ca02c', linewidth=1.5)
            plt.title('Zero Crossing Rate', fontsize=12, fontweight='bold')
            plt.xlabel('Time (s)')
            plt.ylabel('Rate')
            plt.grid(True, alpha=0.3)

            # Pitch (F0) Contour
            plt.subplot(2, 2, 4)
            times_f0 = librosa.times_like(f0)
            plt.plot(times_f0, f0, color='#d62728', linewidth=1.5)
            plt.title('Pitch (F0) Contour', fontsize=12, fontweight='bold')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.ylim(0, librosa.note_to_hz('C7'))
            plt.grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        return np.concatenate([
            mfcc_mean, mfcc_std,
            sc_mean, sc_std,
            [zcr_mean, zcr_std],
            stft_features,
            [f0_mean, f0_std],
            mel_mean, mel_std
        ])

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except:
                pass

if __name__ == "__main__":
    print("Audio Deepfake Detector with Clean Signal Analysis")
    print("=" * 60)

    if not os.path.isfile(AUDIO_FILE_PATH):
        print(f"File not found: {AUDIO_FILE_PATH}")
        exit(1)

    try:
        model = joblib.load("audio_deepfake_xgb_model.pkl")
        scaler = joblib.load("audio_feature_scaler.pkl")
    except FileNotFoundError:
        print("Model files not found!")
        print("Required files:")
        print(" - audio_deepfake_xgb_model.pkl")
        print(" - audio_feature_scaler.pkl")
        exit(1)

    print(f"Analyzing: {os.path.basename(AUDIO_FILE_PATH)}")
    features = extract_features(AUDIO_FILE_PATH, plot=True)
    
    if features is None:
        print("Feature extraction failed.")
        exit(1)

    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    confidence = model.predict_proba(features_scaled)[0].max()
    result = "Real" if prediction == 0 else "Fake"

    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    if result == "Fake":
        print("Detected as SYNTHETIC (Fake)")
        print("- Potential AI-generated voice")
        print("- Check for unnatural pitch patterns")
    else:
        print("Classified as AUTHENTIC (Real)")
        print("- Natural speech characteristics detected")
    
    print(f"\nPrediction: {result}")
    print(f"Confidence: {confidence:.4f}")
    
    if confidence < 0.8:
        print("\nNote: Low confidence - manual verification recommended")
    
    print("\n" + "=" * 60)
    print("Analysis complete")
    print("=" * 60)