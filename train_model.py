import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from pydub import AudioSegment
from joblib import Parallel, delayed, dump
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import matplotlib.pyplot as plt


# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'


def extract_features(
    file_path: str,
    sample_rate: int = 22050,
    augment: bool = False,
    plot: bool = False
) -> np.ndarray:
    temp_wav_path = None
    try:
        # Handle .mp3 files
        if file_path.lower().endswith('.mp3'):
            audio = AudioSegment.from_mp3(file_path)
            temp_wav_path = os.path.join(os.getcwd(), os.path.basename(file_path).replace('.mp3', '_temp.wav'))
            audio.export(temp_wav_path, format='wav')
            file_path = temp_wav_path

        y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        if len(y) == 0:
            return None

        # Augmentation is DISABLED permanently (set to False in pipeline)
        if augment:
            noise = np.random.normal(0, 0.015, y.shape)
            y = y + noise
            n_steps = np.random.uniform(-2, 2)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            rate = np.random.uniform(0.8, 1.2)
            y = librosa.effects.time_stretch(y, rate=rate)

        # Ensure minimum length (0.1 sec)
        min_samples = sr // 10
        if len(y) < min_samples:
            y = np.pad(y, (0, min_samples - len(y)), mode='constant')

        # --- 1. MFCC ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # --- 2. Spectral Contrast ---
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        sc_mean = np.mean(spectral_contrast, axis=1)
        sc_std = np.std(spectral_contrast, axis=1)

        # --- 3. Zero Crossing Rate ---
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean, zcr_std = np.mean(zcr), np.std(zcr)

        # --- 4. STFT-based Features ---
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

        # --- 5. Pitch (F0) Estimation ---
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        f0 = f0[~np.isnan(f0)]
        if len(f0) == 0:
            f0_mean, f0_std = 0.0, 0.0
        else:
            f0_mean, f0_std = np.mean(f0), np.std(f0)

        # --- 6. Mel Spectrogram Stats ---
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = np.mean(log_mel_spec, axis=1)
        mel_std = np.std(log_mel_spec, axis=1)

        # --- Combine all features ---
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            sc_mean, sc_std,
            [zcr_mean, zcr_std],
            stft_features,
            [f0_mean, f0_std],
            mel_mean, mel_std
        ])

        # --- Optional Plotting (for debugging only) ---
        if plot:
            plt.figure(figsize=(14, 10))
            
            plt.subplot(3, 1, 1)
            plt.plot(y)
            plt.title(f'Waveform: {os.path.basename(file_path)}')
            plt.xlabel('Samples'); plt.ylabel('Amplitude')

            plt.subplot(3, 1, 2)
            librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Log-Mel Spectrogram')

            plt.subplot(3, 1, 3)
            times = librosa.times_like(f0)
            plt.plot(times, f0, label='F0', color='r')
            plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)')
            plt.title('Pitch (F0) Contour')
            plt.legend()

            plt.tight_layout()
            plt.show()

        return features

    except Exception as e:
        print(f" Error extracting features from {file_path}: {str(e)}")
        return None
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except:
                pass


def extract_features_with_timeout(file_path: str, sample_rate: int = 22050, augment: bool = False, timeout: int = 15) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(extract_features, file_path, sample_rate, augment, plot=False)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            print(f" Timeout (>{timeout}s) extracting features from {file_path}")
            return None


def load_dataset(for_base_path: str, augment: bool = False, max_files_per_split: int = None) -> tuple[np.ndarray, np.ndarray]:
    print(f"\n Starting dataset loading from base path: {for_base_path}")
    
    def get_audio_files(directory: str) -> list[str]:
        if not os.path.exists(directory):
            print(f"     Directory does not exist: {directory}")
            return []
        files = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.lower().endswith(('.wav', '.mp3'))
        ]
        if max_files_per_split is not None:
            files = files[:max_files_per_split]
        print(f"     Found {len(files)} audio files in {directory}")
        return files

    X, y = [], []
    skipped_files = 0

    folder_map = {
        'for-2sec': 'for-2seconds',
        'for-norm': 'for-norm',
        'for-original': 'for-original',
        'for-rerec': 'for-rerecorded'
    }

    total_folders_checked = 0
    valid_folders_found = 0

    for folder, inner_name in folder_map.items():
        subfolder_path = os.path.join(for_base_path, folder, inner_name)
        total_folders_checked += 1
        print(f"\n Checking folder: {subfolder_path}")
        
        if not os.path.exists(subfolder_path):
            print(f" Skipping {subfolder_path} (not found)")
            continue
        
        valid_folders_found += 1

        for split in ['training', 'testing', 'validation']:
            split_path = os.path.join(subfolder_path, split)
            print(f"   Checking split: {split_path}")
            if not os.path.exists(split_path):
                print(f"     Split not found: {split_path}")
                continue

            for label, label_dir in enumerate(['real', 'fake']):
                data_path = os.path.join(split_path, label_dir)
                files = get_audio_files(data_path)
                if not files:
                    print(f"     No files in {data_path}")
                    continue

                print(f"   Processing {len(files)} {label_dir} files from {split} split")

                # Extract features (augmentation is OFF)
                results = Parallel(n_jobs=2)(
                    delayed(extract_features_with_timeout)(file, augment=False)
                    for file in tqdm(files, desc=f"{split}-{label_dir}", leave=False, position=0)
                )

                valid_count = 0
                for feat in results:
                    if feat is not None:
                        X.append(feat)
                        y.append(label)
                        valid_count += 1
                    else:
                        skipped_files += 1
                print(f"     Successfully extracted {valid_count}/{len(files)} features")

    if not X:
        raise ValueError(" No valid audio files processed! Check paths, file formats, and permissions.")

    y = np.array(y)
    print(f"\n Dataset loading complete!")
    print(f" Total samples loaded: {len(X)} (real: {np.sum(y == 0)}, fake: {np.sum(y == 1)})")
    print(f" Skipped {skipped_files} files due to errors.")
    return np.array(X), y


def train_model(X: np.ndarray, y: np.ndarray) -> tuple[XGBClassifier, StandardScaler]:
    n_real, n_fake = np.sum(y == 0), np.sum(y == 1)
    scale_pos_weight = n_real / n_fake if n_fake > 0 else 1.0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        reg_alpha=0.5,
        reg_lambda=2.0,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42, stratify=y)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']
    cv_scores = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring, n_jobs=-1)

    print("\n Cross-validation results:")
    for metric in scoring:
        key = f'test_{metric}'
        print(f"{metric}: {np.mean(cv_scores[key]):.4f} (+/- {np.std(cv_scores[key]):.4f})")

    return model, scaler


def evaluate_model(model: XGBClassifier, scaler: StandardScaler, X_test: np.ndarray, y_test: np.ndarray) -> None:
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    print(f"\n Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(" Classification Report:\n", classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    for_base_path = r"C:\Users\Hp\OneDrive\Documents\archive"

    print(f"Base path set to: {for_base_path}")
    if not os.path.exists(for_base_path):
        print(f" ERROR: Base path does NOT exist: {for_base_path}")
        exit(1)
    else:
        print(f"Base path exists. Contents:")
        for item in os.listdir(for_base_path):
            print(f"   - {item}")

    print(" Loading dataset...")
    X, y = load_dataset(for_base_path, augment=False, max_files_per_split=100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(" Training model...")
    model, scaler = train_model(X_train, y_train)

    print(" Evaluating model...")
    evaluate_model(model, scaler, X_test, y_test)

    #  SAVE MODEL AND SCALER
    model_path = "audio_deepfake_xgb_model.pkl"
    scaler_path = "audio_feature_scaler.pkl"
    dump(model, model_path)
    dump(scaler, scaler_path)
    print(f"\n Model saved to: {model_path}")
    print(f" Scaler saved to: {scaler_path}")