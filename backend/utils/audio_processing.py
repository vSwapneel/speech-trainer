import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

def extract_voice_dimensions(file_path):

    # ----------------- Load sound -----------------
    snd = parselmouth.Sound(file_path)
    duration = snd.get_total_duration()
    
    # Librosa for Pause related analysis 
    y, sr = librosa.load(file_path, sr=16000)

    # ----------------- Pitch -----------------
    pitch = snd.to_pitch(time_step=0.01)
    pitch_mean = call(pitch, "Get mean", 0, 0, "Hertz")
    pitch_sd = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    pitch_variance = pitch_sd ** 2
    min_pitch = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    max_pitch = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

    # ----------------- Jitter & Shimmer -----------------
    point_process = call(snd, "To PointProcess (periodic, cc)", min_pitch, max_pitch)

    jitter_local = call(
        point_process,
        "Get jitter (local)",
        0, 0, 0.0001, 0.02, 1.3
    )

    shimmer_local = call(
        [snd, point_process],
        "Get shimmer (local)",
        0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    # ----------------- Energy (Intensity) -----------------
    intensity = snd.to_intensity()
    energy_mean = call(intensity, "Get mean", 0, 0)  # dB value

    # ----------------- Pauses via TextGrid (silences) -----------------
    intervals = librosa.effects.split(y, top_db=30)
    pause_durations = []
    for i in range(1, len(intervals)):
        pause = (intervals[i][0] - intervals[i-1][1]) / sr
        pause_durations.append(pause)
    pause_count = len(pause_durations)
    avg_pause_duration = float(np.mean(pause_durations)) if pause_durations else 0
    
    # ----------------- Nasality (Praat approach via formants) -----------------
    # Skipped Nasaility for now
    

    # ----------------- Intonation variability -----------------
    intonation_variance = pitch_variance  # same as pitch variance

    # ----------------- Collect metrics -----------------
    return {
        "pitch_mean": pitch_mean,
        "pitch_variance": pitch_variance,
        "pitch_min": min_pitch,
        "pitch_max": max_pitch,
        "shakiness_jitter": jitter_local,
        "shimmer": shimmer_local,
        "energy": energy_mean,
        "pause_count": pause_count,
        "avg_pause_duration": float(avg_pause_duration),
        "intonation_variance": intonation_variance,
        "duration": duration
    }


if __name__ == "__main__":
    file = "../../datasets/sample_tests/F_0101_10y4m_1.wav"
    metrics = extract_voice_dimensions(file)
    for k, v in metrics.items():
        print(f"{k}: {v}")
