import os
import threading
import time
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import sounddevice as sd
import librosa
from scipy.ndimage import maximum_filter, generate_binary_structure, binary_erosion

app = Flask(__name__)

# Basic configuration
REFERENCE_FILE = "reference_fingerprint.npz"

SAMPLE_RATE = 22050          # Audio sampling rate
REFERENCE_SECONDS = 5.0      # How long we record the ringtone during setup
WINDOW_SECONDS = 1           # How much audio we analyze on each detection pass

COOLDOWN_SECONDS = 8         # After the siren fires, wait this long before listening again

# Spectrogram / fingerprint settings
N_FFT = 2048
HOP_LENGTH = 512
PEAK_AMP_MIN_DB = -40.0      # Only consider strong peaks everything below this is ignored

FAN_VALUE = 5                # For each detected peak, link it to several nearby peaks
MIN_TIME_DELTA = 1           # Ignore peak pairs that are too close together in time
MAX_TIME_DELTA = 60          # Also ignore pairs that are too far apart

# Detection threshold settings (percentage-based)
# require a minimum number of matching hashes, AND
# a certain percent of the reference fingerprint, adjusted based on
# how long our detection window is compared to the reference.
MIN_ABS_MATCHES = 10
MIN_MATCH_FRACTION = 0.25    # Use at least 25% of the reference fingerprint as a baseline

# Loudness checks
MIN_RMS_RATIO = 0.5          # Recording must be at least half as loud as the original reference
MAX_RMS_RATIO = 3.0          # But not absurdly louder either

# Runtime state
detector_thread = None
detector_running = threading.Event()

reference_hashes = None       # Where the fingerprint lives once recorded
reference_rms = None          # Loudness of the reference recording

state_lock = threading.Lock()

siren_playing = threading.Event()

# Audio helpers
def record_audio(seconds: float, samplerate: int = SAMPLE_RATE) -> np.ndarray:
    """Record audio from the microphone for the requested amount of time."""
    frames = int(seconds * samplerate)
    rec = sd.rec(frames, samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    return rec.flatten()

def compute_rms(audio: np.ndarray) -> float:
    """Measure how loud the audio is (root mean square)."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))

def compute_spectrogram_db(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Turn raw audio into a dB-based spectrogram.
    Rows = frequency bins, columns = time slices.
    """
    S = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
    return S_db

# Peak detection + fingerprinting
def find_peaks(S_db: np.ndarray, amp_min_db: float):
    """
    Identify prominent peaks in the spectrogram.
    We only keep points that really stand out from their neighborhood.
    """
    # Mask out quiet parts of the spectrogram
    mask_amp = S_db >= amp_min_db
    S_thresh = np.where(mask_amp, S_db, 0.0)

    # Define what counts as a local neighborhood
    struct = generate_binary_structure(2, 1)

    # A peak is where the value is equal to the max of its neighborhood
    local_max = maximum_filter(S_thresh, footprint=struct) == S_thresh

    # Build a background mask so I don’t accidentally treat silence as peaks
    background = (S_thresh == 0)
    eroded_background = binary_erosion(background, structure=struct, border_value=1)

    # True peaks are local maxima that aren’t part of the background
    detected_peaks = local_max ^ eroded_background

    freq_idx, time_idx = np.where(detected_peaks)
    peaks = list(zip(time_idx, freq_idx))  # (time, frequency)
    return peaks

def generate_hashes_from_peaks(
    peaks,
    fan_value=FAN_VALUE,
    min_time_delta=MIN_TIME_DELTA,
    max_time_delta=MAX_TIME_DELTA,
):
    """
    Convert detected peaks into hash triples.
    Each hash encodes the relationship between two peaks:
    (frequency1, frequency2, time_difference).
    """
    if not peaks:
        return []

    peaks_sorted = sorted(peaks, key=lambda x: (x[0], x[1]))
    hashes = []

    n_peaks = len(peaks_sorted)
    for i in range(n_peaks):
        t1, f1 = peaks_sorted[i]

        # Link this peak to several upcoming peaks to form fingerprints
        for j in range(1, fan_value + 1):
            if i + j >= n_peaks:
                break

            t2, f2 = peaks_sorted[i + j]
            dt = t2 - t1

            # Ignore unreasonable peak pairings
            if dt < min_time_delta or dt > max_time_delta:
                continue

            hashes.append((int(f1), int(f2), int(dt)))

    return hashes

def fingerprint_audio(audio: np.ndarray, sr: int = SAMPLE_RATE):
    """
    Generate the full fingerprint for an audio clip.
    This gives us:
      - a list of peak-based hashes,
      - the loudness,
      - and how many peaks we found.
    """
    S_db = compute_spectrogram_db(audio, sr)
    peaks = find_peaks(S_db, PEAK_AMP_MIN_DB)
    hashes = generate_hashes_from_peaks(peaks)
    rms = compute_rms(audio)
    return hashes, rms, len(peaks)

# Reference load/save
def load_reference():
    """Load the saved fingerprint from disk, if it exists."""
    global reference_hashes, reference_rms

    if not os.path.exists(REFERENCE_FILE):
        reference_hashes = None
        reference_rms = None
        print("[Reference] No stored fingerprint found yet.")
        return

    try:
        data = np.load(REFERENCE_FILE)
        hashes_arr = data["hashes"]
        reference_rms = float(data["rms"])
        reference_hashes = set(tuple(row.astype(int)) for row in hashes_arr)

        print(
            f"[Reference] Loaded fingerprint: "
            f"{len(reference_hashes)} hashes, RMS={reference_rms:.4f}"
        )
    except Exception as e:
        print(f"[Reference] Failed to load fingerprint: {e}")
        reference_hashes = None
        reference_rms = None

def save_reference(hashes, rms: float):
    """Write the fingerprint and RMS to disk so we can use it later."""
    global reference_hashes, reference_rms

    hashes_arr = np.array(hashes, dtype=np.int32)
    np.savez(REFERENCE_FILE, hashes=hashes_arr, rms=rms)

    reference_hashes = set(tuple(row) for row in hashes_arr)
    reference_rms = rms

    ref_size = len(reference_hashes)

    if ref_size > 0:
        window_fraction = min(1.0, WINDOW_SECONDS / REFERENCE_SECONDS)
        needed = max(
            MIN_ABS_MATCHES,
            int(ref_size * MIN_MATCH_FRACTION * window_fraction),
        )
    else:
        needed = MIN_ABS_MATCHES

    print(
        f"[Reference] Saved fingerprint: {ref_size} hashes, RMS={reference_rms:.4f}, "
        f"required matches ≈ {needed}"
    )

# Siren playback
def play_siren():
    """Play an air-raid style rising and falling siren until the user stops it."""
    fs = 44100
    total_duration = 12.0
    cycle_duration = 3.0

    # Time axis for the full siren
    t = np.linspace(0, total_duration, int(fs * total_duration), endpoint=False)

    # Build a repeating triangle wave (0→1→0)
    cycle_pos = (t % cycle_duration) / cycle_duration
    tri = 1.0 - np.abs(2.0 * cycle_pos - 1.0)

    # Sweep between two frequencies based on that triangle wave
    f_low = 400.0
    f_high = 1400.0
    freq = f_low + tri * (f_high - f_low)

    # Integrate frequency into a phase signal and build the actual tone
    phase = 2.0 * np.pi * np.cumsum(freq) / fs
    tone = 0.9 * np.sin(phase).astype("float32")

    siren_playing.set()
    sd.play(tone, fs)

    # Poll for stop requests so it's responsive to UI actions
    for _ in range(60):
        if not siren_playing.is_set():
            sd.stop()
            return
        time.sleep(0.2)

    sd.wait()
    siren_playing.clear()

@app.route("/stop_siren", methods=["POST"])
def stop_siren():
    siren_playing.clear()
    sd.stop()
    return redirect(url_for("index", msg="Siren stopped."))

# Detection loop
def detector_loop():
    """Main loop that continuously listens for your ringtone."""
    while detector_running.is_set():
        with state_lock:
            ref_hashes = reference_hashes
            ref_rms = reference_rms

        if not ref_hashes or ref_rms is None:
            time.sleep(1.0)
            continue

        ref_size = len(ref_hashes)
        window_fraction = min(1.0, WINDOW_SECONDS / REFERENCE_SECONDS)
        needed = max(
            MIN_ABS_MATCHES,
            int(ref_size * MIN_MATCH_FRACTION * window_fraction),
        )

        try:
            audio = record_audio(WINDOW_SECONDS, SAMPLE_RATE)
        except Exception as e:
            print(f"[Detector] Microphone error: {e}")
            time.sleep(1.0)
            continue

        try:
            hashes, rms, n_peaks = fingerprint_audio(audio, SAMPLE_RATE)

            # Make sure the sound level is roughly in the same ballpark as the reference
            if ref_rms > 0:
                if rms < ref_rms * MIN_RMS_RATIO or rms > ref_rms * MAX_RMS_RATIO:
                    print(
                        f"[Detector] Ignoring: RMS={rms:.4f} (ref={ref_rms:.4f})"
                    )
                    time.sleep(0.2)
                    continue

            window_hashes = set(hashes)
            matches = ref_hashes.intersection(window_hashes)
            match_count = len(matches)

            print(
                f"[Detector] peaks={n_peaks}, hashes={len(window_hashes)}, "
                f"matches={match_count}/{needed}, RMS={rms:.4f}"
            )
        except Exception as e:
            print(f"[Detector] Error analyzing audio: {e}")
            time.sleep(1.0)
            continue

        # If enough hashes line up treat it as ringtone
        if match_count >= needed:
            print(
                f"[Detector] Match found ({match_count} >= {needed}). Triggering siren..."
            )
            try:
                play_siren()
            except Exception as e:
                print(f"[Detector] Could not play siren: {e}")

            time.sleep(COOLDOWN_SECONDS)
        else:
            time.sleep(0.2)

# Flask routes
@app.route("/")
def index():
    with state_lock:
        ref_loaded = reference_hashes is not None and len(reference_hashes) > 0
        running = detector_running.is_set()
        siren = siren_playing.is_set()

        if ref_loaded:
            ref_size = len(reference_hashes)
            window_fraction = min(1.0, WINDOW_SECONDS / REFERENCE_SECONDS)
            required_matches = max(
                MIN_ABS_MATCHES,
                int(ref_size * MIN_MATCH_FRACTION * window_fraction),
            )
        else:
            required_matches = MIN_ABS_MATCHES

    return render_template(
        "index.html",
        reference_loaded=ref_loaded,
        detector_running=running,
        siren_active=siren,
        message=request.args.get("msg", ""),
        similarity_threshold=float(required_matches),
        ref_seconds=REFERENCE_SECONDS,
    )

@app.route("/record_reference", methods=["POST"])
def record_reference_route():
    """Capture a fresh sample of your ringtone and update the stored fingerprint."""
    try:
        audio = record_audio(REFERENCE_SECONDS, SAMPLE_RATE)
        hashes, rms, n_peaks = fingerprint_audio(audio, SAMPLE_RATE)

        if not hashes:
            msg = (
                "No usable fingerprint detected. Try again with the phone closer "
                "to the microphone and at full volume."
            )
            return redirect(url_for("index", msg=msg))

        save_reference(hashes, rms)

        msg = (
            f"Saved fingerprint: {len(hashes)} hashes from {n_peaks} peaks. "
            f"RMS={rms:.4f}"
        )
        return redirect(url_for("index", msg=msg))
    except Exception as e:
        return redirect(url_for("index", msg=f"Error while recording reference: {e}"))

@app.route("/start", methods=["POST"])
def start_detector():
    global detector_thread

    with state_lock:
        if not reference_hashes:
            return redirect(url_for("index", msg="Record a reference first."))

        if detector_running.is_set():
            return redirect(url_for("index", msg="Detector already running."))

        detector_running.set()
        detector_thread = threading.Thread(target=detector_loop, daemon=True)
        detector_thread.start()

    return redirect(url_for("index", msg="On-call detection started."))

@app.route("/stop", methods=["POST"])
def stop_detector():
    detector_running.clear()
    return redirect(url_for("index", msg="On-call detection stopped."))

if __name__ == "__main__":
    load_reference()
    app.run(host="127.0.0.1", port=5000, debug=True)
