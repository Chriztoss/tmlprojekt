from pathlib import Path
import csv
import math
import struct
import wave


# =========================
# INDSTILLINGER
# =========================
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR

TARGET_SR = 8000
N_SEGMENTS = 3

N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 5
N_MELS = 26

EPS = 1e-12

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

FEATURE_COLUMNS = (
    ["spectral_centroid"] +
    [f"mfcc_{i}" for i in range(1, N_MFCC + 1)] +
    [f"chroma_stft_{note}" for note in NOTE_NAMES]
)

OUTPUT_COLUMNS = FEATURE_COLUMNS + ["label", "filename", "source_file", "segment_id"]


def mean(values):
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)


def max_abs(values):
    m = 0.0
    for x in values:
        ax = abs(x)
        if ax > m:
            m = ax
    return m


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def get_wav_files(folder):
    """
    Finder .wav-filer uden at tælle dem dobbelt.
    Virker både for .wav og .WAV.
    """
    return sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".wav"
    ])


# =========================
# WAV LOAD UDEN LIBROSA
# =========================
def decode_pcm(raw_bytes, sample_width):
    """
    Konverterer PCM bytes til float samples cirka i området [-1, 1].
    Understøtter 8-bit, 16-bit, 24-bit og 32-bit PCM.
    """
    samples = []

    if sample_width == 1:
        # 8-bit WAV er normalt unsigned
        for b in raw_bytes:
            samples.append((b - 128) / 128.0)

    elif sample_width == 2:
        count = len(raw_bytes) // 2
        values = struct.unpack("<" + "h" * count, raw_bytes)
        scale = 32768.0
        for v in values:
            samples.append(v / scale)

    elif sample_width == 3:
        # 24-bit little endian signed
        for i in range(0, len(raw_bytes), 3):
            b0 = raw_bytes[i]
            b1 = raw_bytes[i + 1]
            b2 = raw_bytes[i + 2]

            value = b0 | (b1 << 8) | (b2 << 16)

            # sign extension
            if value & 0x800000:
                value -= 0x1000000

            samples.append(value / 8388608.0)

    elif sample_width == 4:
        count = len(raw_bytes) // 4
        values = struct.unpack("<" + "i" * count, raw_bytes)
        scale = 2147483648.0
        for v in values:
            samples.append(v / scale)

    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    return samples


def to_mono(samples, channels):
    """
    Laver stereo/multichannel om til mono ved gennemsnit af kanaler.
    """
    if channels == 1:
        return samples

    mono = []
    for i in range(0, len(samples), channels):
        s = 0.0
        n = 0

        for ch in range(channels):
            idx = i + ch
            if idx < len(samples):
                s += samples[idx]
                n += 1

        if n > 0:
            mono.append(s / n)

    return mono


def resample_linear(y, sr_in, sr_out):
    """
    Simpel lineær resampling.
    C-venlig, men ikke lige så god som librosa.
    """
    if sr_in == sr_out:
        return y

    if len(y) == 0:
        return []

    out_len = int(len(y) * sr_out / sr_in)
    if out_len <= 1:
        return y[:]

    result = []
    ratio = sr_in / sr_out

    for i in range(out_len):
        pos = i * ratio
        idx = int(pos)
        frac = pos - idx

        if idx >= len(y) - 1:
            result.append(y[-1])
        else:
            a = y[idx]
            b = y[idx + 1]
            result.append(a + frac * (b - a))

    return result


def trim_silence(y, top_db=25, frame_length=2048, hop_length=512):
    """
    Simpel silence trim.
    Bruger RMS pr. frame og fjerner områder langt under max RMS.
    Minder om idéen i librosa.effects.trim, men er ikke 1:1 identisk.
    """
    if len(y) <= frame_length:
        return y

    rms_values = []
    starts = []

    start = 0
    while start + frame_length <= len(y):
        frame = y[start:start + frame_length]
        power = 0.0
        for x in frame:
            power += x * x

        rms = math.sqrt(power / frame_length)
        rms_values.append(rms)
        starts.append(start)

        start += hop_length

    if len(rms_values) == 0:
        return y

    max_rms = max(rms_values)
    threshold = max_rms * (10.0 ** (-top_db / 20.0))

    active = []
    for i, rms in enumerate(rms_values):
        if rms >= threshold:
            active.append(i)

    if len(active) == 0:
        return y

    first = active[0]
    last = active[-1]

    start_sample = starts[first]
    end_sample = min(len(y), starts[last] + frame_length)

    return y[start_sample:end_sample]


def load_audio(wav_path):
    """
    Loader WAV med standardbiblioteket i stedet for librosa.
    Returnerer mono, resamplet til TARGET_SR, uden DC og normaliseret.
    """
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    samples = decode_pcm(raw, sample_width)
    y = to_mono(samples, channels)

    if sr != TARGET_SR:
        y = resample_linear(y, sr, TARGET_SR)
        sr = TARGET_SR

    # Fjern stilhed
    y = trim_silence(y, top_db=25)

    # Fjern DC
    dc = mean(y)
    y = [x - dc for x in y]

    # Normalisér
    peak = max_abs(y)
    if peak > EPS:
        y = [x / peak for x in y]

    return y, sr


def split_audio(y):
    """
    Deler én lydfil op i N_SEGMENTS dele.
    """
    n = len(y)
    segments = []

    for i in range(N_SEGMENTS):
        start = int(i * n / N_SEGMENTS)
        end = int((i + 1) * n / N_SEGMENTS)

        seg = y[start:end]

        if len(seg) < N_FFT:
            seg = seg + [0.0] * (N_FFT - len(seg))

        segments.append(seg)

    return segments


# =========================
# FFT UDEN NUMPY
# =========================
def hann_window(n):
    if n <= 1:
        return [1.0] * n

    w = []
    for i in range(n):
        w.append(0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1)))
    return w


_HANN_CACHE = {}


def get_hann(n):
    if n not in _HANN_CACHE:
        _HANN_CACHE[n] = hann_window(n)
    return _HANN_CACHE[n]


def fft_radix2(real_values):
    """
    Iterativ radix-2 FFT.
    Input: liste af real-tal.
    Output: liste af komplekse FFT bins.

    Denne er meget nemmere at oversætte til C end numpy.fft.
    """
    n = len(real_values)

    if not is_power_of_two(n):
        raise ValueError("N_FFT skal være en potens af 2 for radix-2 FFT")

    data = [complex(x, 0.0) for x in real_values]

    # Bit reversal
    j = 0
    for i in range(1, n):
        bit = n >> 1

        while j & bit:
            j ^= bit
            bit >>= 1

        j ^= bit

        if i < j:
            data[i], data[j] = data[j], data[i]

    # Butterfly stages
    length = 2
    while length <= n:
        angle = -2.0 * math.pi / length
        w_length = complex(math.cos(angle), math.sin(angle))
        half = length // 2

        for start in range(0, n, length):
            w = complex(1.0, 0.0)

            for k in range(start, start + half):
                u = data[k]
                v = data[k + half] * w

                data[k] = u + v
                data[k + half] = u - v

                w *= w_length

        length *= 2

    return data


def frame_signal(y, frame_length=N_FFT, hop_length=HOP_LENGTH):
    """
    Laver frames af signalet.
    Sidste frame paddes med nuller, hvis nødvendigt.
    """
    frames = []

    if len(y) <= frame_length:
        frames.append(y + [0.0] * (frame_length - len(y)))
        return frames

    start = 0
    while start < len(y):
        frame = y[start:start + frame_length]

        if len(frame) < frame_length:
            frame = frame + [0.0] * (frame_length - len(frame))

        frames.append(frame)
        start += hop_length

        # Undgå mange næsten tomme frames til sidst
        if start + frame_length > len(y) and len(y) - start < hop_length:
            break

    return frames


def magnitude_and_power_spectrum(frame):
    """
    Returnerer magnitude og power for positivt spektrum.
    """
    window = get_hann(len(frame))
    xw = []

    for x, w in zip(frame, window):
        xw.append(x * w)

    X = fft_radix2(xw)

    n_bins = len(frame) // 2 + 1
    magnitude = []
    power = []

    for k in range(n_bins):
        re = X[k].real
        im = X[k].imag

        p = re * re + im * im
        power.append(p)
        magnitude.append(math.sqrt(p))

    return magnitude, power


# =========================
# MFCC UDEN LIBROSA
# =========================
def hz_to_mel(freq_hz):
    return 2595.0 * math.log10(1.0 + freq_hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * ((10.0 ** (mel / 2595.0)) - 1.0)


_MEL_CACHE = {}


def mel_filterbank(sr, n_fft, n_mels):
    """
    Laver simple trekantede mel-filtre.
    Returnerer en matrix: filters[mel_index][fft_bin]
    """
    key = (sr, n_fft, n_mels)

    if key in _MEL_CACHE:
        return _MEL_CACHE[key]

    n_bins = n_fft // 2 + 1

    mel_min = hz_to_mel(0.0)
    mel_max = hz_to_mel(sr / 2.0)

    mel_points = []
    for i in range(n_mels + 2):
        mel = mel_min + i * (mel_max - mel_min) / (n_mels + 1)
        mel_points.append(mel)

    hz_points = [mel_to_hz(m) for m in mel_points]

    bin_points = []
    for f in hz_points:
        b = int(math.floor((n_fft + 1) * f / sr))
        if b < 0:
            b = 0
        if b >= n_bins:
            b = n_bins - 1
        bin_points.append(b)

    filters = []

    for m in range(1, n_mels + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]

        filt = [0.0] * n_bins

        if center > left:
            for k in range(left, center):
                filt[k] = (k - left) / (center - left)

        if right > center:
            for k in range(center, right):
                filt[k] = (right - k) / (right - center)

        filters.append(filt)

    _MEL_CACHE[key] = filters
    return filters


def apply_mel_filters(power_spectrum, sr):
    filters = mel_filterbank(sr, N_FFT, N_MELS)
    energies = []

    for filt in filters:
        e = 0.0
        for p, w in zip(power_spectrum, filt):
            e += p * w

        energies.append(math.log(e + EPS))

    return energies


def dct_type2(values, n_coeffs):
    """
    DCT-II brugt til MFCC.
    """
    n = len(values)
    coeffs = []

    for k in range(n_coeffs):
        s = 0.0

        for i, x in enumerate(values):
            angle = math.pi * k * (2 * i + 1) / (2 * n)
            s += x * math.cos(angle)

        coeffs.append(s)

    return coeffs


def mfcc_from_power(power_spectrum, sr):
    mel_log_energies = apply_mel_filters(power_spectrum, sr)
    return dct_type2(mel_log_energies, N_MFCC)


# =========================
# CHROMA UDEN LIBROSA
# =========================
def chroma_from_magnitude(magnitude, sr, n_fft):
    """
    Simpel chroma:
    Hver FFT-bin lægges i én af 12 pitch classes ud fra frekvensen.
    C = indeks 0, C# = indeks 1, ..., B = indeks 11.
    """
    chroma = [0.0] * 12

    for k in range(1, len(magnitude)):
        freq = k * sr / n_fft

        # Meget lave frekvenser er ofte ikke nyttige til akkord-klasser
        if freq < 50.0:
            continue

        # MIDI note: A4=440 Hz -> MIDI 69
        midi = int(round(69.0 + 12.0 * math.log(freq / 440.0, 2.0)))

        # MIDI 60 er C, og 60 % 12 = 0
        pitch_class = midi % 12

        chroma[pitch_class] += magnitude[k]

    # Normalisér pr. frame
    m = max(chroma)
    if m > EPS:
        chroma = [x / m for x in chroma]

    return chroma


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(y, sr):
    features = {}

    frames = frame_signal(y, N_FFT, HOP_LENGTH)

    centroid_sum = 0.0
    mfcc_sums = [0.0] * N_MFCC
    chroma_sums = [0.0] * 12

    n_frames = 0

    for frame in frames:
        magnitude, power = magnitude_and_power_spectrum(frame)

        # Spectral centroid
        mag_sum = sum(magnitude) + EPS
        centroid = 0.0

        for k, mag in enumerate(magnitude):
            freq = k * sr / N_FFT
            centroid += freq * mag

        centroid /= mag_sum
        centroid_sum += centroid

        # MFCC
        mfcc_values = mfcc_from_power(power, sr)
        for i in range(N_MFCC):
            mfcc_sums[i] += mfcc_values[i]

        # Chroma STFT
        chroma = chroma_from_magnitude(magnitude, sr, N_FFT)
        for i in range(12):
            chroma_sums[i] += chroma[i]

        n_frames += 1

    if n_frames == 0:
        n_frames = 1

    features["spectral_centroid"] = centroid_sum / n_frames

    for i in range(N_MFCC):
        features[f"mfcc_{i + 1}"] = mfcc_sums[i] / n_frames

    for i, note in enumerate(NOTE_NAMES):
        features[f"chroma_stft_{note}"] = chroma_sums[i] / n_frames

    return features


# =========================
# DATASET BUILD UDEN PANDAS
# =========================
def build_dataset():
    rows = []
    class_counts = {}

    for class_dir in sorted(DATASET_PATH.iterdir()):
        if not class_dir.is_dir():
            continue

        if class_dir.name.startswith(".") or class_dir.name == "src":
            continue

        wav_files = get_wav_files(class_dir)

        if len(wav_files) == 0:
            continue

        label = class_dir.name
        print(f"{label}: {len(wav_files)} filer")

        for wav_file in wav_files:
            y, sr = load_audio(wav_file)
            segments = split_audio(y)

            for segment_id, segment in enumerate(segments):
                features = extract_features(segment, sr)

                features["label"] = label
                features["filename"] = f"{wav_file.stem}_seg{segment_id}.wav"
                features["source_file"] = f"{label}/{wav_file.name}"
                features["segment_id"] = segment_id

                rows.append(features)
                class_counts[label] = class_counts.get(label, 0) + 1

    return rows, class_counts


def save_csv(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    print("\n===== BYGGER 18-FEATURE DATASET UDEN LIBROSA/NUMPY/PANDAS =====")

    rows, class_counts = build_dataset()

    if len(rows) == 0:
        raise ValueError(
            "Ingen WAV-filer fundet. Tjek at akkordmapperne ligger direkte i projektmappen."
        )

    output_path = PROJECT_DIR / "feature_dataset_2.csv"
    save_csv(rows, output_path)

    print("\nFeature-datasæt gemt:")
    print(output_path)

    print("\nAntal samples:", len(rows))

    print("\nKlasser:")
    for label in sorted(class_counts):
        print(f"{label}: {class_counts[label]}")


if __name__ == "__main__":
    main()
