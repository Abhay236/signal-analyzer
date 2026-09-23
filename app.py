import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io.wavfile import write
from io import BytesIO


st.set_page_config(
    page_title="Signal Lab | Generator & Analyzer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

FS_FIXED = 44100

# -------------------------
# Styling — UI only
# -------------------------
st.markdown(
    r"""
    <style>
    :root {
        --bg: #0b0f16;
        --panel: #111722;
        --line: #273143;
        --text: #f4f7fb;
        --muted: #9ca8ba;
        --accent: #8b7cff;
        --accent-2: #34d6d0;
        --accent-soft: rgba(139, 124, 255, 0.12);
    }

    .stApp {
        background:
            radial-gradient(circle at 82% 4%, rgba(139,124,255,0.08), transparent 25%),
            radial-gradient(circle at 0% 90%, rgba(52,214,208,0.045), transparent 22%),
            var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1420px;
        padding-top: 1.45rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: #090d14;
        border-right: 1px solid #202938;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 1rem 0.95rem 1.3rem 0.95rem;
    }

    .brand-kicker {
        color: var(--accent-2);
        font-size: 0.65rem;
        font-weight: 850;
        letter-spacing: 0.20em;
        text-transform: uppercase;
        margin-bottom: 0.22rem;
    }

    .hero-title {
        font-size: clamp(2.0rem, 4vw, 3.15rem);
        line-height: 1.03;
        font-weight: 850;
        letter-spacing: -0.045em;
        color: var(--text);
        margin: 0;
    }

    .hero-title span { color: var(--accent); }

    .hero-subtitle {
        margin-top: 0.45rem;
        color: var(--muted);
        font-size: 0.92rem;
        max-width: 760px;
        line-height: 1.55;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.42rem;
        margin-top: 0.8rem;
        margin-bottom: 1.1rem;
    }

    .badge {
        border: 1px solid var(--line);
        background: rgba(17,23,34,0.75);
        color: #cbd3e0;
        border-radius: 999px;
        padding: 0.27rem 0.60rem;
        font-size: 0.70rem;
        font-weight: 700;
    }

    .badge.accent {
        border-color: rgba(139,124,255,0.45);
        color: #d9d4ff;
        background: var(--accent-soft);
    }

    .badge.live {
        border-color: rgba(52,214,208,0.38);
        color: #aaf3ef;
        background: rgba(52,214,208,0.07);
    }

    .section-kicker {
        color: #7f8da3;
        font-size: 0.63rem;
        font-weight: 850;
        letter-spacing: 0.19em;
        text-transform: uppercase;
        margin-bottom: 0.14rem;
    }

    .section-title {
        color: var(--text);
        font-size: 1.22rem;
        font-weight: 800;
        letter-spacing: -0.018em;
        margin-bottom: 0.62rem;
    }

    .panel {
        background: linear-gradient(180deg, rgba(21,28,41,0.94), rgba(14,20,30,0.97));
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.95rem 1rem 0.95rem 1rem;
        box-shadow: 0 12px 32px rgba(0,0,0,0.14);
    }

    .small-panel {
        background: rgba(17,23,34,0.7);
        border: 1px solid #222c3b;
        border-radius: 12px;
        padding: 0.72rem 0.8rem;
    }

    .status-line {
        display: flex;
        align-items: center;
        gap: 0.52rem;
        color: #bcc6d5;
        font-size: 0.78rem;
        margin: 0.65rem 0 0.82rem;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent-2);
        box-shadow: 0 0 12px rgba(52,214,208,0.52);
        display: inline-block;
    }

    .pipeline {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.34rem;
        margin: 0.72rem 0 1.05rem;
    }

    .pipeline-step {
        border: 1px solid #303b4e;
        background: #0f1520;
        color: #cbd3df;
        border-radius: 999px;
        padding: 0.27rem 0.58rem;
        font-size: 0.67rem;
        font-weight: 800;
        letter-spacing: 0.05em;
    }

    .pipeline-arrow { color: #66748a; font-size: 0.78rem; }

    .info-card {
        background: linear-gradient(180deg, rgba(17,23,34,0.95), rgba(14,20,30,0.95));
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 0.72rem 0.78rem;
        min-height: 76px;
    }

    .info-label {
        color: #8c98aa;
        font-size: 0.66rem;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        font-weight: 800;
    }

    .info-value {
        color: var(--text);
        font-size: 1.06rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    .info-help { color: #7f8a9b; font-size: 0.67rem; margin-top: 0.07rem; }

    .filter-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.32rem;
        background: rgba(139,124,255,0.11);
        color: #d2ccff;
        border: 1px solid rgba(139,124,255,0.30);
        border-radius: 999px;
        padding: 0.24rem 0.56rem;
        font-size: 0.70rem;
        font-weight: 800;
        margin-bottom: 0.65rem;
    }

    .sidebar-card {
        background: linear-gradient(180deg, #101722, #0d131d);
        border: 1px solid #222d3d;
        border-radius: 13px;
        padding: 0.72rem 0.78rem;
        margin: 0.55rem 0 0.78rem;
    }

    .sidebar-card .title { color:#f0f3f8; font-weight:800; font-size:0.81rem; margin-bottom:0.12rem; }
    .sidebar-card .text { color:#8e9aac; font-size:0.70rem; line-height:1.5; }

    div[data-baseweb="radio"] > div { gap: 0.30rem; flex-wrap: wrap; }

    div[data-baseweb="radio"] label {
        color: #cdd5e1 !important;
        background: #101722;
        border: 1px solid #253043;
        border-radius: 999px;
        padding: 0.31rem 0.66rem;
    }

    div[data-baseweb="radio"] label:has(input:checked) {
        background: rgba(139,124,255,0.13);
        border-color: rgba(139,124,255,0.48);
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #313b4e;
        background: #141b27;
        color: #eef2f7;
        font-weight: 750;
        min-height: 2.42rem;
    }

    .stButton > button:hover { border-color:#6558e8; color:#fff; }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #786aff, #5d55dd);
        border-color: transparent;
        color: white;
        box-shadow: 0 7px 20px rgba(98,86,221,0.22);
    }

    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #313b4e;
        background: #141b27;
        font-weight: 750;
    }

    [data-testid="stExpander"] { background:rgba(17,23,34,0.50); border:1px solid #222c3b; border-radius:12px; }
    .footer { border-top:1px solid #202938; margin-top:1.8rem; padding-top:0.7rem; color:#6f7c8f; font-size:0.66rem; text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Signal generation
# -------------------------
def generate_sine(A, f, Fs, N):
    t = np.arange(N) / Fs
    x = A * np.sin(2 * np.pi * f * t)
    return t, x


def generate_square(A, f, Fs, N, duty_cycle):
    t = np.arange(N) / Fs
    x = A * signal.square(2 * np.pi * f * t, duty=duty_cycle)
    return t, x


def generate_triangle(A, f, Fs, N):
    t = np.arange(N) / Fs
    x = A * signal.sawtooth(2 * np.pi * f * t, width=0.5)
    return t, x


def generate_chirp(A, f0, f1, Fs, N):
    t = np.arange(N) / Fs
    t1 = t[-1] if N > 1 else 0.0
    x = A * signal.chirp(t, f0=f0, f1=f1, t1=t1, method="linear")
    return t, x


def generate_sinc(A, f, Fs, N):
    duration = N / Fs
    t = np.linspace(-duration / 2, duration / 2, N, endpoint=False)
    x = A * np.sinc(2 * f * t)
    return t, x

# -------------------------
# Analysis
# -------------------------
def calculate_fft(x, Fs):
    N = len(x)
    X = np.fft.rfft(x)
    magnitude = np.abs(X) / N
    if N > 1:
        if N % 2 == 0:
            magnitude[1:-1] *= 2
        else:
            magnitude[1:] *= 2
    freq = np.fft.rfftfreq(N, 1 / Fs)
    return freq, magnitude


def calculate_stft(x, Fs, window_size, overlap):
    effective_window = min(int(window_size), len(x))
    effective_window = max(2, effective_window)
    effective_overlap = min(int(overlap), effective_window - 1)
    f_stft, t_stft, Zxx = signal.stft(
        x,
        Fs,
        nperseg=effective_window,
        noverlap=effective_overlap,
    )
    return f_stft, t_stft, Zxx

# -------------------------
# Filters
# -------------------------
def apply_filter(x, Fs, filter_type, cutoff_low, cutoff_high, order):
    nyquist = Fs / 2
    order = int(order)

    if filter_type == "None":
        return x.copy()

    if filter_type in ["Low-Pass", "High-Pass"]:
        if not (0 < cutoff_low < nyquist):
            raise ValueError("Cut-off frequency must be between 0 and the Nyquist frequency.")
        btype = "low" if filter_type == "Low-Pass" else "high"
        b, a = signal.butter(order, cutoff_low, btype=btype, fs=Fs)

    else:
        if not (0 < cutoff_low < cutoff_high < nyquist):
            raise ValueError("Band-pass frequencies must satisfy 0 < low < high < Nyquist frequency.")
        b, a = signal.butter(order, [cutoff_low, cutoff_high], btype="band", fs=Fs)

    return signal.filtfilt(b, a, x)

# -------------------------
# Audio helpers
# -------------------------
def signal_to_wav_bytes(x, Fs):
    audio = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak
    audio_int16 = np.int16(np.clip(audio, -1, 1) * 32767)
    buffer = BytesIO()
    write(buffer, int(Fs), audio_int16)
    return buffer.getvalue()


def signal_to_csv_bytes(t, x):
    data = np.column_stack((t, x))
    return data.tobytes() if False else ("Time_s,Amplitude\n" + "\n".join(f"{a:.10g},{b:.10g}" for a, b in data)).encode()

# -------------------------
# Session state
# -------------------------
if "signal_generated" not in st.session_state:
    st.session_state.signal_generated = False
if "filter_applied" not in st.session_state:
    st.session_state.filter_applied = False
if "filter_type" not in st.session_state:
    st.session_state.filter_type = "None"

# -------------------------
# Sidebar — signal setup
# -------------------------
with st.sidebar:
    st.markdown('<div class="brand-kicker">DSP WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.16rem;font-weight:800;color:#f4f7fb;">Signal Lab</div>', unsafe_allow_html=True)
    st.caption("A clean workspace for generating and inspecting signals.")

    st.markdown("### 01 · Generate")
    waveform = st.selectbox(
        "Signal",
        ["Sine", "Square", "Triangle", "Chirp", "Sinc"],
    )

    amplitude = st.number_input(
        "Amplitude",
        min_value=0.1,
        value=1.0,
        step=0.1,
    )

    if waveform in ["Sine", "Square", "Triangle"]:
        frequency = st.number_input(
            "Frequency (Hz)",
            min_value=1.0,
            max_value=float(FS_FIXED / 2 - 1),
            value=1000.0 if waveform != "Triangle" else 1200.0,
            step=100.0,
        )
    elif waveform == "Chirp":
        start_frequency = st.number_input(
            "Starting Frequency (Hz)",
            min_value=1.0,
            max_value=float(FS_FIXED / 2 - 1),
            value=300.0,
            step=100.0,
        )
        end_frequency = st.number_input(
            "Ending Frequency (Hz)",
            min_value=1.0,
            max_value=float(FS_FIXED / 2 - 1),
            value=3000.0,
            step=100.0,
        )
    else:
        sinc_frequency = st.number_input(
            "Sinc Frequency Parameter (Hz)",
            min_value=1.0,
            max_value=float(FS_FIXED / 2 - 1),
            value=1500.0,
            step=100.0,
        )

    if waveform == "Square":
        duty_cycle = st.number_input(
            "Duty Cycle (%)",
            min_value=1.0,
            max_value=99.0,
            value=50.0,
            step=1.0,
        )

    duration = st.number_input(
        "Time / Duration (seconds)",
        min_value=0.001,
        value=1.0,
        step=0.1,
    )

    Fs = FS_FIXED
    N_preview = max(1, int(round(Fs * duration)))
    max_plot_samples = min(N_preview, 50000)

    st.markdown(
        f'''<div class="sidebar-card">
        <div class="title">Sampling locked</div>
        <div class="text">44.1 kHz fixed · Nyquist 22.05 kHz · 16-bit WAV export</div>
        </div>''',
        unsafe_allow_html=True,
    )

    with st.expander("Display", expanded=False):
        plot_samples_input = st.number_input(
            "Samples to Plot",
            min_value=10,
            max_value=max_plot_samples,
            value=min(3000, max_plot_samples),
            step=100,
            format="%d",
            help="Controls only how many samples are shown in the time-domain plot.",
        )
        st.caption(f"Total generated samples: N = Fs × T = {N_preview:,}")

    generate_button = st.button(
        "⚡ Generate Signal",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    st.markdown("### 02 · Conditioning")

    filter_type = st.selectbox(
        "Filter",
        ["None", "Low-Pass", "High-Pass", "Band-Pass"],
    )

    if filter_type in ["Low-Pass", "High-Pass"]:
        cutoff_low = st.number_input(
            "Cut-off Frequency (Hz)",
            min_value=1.0,
            max_value=float(FS_FIXED / 2 - 1),
            value=1500.0,
            step=100.0,
        )
        cutoff_high = 0.0
    elif filter_type == "Band-Pass":
        cutoff_low = st.number_input(
            "Lower Cut-off (Hz)",
            min_value=1.0,
            max_value=float(FS_FIXED / 2 - 2),
            value=1000.0,
            step=100.0,
        )
        cutoff_high = st.number_input(
            "Upper Cut-off (Hz)",
            min_value=2.0,
            max_value=float(FS_FIXED / 2 - 1),
            value=3000.0,
            step=100.0,
        )
    else:
        cutoff_low = 0.0
        cutoff_high = 0.0

    filter_order = st.number_input(
        "Filter Order",
        min_value=1,
        max_value=12,
        value=4,
        step=1,
    )

    apply_filter_button = st.button(
        "Apply Filter",
        use_container_width=True,
        disabled=not st.session_state.signal_generated or filter_type == "None",
    )

    with st.expander("STFT Settings", expanded=False):
        window_size = st.selectbox(
            "Window Size",
            [256, 512, 1024, 2048, 4096],
            index=2,
        )
        overlap_options = [v for v in [0, 128, 256, 512, 1024, 2048] if v < window_size]
        overlap = st.selectbox("Overlap", overlap_options)

    if st.button("Reset Workspace", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# -------------------------
# Generate — same signal logic
# -------------------------
if generate_button:
    Fs_value = int(Fs)
    duration_value = float(duration)
    N_value = max(1, int(round(Fs_value * duration_value)))
    nyquist = Fs_value / 2

    try:
        if waveform == "Sine":
            if frequency >= nyquist:
                raise ValueError("Frequency must be less than the Nyquist frequency.")
            t, x = generate_sine(amplitude, frequency, Fs_value, N_value)

        elif waveform == "Square":
            if frequency >= nyquist:
                raise ValueError("Frequency must be less than the Nyquist frequency.")
            t, x = generate_square(amplitude, frequency, Fs_value, N_value, duty_cycle / 100)

        elif waveform == "Triangle":
            if frequency >= nyquist:
                raise ValueError("Frequency must be less than the Nyquist frequency.")
            t, x = generate_triangle(amplitude, frequency, Fs_value, N_value)

        elif waveform == "Chirp":
            if start_frequency >= nyquist or end_frequency >= nyquist:
                raise ValueError("Starting and ending frequencies must be less than the Nyquist frequency.")
            if end_frequency <= start_frequency:
                raise ValueError("Ending frequency must be greater than starting frequency.")
            t, x = generate_chirp(amplitude, start_frequency, end_frequency, Fs_value, N_value)

        else:
            if sinc_frequency >= nyquist:
                raise ValueError("Sinc frequency parameter must be less than the Nyquist frequency.")
            t, x = generate_sinc(amplitude, sinc_frequency, Fs_value, N_value)

        st.session_state.signal_generated = True
        st.session_state.filter_applied = False
        st.session_state.t_value = t
        st.session_state.x_value = x
        st.session_state.Fs_value = Fs_value
        st.session_state.N_value = N_value
        st.session_state.duration_value = N_value / Fs_value
        st.session_state.plot_samples_value = min(int(plot_samples_input), N_value)
        st.session_state.waveform_value = waveform
        st.session_state.window_size_value = int(window_size)
        st.session_state.overlap_value = int(overlap)
        st.session_state.filter_type = "None"

    except Exception as exc:
        st.error(str(exc))

# -------------------------
# Apply filter — same filter logic
# -------------------------
if apply_filter_button:
    try:
        y = apply_filter(
            st.session_state.x_value,
            st.session_state.Fs_value,
            filter_type,
            cutoff_low,
            cutoff_high,
            int(filter_order),
        )
        st.session_state.y_value = y
        st.session_state.filter_applied = True
        st.session_state.filter_type = filter_type
        st.session_state.cutoff_low_value = float(cutoff_low)
        st.session_state.cutoff_high_value = float(cutoff_high)
        st.session_state.order_value = int(filter_order)
    except Exception as exc:
        st.error(f"Filtering error: {exc}")

# -------------------------
# Main workspace
# -------------------------
st.markdown('<div class="brand-kicker">DIGITAL SIGNAL PROCESSING LABORATORY</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Signal Generator <span>&amp;</span> Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Generate, condition and inspect signals in one focused workspace. The sampling rate is fixed at 44.1 kHz throughout the application.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '''<div class="badge-row">
        <span class="badge accent">Fs = 44.1 kHz fixed</span>
        <span class="badge">Nyquist = 22.05 kHz</span>
    </div>''',
    unsafe_allow_html=True,
)

if not st.session_state.signal_generated:
    st.markdown(
        '''<div class="panel">
        <div class="section-kicker"> </div>
        <div class="section-title">Generate Signal</div>
        <div style="color:#9ca8ba;font-size:0.88rem;line-height:1.6;max-width:760px;">
        Select a waveform and its parameters in the left panel, then click <b>Generate Signal</b>.
        The analysis workspace will appear after generation.
        </div>
        <div class="pipeline">
            <span class="pipeline-step">01 GENERATE</span><span class="pipeline-arrow">→</span>
            <span class="pipeline-step">02 CONDITION</span><span class="pipeline-arrow">→</span>
            <span class="pipeline-step">03 ANALYZE</span><span class="pipeline-arrow">→</span>
            <span class="pipeline-step">04 EXPORT</span>
        </div>
        </div>''',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="footer">Signal Lab · Fixed sampling frequency = 44.1 kHz</div>', unsafe_allow_html=True)
    st.stop()

x = st.session_state.x_value
t = st.session_state.t_value
Fs_value = st.session_state.Fs_value
waveform_value = st.session_state.waveform_value
N_value = st.session_state.N_value

processed = st.session_state.get("filter_applied", False)
y = st.session_state.get("y_value", x)
analysis_signal = y if processed else x

# Signal snapshot
st.markdown('<div class="section-kicker">01 · SIGNAL SNAPSHOT</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Current experiment</div>', unsafe_allow_html=True)

snapshot_cols = st.columns(6)
snapshot_data = [
    ("Waveform", waveform_value, "source"),
    ("Samples", f"{N_value:,}", "sample count"),
    ("Duration", f"{N_value / Fs_value:.3f} s", "record length"),
    ("Sampling", "44.1 kHz", "fixed"),
    ("Peak", f"{np.max(np.abs(analysis_signal)):.4f}", "absolute"),
    ("RMS", f"{np.sqrt(np.mean(analysis_signal ** 2)):.4f}", "current signal"),
]
for col, (label, value, help_text) in zip(snapshot_cols, snapshot_data):
    with col:
        st.markdown(
            f'''<div class="info-card">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}</div>
            <div class="info-help">{help_text}</div>
            </div>''',
            unsafe_allow_html=True,
        )

pipeline_parts = ["SOURCE", waveform_value.upper(), "DSP PIPELINE"]
if processed:
    pipeline_parts.append(st.session_state.filter_type.upper())
pipeline_parts.append("OUTPUT")

pipeline_html = ['<div class="pipeline">']
for idx, part in enumerate(pipeline_parts):
    if idx:
        pipeline_html.append('<span class="pipeline-arrow">→</span>')
    pipeline_html.append(f'<span class="pipeline-step">{part}</span>')
pipeline_html.append('</div>')
st.markdown("".join(pipeline_html), unsafe_allow_html=True)

if processed:
    st.markdown(
        f'<div class="filter-badge">✓ {st.session_state.filter_type} · Order {st.session_state.order_value}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="status-line"><span class="status-dot"></span>Generated signal ready for analysis.</div>', unsafe_allow_html=True)

# Analysis navigation
st.markdown('<div class="section-kicker">02 · ANALYSIS</div>', unsafe_allow_html=True)
analysis_mode = st.radio(
    "Analysis View",
    ["Time + FFT", "Original vs Processed", "STFT Spectrogram", "Playback + Export"],
    horizontal=True,
    label_visibility="collapsed",
)

samples_to_plot = min(
    int(st.session_state.get("plot_samples_value", min(3000, N_value))),
    len(x),
)

# -------------------------
# Time + FFT
# -------------------------
if analysis_mode == "Time + FFT":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Time-domain waveform &amp; FFT spectrum</div>', unsafe_allow_html=True)
    st.caption(f"Displaying {samples_to_plot:,} samples in the time-domain view. The full signal is retained for export.")

    left, right = st.columns(2, gap="large")
    samples_for_plot = max(2, min(samples_to_plot, len(analysis_signal)))

    if waveform_value == "Sinc":
        center = len(analysis_signal) // 2
        half = samples_for_plot // 2
        start = max(0, center - half)
        end = min(len(analysis_signal), center + half)
        t_plot = t[start:end]
        x_plot = analysis_signal[start:end]
    else:
        t_plot = t[:samples_for_plot]
        x_plot = analysis_signal[:samples_for_plot]

    with left:
        st.markdown(f'<div style="font-weight:800;margin-bottom:0.35rem;">{"Processed" if processed else "Generated"} Signal</div>', unsafe_allow_html=True)
        fig_time, ax_time = plt.subplots(figsize=(6.6, 4.45), facecolor="#111722")
        ax_time.set_facecolor("#111722")
        ax_time.plot(t_plot, x_plot, color="#8b7cff", linewidth=1.8)
        ax_time.set_xlabel("Time (s)", color="#c8d0dc")
        ax_time.set_ylabel("Amplitude", color="#c8d0dc")
        ax_time.tick_params(colors="#aeb8c7")
        ax_time.grid(True, alpha=0.16, color="#8f97a5")
        for spine in ax_time.spines.values():
            spine.set_color("#2d3849")
        fig_time.tight_layout()
        st.pyplot(fig_time, use_container_width=True)
        plt.close(fig_time)

    with right:
        st.markdown(f'<div style="font-weight:800;margin-bottom:0.35rem;">{"Processed" if processed else "Signal"} — FFT Spectrum</div>', unsafe_allow_html=True)
        freq, magnitude = calculate_fft(analysis_signal, Fs_value)
        max_display_frequency = min(7000.0, Fs_value / 2)
        fft_mask = freq <= max_display_frequency

        fig_fft, ax_fft = plt.subplots(figsize=(6.6, 4.45), facecolor="#111722")
        ax_fft.set_facecolor("#111722")
        ax_fft.plot(freq[fft_mask], magnitude[fft_mask], color="#34d6d0", linewidth=1.55)
        ax_fft.set_xlim(0, max_display_frequency)
        ax_fft.set_xlabel("Frequency (Hz)", color="#c8d0dc")
        ax_fft.set_ylabel("Magnitude", color="#c8d0dc")
        ax_fft.tick_params(colors="#aeb8c7")
        ax_fft.grid(True, alpha=0.16, color="#8f97a5")
        for spine in ax_fft.spines.values():
            spine.set_color("#2d3849")
        fig_fft.tight_layout()
        st.pyplot(fig_fft, use_container_width=True)
        plt.close(fig_fft)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Original vs Processed
# -------------------------
elif analysis_mode == "Original vs Processed":
    if not processed:
        st.info("Apply a filter from the left panel first to compare the original and processed signals.")
    else:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Original vs processed</div>', unsafe_allow_html=True)
        st.caption(f"Current conditioning: {st.session_state.filter_type}, order {st.session_state.order_value}.")

        left, right = st.columns(2, gap="large")
        samples_for_plot = max(2, min(samples_to_plot, len(x)))

        if waveform_value == "Sinc":
            center = len(x) // 2
            half = samples_for_plot // 2
            start = max(0, center - half)
            end = min(len(x), center + half)
        else:
            start, end = 0, samples_for_plot

        t_plot = t[start:end]
        original_plot = x[start:end]
        processed_plot = y[start:end]

        with left:
            st.markdown('<div style="font-weight:800;margin-bottom:0.35rem;">Time-domain comparison</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6.6, 4.45), facecolor="#111722")
            ax.set_facecolor("#111722")
            ax.plot(t_plot, original_plot, color="#667386", linewidth=1.25, label="Original")
            ax.plot(t_plot, processed_plot, color="#8b7cff", linewidth=1.75, label="Processed")
            ax.set_xlabel("Time (s)", color="#c8d0dc")
            ax.set_ylabel("Amplitude", color="#c8d0dc")
            ax.tick_params(colors="#aeb8c7")
            ax.grid(True, alpha=0.16)
            ax.legend(frameon=False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:
            st.markdown('<div style="font-weight:800;margin-bottom:0.35rem;">Frequency-domain comparison</div>', unsafe_allow_html=True)
            f0, m0 = calculate_fft(x, Fs_value)
            f1, m1 = calculate_fft(y, Fs_value)
            max_display_frequency = min(7000.0, Fs_value / 2)
            mask0 = f0 <= max_display_frequency
            mask1 = f1 <= max_display_frequency

            fig, ax = plt.subplots(figsize=(6.6, 4.45), facecolor="#111722")
            ax.set_facecolor("#111722")
            ax.plot(f0[mask0], m0[mask0], color="#667386", linewidth=1.15, label="Original")
            ax.plot(f1[mask1], m1[mask1], color="#34d6d0", linewidth=1.55, label="Processed")
            ax.set_xlim(0, max_display_frequency)
            ax.set_xlabel("Frequency (Hz)", color="#c8d0dc")
            ax.set_ylabel("Magnitude", color="#c8d0dc")
            ax.tick_params(colors="#aeb8c7")
            ax.grid(True, alpha=0.16)
            ax.legend(frameon=False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# STFT
# -------------------------
elif analysis_mode == "STFT Spectrogram":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Short-Time Fourier Transform</div>', unsafe_allow_html=True)
    st.caption(f"Window = {st.session_state.window_size_value} samples · Overlap = {st.session_state.overlap_value} samples")

    f_stft, t_stft, Zxx = calculate_stft(
        analysis_signal,
        Fs_value,
        st.session_state.window_size_value,
        st.session_state.overlap_value,
    )

    fig, ax = plt.subplots(figsize=(13.2, 5.25), facecolor="#111722")
    ax.set_facecolor("#111722")
    mesh = ax.pcolormesh(t_stft, f_stft, np.abs(Zxx), shading="gouraud", cmap="magma")
    ax.set_ylim(0, min(7000, Fs_value / 2))
    ax.set_xlabel("Time (s)", color="#c8d0dc")
    ax.set_ylabel("Frequency (Hz)", color="#c8d0dc")
    ax.tick_params(colors="#aeb8c7")
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label="Magnitude")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Playback + Export
# -------------------------
else:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Playback &amp; export</div>', unsafe_allow_html=True)
    st.caption("Preview or download the signal currently selected in the analysis workspace.")

    wav_bytes = signal_to_wav_bytes(analysis_signal, Fs_value)
    csv_bytes = signal_to_csv_bytes(t, analysis_signal)

    st.audio(wav_bytes, format="audio/wav")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.download_button(
            "↓ Download WAV",
            data=wav_bytes,
            file_name=f"{waveform_value.lower()}_signal.wav",
            mime="audio/wav",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "↓ Download CSV",
            data=csv_bytes,
            file_name=f"{waveform_value.lower()}_signal.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown(
        f'''<div class="small-panel" style="margin-top:0.9rem;">
        <div style="color:#8e9aac;font-size:0.69rem;text-transform:uppercase;letter-spacing:0.1em;font-weight:800;">Export summary</div>
        <div style="color:#dfe5ee;margin-top:0.28rem;">Fs = {Fs_value:,} Hz · Samples = {N_value:,} · Duration = {N_value / Fs_value:.3f} s</div>
        </div>''',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Signal Lab · Python · NumPy · SciPy · Matplotlib · Fixed sampling frequency = 44.1 kHz</div>', unsafe_allow_html=True)
