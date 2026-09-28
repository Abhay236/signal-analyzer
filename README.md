# Signal Generator & Analyzer

A Python-based Digital Signal Processing (DSP) web application built with
Streamlit for generating, processing, analyzing, and exporting audio-range signals.

## Features

### Signal Generation
The application can generate:

- Sine wave
- Square wave
- Triangle wave
- Chirp signal
- Sinc signal

Users can control parameters such as amplitude, frequency, duration, and
duty cycle where applicable.

### Digital Filtering

Supports:

- Low-Pass Butterworth filter
- High-Pass Butterworth filter
- Band-Pass Butterworth filter

The filter order and cutoff frequencies can be configured by the user.

### Signal Analysis

The generated/processed signal can be analyzed in:

- **Time Domain** — waveform amplitude versus time
- **Frequency Domain** — FFT spectrum
- **Time-Frequency Domain** — STFT spectrogram

### Playback & Export

The application allows users to:

- Play the generated/processed signal
- Download the signal as a WAV file
- Download time-amplitude data as a CSV file

## DSP Concepts

This project demonstrates:

- Sampling and Nyquist theorem
- Discrete-time signal generation
- Periodic waveforms
- Duty cycle
- Fourier Transform
- Fast Fourier Transform (FFT)
- Digital filtering
- Butterworth filters
- Filter order
- Zero-phase filtering
- Short-Time Fourier Transform (STFT)
- Time-frequency analysis

## Technical Details

- **Sampling Frequency:** 44.1 kHz
- **Nyquist Frequency:** 22.05 kHz
- **Programming Language:** Python

## Technologies Used

- Python
- Streamlit
- NumPy
- SciPy
- Matplotlib

## Run Locally

Clone the repository:

```
git clone https://github.com/Abhay236/signal-analyzer.git
cd signal-analyzer
```

Install the required dependencies:
```
pip install -r requirements.txt
```

Run the application:
```
streamlit run app.py
```

## Live Demo

[Signal Generator & Analyzer](https://signal-analyzer-4s4sfbviqznc6vrckaiegc.streamlit.app/)
