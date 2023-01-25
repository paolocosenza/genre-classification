import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from pathlib import Path
from fastai.vision.all import *
from ipywidgets import widgets
import youtube_dl
import streamlit as st
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
filename = "predict.wav"
def extract_audio_from_yt_video(url):
    
    filename = "yt_download_" + url[-11:] + ".mp3"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': filename,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
    }
    with st.spinner("We are extracting the audio from the video"):
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    # Handle DownloadError: ERROR: unable to download video data: HTTP Error 403: Forbidden / happens sometimes
    return filename

url = st.text_input("Enter the YouTube video URL then press Enter to confirm!")
    
# If link seems correct, we try to transcribe
if "youtu" in url:
    filename = extract_audio_from_yt_video(url)
    if filename is not None:
        transcription(stt_tokenizer, stt_model, filename)
    else:
        st.error("We were unable to extract the audio. Please verify your link, retry or choose another video")

signal, _ = librosa.load(filename, sr=16000)

# this is the number of samples in a window per fft
n_fft = 2048
# The amount of samples we are shifting after each fft
hop_length = 512
# Short-time Fourier Transformation on our audio data
audio_stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
# gathering the absolute values for all values in our audio_stft
spectrogram = np.abs(audio_stft)
# Converting the amplitude to decibels
log_spectro = librosa.amplitude_to_db(spectrogram)
# Plotting the short-time Fourier Transformation
plt.figure(figsize=(4.32, 2.88))
# Using librosa.display.specshow() to create our spectrogram
librosa.display.specshow(log_spectro, sr=sr, hop_length=hop_length, cmap='magma')
plt.savefig('predict.png')
learn_inf = load_learner('export.pkl')
pred,pred_idx,probs = learn_inf.predict('predict.png')
img = Image.open("predict.png")
st.image(img)
st.write('Looks like you were listening to a ' + pred + ' track! I can assess that with ' + str(round(float(probs[pred_idx])*100)) + '% probability')
