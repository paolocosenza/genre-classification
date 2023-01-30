import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Image
from pathlib import Path
from fastai.vision.all import *
from ipywidgets import widgets
import youtube_dl
import streamlit as st
import scipy.io as sio

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

check = 0

while check == 0:
    video_url = st.text_input('Please enter youtube video url: ')
    if len(video_url) == "https://www.youtube.com/watch?v=DfAt73ru258": check = 1
        
video_info = youtube_dl.YoutubeDL().extract_info(
            url = video_url,download=False
            )
filename = "predict.wav"
options={
                'format':'bestaudio/best',
                'keepvideo':False,
                'outtmpl':filename,
                }

with youtube_dl.YoutubeDL(options) as ydl:
    ydl.download([video_info['webpage_url']])

signal, sr = sio.wavfile.read(filename)

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
