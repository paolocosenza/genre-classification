import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

from IPython.display import Image
from fastai.vision.all import *
from ipywidgets import widgets
import youtube_dl
import streamlit as st
import scipy.io as sio

def run(video_url, filename):


    video_info = youtube_dl.YoutubeDL().extract_info(
            url = video_url,download=False
            )

    options={
                'format':'bestaudio/best',
                'keepvideo':False,
                'outtmpl':filename,
                }
    st.write('Downloading song, please wait...')
    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([video_info['webpage_url']])

if __name__=='__main__':
    filename = "predict.wav"
    
    video_url = st.text_input('Please enter youtube video url: ')
    
    if len(folderPath) != 0: 
        run(video_url, filename)

        signal, sr = librosa.load(filename)

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
        st.write("Song correctly downloaded! Here's the spectrogram:")
        st.image(img)

        st.write('Looks like you were listening to a ' + pred + ' track! I can assess that with ' + str(round(float(probs[pred_idx])*100)) + '% probability')
        os.remove('predict.wav')
        os.remove('predict.png')
