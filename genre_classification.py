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

from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

def run(video_url, filename):

    try:
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
    except:
        st.write('Video unavailable. Please try again with a different URL.')
        return 1

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by <a style='display: block; text-align: center;' href="https://github.com/paolocosenza" target="_blank">Paolo Cosenza</a></p>
</div>
"""

st.markdown(footer,unsafe_allow_html=True)

if __name__=='__main__':
    st.title('Music genre classifier')
    footer()
    filename = "predict.wav"
    
    video_url = st.text_input('Please enter a YouTube video URL: ')
    
    if len(video_url) != 0: 
        if run(video_url, filename) != 1:
            
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
            if probs[pred_idx] <= 0.5: 
                st.write("I'm not really sure about the genre of this track, but it may be " + pred + " (" + str(float(probs[pred_idx])) + " probability).")
            else:
                st.write('Looks like you were listening to a ' + pred + ' track! I can assess that with ' + str(round(float(probs[pred_idx])*100)) + '% probability.')
            os.remove('predict.wav')
            os.remove('predict.png')
