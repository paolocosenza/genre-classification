import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import plotly.express
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
        st.write('Downloading song, this may take some time...')
        with youtube_dl.YoutubeDL(options) as ydl:
            ydl.download([video_info['webpage_url']])
    except:
        st.write('youtube_dl is temporarily out of service. A new patch should be released in the next few days (end of February).') #Video unavailable. Please refresh the page and try again with a different URL.')
        return 1

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer():
    myargs = [
        "Made in ",
        image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',
              width=px(25), height=px(25)),
        " by ",
        "<a href=https://github.com/paolocosenza>Paolo Cosenza</a>"
    ]
    layout(*myargs)

if __name__=='__main__':
    st.title('Music genre classifier')
    link = '[How does it work?](https://github.com/paolocosenza/music-genre-classification#how-does-it-work)'
    st.markdown(link, unsafe_allow_html=True)
    footer()
    filename = "predict.mp3"
    
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
            spec=plt.figure(figsize=(4.32, 2.88))
            # Using librosa.display.specshow() to create our spectrogram
            librosa.display.specshow(log_spectro, sr=sr, hop_length=hop_length, cmap='magma')
            plt.savefig('spectrogram.png')


            learn_inf = load_learner('export.pkl')

            pred,pred_idx,probs = learn_inf.predict('spectrogram.png')

            st.write('Looks like you were listening to a ' + pred + ' track! I can assess that with ' + str(round(float(probs[pred_idx])*100)) + '% probability.')

            df = pd.DataFrame(dict(
                r=probs,
                theta=['blues','classical','country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']))
            fig = plotly.express.line_polar(df, r='r', theta='theta', line_close=True)
            st.plotly_chart(fig, use_container_width=True)

            st.write("Here's the spectrogram I used to classify the song:")
            img = Image.open("spectrogram.png")
            col1, col2, col3 = st.columns([0.25, 4, 0.25])
            col2.image(img, use_column_width=True)
                     
            os.remove('predict.mp3')
            os.remove('spectrogram.png')
