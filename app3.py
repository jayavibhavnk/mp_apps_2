import streamlit as st
import time
from youtube_transcript_api import YouTubeTranscriptApi
from streamlit_player import st_player

# Prompt the user for a YouTube link
video_url = st.text_input("Enter a YouTube video URL:")

if video_url:
    # Extract the video ID from the URL
    video_id = video_url.split("v=")[-1]

    try:
        # Get the transcript of the YouTube video
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        script1 = transcript_list.find_transcript(['en'])
        script2 = script1.translate('de')
        script1 = script1.fetch()
        script2 = script2.fetch()

        # Display the video
        st_player(video_url, controls=True)

        # Display subtitles dynamically
        subtitle_placeholder = st.empty()
        sub2 = st.empty()
        if st.button("Show Transcripts"):
            for i in range(len(script1)):
                current_subtitle = script1[i]['text']
                cur1 = script2[i]['text']
                subtitle_placeholder.markdown(f"{current_subtitle}")
                sub2.markdown(f"{cur1}")
                time.sleep(script1[i]["duration"])
                subtitle_placeholder.empty()
                sub2.empty()

    except Exception as e:
        st.error(f"An error occurred: {e}")
