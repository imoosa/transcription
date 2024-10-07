from flask import Flask, render_template, request, redirect, url_for
import subprocess
import os
import pandas as pd
from pydub import AudioSegment
import whisper
from deep_translator import GoogleTranslator
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False")

app = Flask(__name__)

def extract_audio_from_youtube(url, audio_path='audio.wav'):
    """Extract audio directly from a YouTube video and save as WAV."""
    try:
        result = subprocess.run(
            ['yt-dlp', '--extract-audio', '--audio-format', 'wav', '-o', audio_path, url],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Audio extracted and saved to: {audio_path}")
            return audio_path  # Return the path to the saved audio file
        else:
            print(f"Error extracting audio: {result.stderr}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def extract_text_from_audio_whisper(audio_file_path, start_time, duration):
    """Extract text from an audio segment using Whisper."""
    model = whisper.load_model("medium")  # Whisper model loading
    audio = AudioSegment.from_wav(audio_file_path)
    start_ms = start_time * 1000
    end_ms = (start_time + duration) * 1000
    audio_segment = audio[start_ms:end_ms]
    audio_segment.export("/tmp/temp_segment.wav", format="wav")  # Export temp segment
    result = model.transcribe("/tmp/temp_segment.wav")  # Transcribe with Whisper
    text = result['text']
    return text

def split_text_into_chunks(text, max_length=200):
    """Split text into chunks of a specified maximum length."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if sum(len(w) for w in current_chunk) + len(word) + len(current_chunk) - 1 < max_length:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_audio_and_transcribe(audio_file_path, translate, target_language=None):
    """Process audio and transcribe it into text chunks with optional translation."""
    audio = AudioSegment.from_wav(audio_file_path)
    audio_duration = audio.duration_seconds

    # Check if the video/audio is longer than 5 minutes
    if audio_duration > 300:
        return None, "This video is longer than 5 minutes. Please subscribe!"

    # Dummy speaker diarization step or skip for now
    df = pd.DataFrame({"Start time": [0], "Duration": [audio_duration], "Speaker": [1]})
    df = df.astype({"Start time": "float", "Duration": "float"})
    df["Utterance"] = None
    df["End time"] = df["Start time"] + df["Duration"]
    new_entries = []

    for ind in df.index:
        start_time = df["Start time"][ind]
        duration = df["Duration"][ind]
        try:
            transcription = extract_text_from_audio_whisper(audio_file_path, start_time, duration)
            if len(transcription) > 200:
                chunks = split_text_into_chunks(transcription, 200)
                for chunk in chunks:
                    new_entry = df.iloc[ind].copy()
                    new_entry["Utterance"] = chunk
                    new_entries.append(new_entry)
            else:
                df.at[ind, "Utterance"] = transcription
        except Exception as e:
            df.at[ind, "Utterance"] = f"Error: {e}"

    if new_entries:
        df = pd.concat([df, pd.DataFrame(new_entries)], ignore_index=True)

    df = df.sort_values(by="Start time").reset_index(drop=True)

    # Perform translation if required
    if translate == 'yes' and target_language:
        translator = GoogleTranslator(source='auto', target=target_language)
        for ind in df.index:
            utterance = df["Utterance"][ind]
            if utterance is not None and not utterance.startswith("Error:"):
                try:
                    translated_text = translator.translate(utterance)
                    df.at[ind, "Translated Utterance"] = translated_text
                except Exception as e:
                    df.at[ind, "Translated Utterance"] = f"Error: {e}"

    return df, None

@app.route('/', methods=['GET', 'POST'])
def index():
    transcript = ""
    error = ""

    if request.method == 'POST':
        video_url = request.form.get('youtube_url')
        translate = request.form.get('translate')
        target_language = request.form.get('target_language')

        if not video_url:
            error = "YouTube URL is required."
        else:
            # Extract audio from YouTube
            audio_file = extract_audio_from_youtube(video_url)
            if audio_file:
                # Transcribe the extracted audio
                transcript_df, transcribe_error = process_audio_and_transcribe(audio_file, translate, target_language)
                if transcribe_error:
                    error = transcribe_error
                else:
                    # Join transcript parts into a string
                    transcript = "\n".join(transcript_df["Utterance"].dropna())
                # Cleanup: Delete the audio file after processing
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            else:
                error = "Failed to extract audio from YouTube."

    return render_template('index.html', transcript=transcript, error=error)

if __name__ == '__main__':
    app.run(debug=True, port=7988)
