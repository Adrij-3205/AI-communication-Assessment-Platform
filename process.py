import cv2
import speech_recognition as sr
import language_tool_python
import nltk
import time
import requests
import numpy as np
import os
import pyaudio
import wave
import threading
import subprocess
import azure.cognitiveservices.speech as speechsdk
import json
from pydub import AudioSegment, silence
from nltk.tokenize import word_tokenize
import pronouncing
import mediapipe as mp

# Download necessary NLTK data
nltk.download('punkt')

# Set the NLTK data path explicitly
nltk.data.path.append('C:\\Users\\adrij\\AppData\\Roaming\\nltk_data')

# Now you can proceed with other imports and function calls
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt_tab')


# Function to extract audio from video and perform speech-to-text
def video_to_text(video_file, language='en-US'):
    audio_file = "output.wav"

    # Use absolute paths
    video_file = os.path.abspath(video_file)
    audio_file = os.path.abspath(audio_file)

    # Debug: Print paths
    print(f"Video file: {video_file}")
    print(f"Audio file: {audio_file}")

    # FFmpeg command to extract audio
    command = f"ffmpeg -i \"{video_file}\" -ar 16000 -ac 1 -y \"{audio_file}\""
    os.system(command)

    # Verify if the audio file was created
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file {audio_file} was not created. Check the FFmpeg command.")

    # Initialize recognizer and process audio
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)

        # Convert audio to text
        text = recognizer.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand the audio."
    except sr.RequestError:
        return "Speech recognition service is unavailable."
    
def analyze_grammar(text):
    """
    Uses LanguageTool API to check for grammatical issues in the text.
    Filters out irrelevant issues like capitalization errors.
    """
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    
    # Define rules to ignore (e.g., capitalization or punctuation errors)
    ignored_rules = {
        "UPPERCASE_SENTENCE_START",  # Ignore sentences not starting with a capital letter
        "PUNCTUATION_PARAGRAPH_END",  # Ignore missing punctuation at paragraph end
    }

    # Filter out matches with ignored rules
    relevant_matches = [match for match in matches if match.ruleId not in ignored_rules]
    
    # Simplistic scoring: reduce score by 10 points for each relevant issue
    grammar_score = max(0, 100 - len(relevant_matches) * 10)
    
    flagged_issues = [(match.ruleId, match.message, match.context) for match in relevant_matches]
    
    print(f"Grammar issues found: {len(relevant_matches)}")
    print("Flagged Issues:")
    for issue in flagged_issues:
        print(f" - {issue[0]}: {issue[1]} (Context: {issue[2]})")
    
    return grammar_score, flagged_issues

# Function for speaking rate analysis
def speaking_rate(text, duration):
    """
    Calculates speaking rate (words per minute).
    """
    word_count = len(word_tokenize(text))
    speaking_rate = word_count / (duration / 60)  # words per minute
    return speaking_rate

from pydub import AudioSegment, silence
from nltk.tokenize import word_tokenize

def analyze_pause_filler(audio_file, transcript):
    """
    Counts filler words from transcript and calculates pause time (>2 sec) from audio.
    Handles multi-word fillers correctly.
    """
    # Single word fillers
    single_fillers = {"uh", "um", "umm", "like", "well", "ah"}
    # Multi-word fillers
    multi_fillers = {"you know", "or something"}

    # ---- Count filler words ----
    words = word_tokenize(transcript.lower())
    filler_count = sum(1 for word in words if word in single_fillers)

    # Count multi-word fillers from the transcript text
    transcript_lower = transcript.lower()
    for phrase in multi_fillers:
        filler_count += transcript_lower.count(phrase)

    # ---- Detect pauses ----
    audio = AudioSegment.from_file(audio_file, format="wav")
    silent_chunks = silence.detect_silence(
        audio,
        min_silence_len=2000,  # 2 seconds
        silence_thresh=audio.dBFS - 16
    )
    total_pause_time = sum((end - start) / 1000.0 for start, end in silent_chunks)

    return filler_count, total_pause_time

def calculate_pause_score(total_pause_time):
    """
    Calculates a penalty score for total pause time based on durations.
    """
    max_score = 100
    penalty_per_second = 15
    pause_penalty = min(max_score, total_pause_time * penalty_per_second)
    return max(0, max_score - pause_penalty)

def measure_fluency(audio_file, transcript, wpm_target=140):
    # Get pauses from analyze_pause_filler
    _, total_pause_time = analyze_pause_filler(audio_file, transcript)

    # Speaking rate
    words = word_tokenize(transcript)
    duration_sec = AudioSegment.from_file(audio_file).duration_seconds
    wpm = (len(words) / duration_sec) * 60

    # Pause penalty
    pause_penalty = total_pause_time * 8  # 15 points per sec pause
    # WPM penalty
    wpm_penalty = abs(wpm_target - wpm) * 0.5  # penalize deviation

    fluency_score = max(0, 100 - pause_penalty - wpm_penalty)
    return round(fluency_score, 2)

def measure_completeness(transcript):
    words = [w.lower() for w in word_tokenize(transcript) if w.isalpha()]
    total_words = len(words)
    unique_words = len(set(words))

    if total_words == 0:
        return 0

    diversity_ratio = unique_words / total_words  # 0 to 1
    repetition_penalty = (1 - diversity_ratio) * 50  # up to -50 points

    completeness_score = max(0, 100 - repetition_penalty)
    return round(completeness_score, 2)

def analyze_pronunciation(audio_file, transcript):
    """
    Improved pronunciation analysis without Azure.
    """
    # Tokenize & filter
    words = [w.lower() for w in word_tokenize(transcript) if w.isalpha()]
    total_words = len(words)
    if total_words == 0:
        return 0, 0, 0, 0

    # Accuracy: how many words have dictionary pronunciations
    words_with_pron = [w for w in words if pronouncing.phones_for_word(w)]
    accuracy_score = (len(words_with_pron) / total_words) * 100

    # Fluency: pauses + speaking rate stability
    fluency_score = measure_fluency(audio_file, transcript)

    # Completeness: lexical diversity & repetition penalty
    completeness_score = measure_completeness(transcript)

    # Overall: weighted average
    overall_score = round(
        (accuracy_score * 0.4 + fluency_score * 0.4 + completeness_score * 0.2), 2
    )

    return overall_score, accuracy_score, fluency_score, completeness_score


mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

def analyze_body_language(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return 0, 0, 0

    posture_score = 0
    gesture_score = 0
    eye_contact_score = 0
    total_frames = 0

    prev_hand_positions = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Pose detection
            pose_results = pose.process(rgb)
            face_results = face_mesh.process(rgb)
            hands_results = hands.process(rgb)

            # Posture: Check if head is upright
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
                
                # Upright check: shoulders should be roughly horizontal
                shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
                if shoulder_diff < 0.05:  # Adjust threshold
                    posture_score += 1

            # Eye Contact: Check iris position
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # Iris landmark indices for MediaPipe Face Mesh
                    LEFT_IRIS = [474, 475, 476, 477]
                    right_iris_x = face_landmarks.landmark[LEFT_IRIS[0]].x
                    if 0.4 < right_iris_x < 0.6:  # Eye is looking forward
                        eye_contact_score += 1

            # Gesture: Track hand movement
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    cx = hand_landmarks.landmark[0].x
                    cy = hand_landmarks.landmark[0].y
                    prev_hand_positions.append((cx, cy))
                    if len(prev_hand_positions) > 2:
                        dist = ((cx - prev_hand_positions[-2][0])**2 + (cy - prev_hand_positions[-2][1])**2)**0.5
                        if dist > 0.02:  # Hand moved significantly
                            gesture_score += 1

    cap.release()

    # Normalize scores
    posture_score = min((posture_score / total_frames) * 100, 100)
    #gesture_score = min((gesture_score / total_frames) * 100, 100)
    eye_contact_score = min((eye_contact_score / total_frames) * 100, 100)

    return posture_score, gesture_score, eye_contact_score

def calculate_overall_score(grammar_score, speaking_rate, filler_count, total_pause_time, 
                            pronunciation_score, accuracy_score, fluency_score):
    """
    Combines all metrics to calculate a final score.
    """
    # Weighting of metrics
    weights = {
        "grammar": 0.3,
        "speaking_rate": 0.1,
        "filler_words": 0.1,
        "pause_patterns": 0.1,
        "pronunciation": 0.2,
        "accuracy": 0.1,
        "fluency": 0.1
    }

    # Normalize scores and calculate weighted sum
    speaking_rate_score = min(100, max(0, 100 - abs(150 - speaking_rate)))  # Target: 150 WPM
    filler_word_penalty = max(0, 100 - filler_count * 10)
    pause_score = calculate_pause_score(total_pause_time)

    overall_score = (
        weights["grammar"] * grammar_score +
        weights["speaking_rate"] * speaking_rate_score +
        weights["filler_words"] * filler_word_penalty +
        weights["pause_patterns"] * pause_score +
        weights["pronunciation"] * pronunciation_score +
        weights["accuracy"] * accuracy_score +
        weights["fluency"] * fluency_score
    )
    return overall_score

def provide_feedback(speaking_rate, fluency_score, pronunciation_score):
    """
    Provides feedback based on speaking rate, fluency, and pronunciation.
    """
    # Speaking Pace Feedback
    if speaking_rate < 80:
        pace_feedback = "You’re speaking too slowly. Consider speeding up to maintain engagement."
    elif 80 <= speaking_rate < 130:
        pace_feedback = "A moderate pace, but slightly faster speech may improve energy."
    elif 130 <= speaking_rate < 170:
        pace_feedback = "Perfect! Your speaking pace is in the ideal range."
    elif 170 <= speaking_rate < 200:
        pace_feedback = "A bit fast. Consider slowing down slightly to enhance clarity."
    else:
        pace_feedback = "Too fast. Try slowing down for better audience comprehension."

    # Voice Clarity Feedback
    if fluency_score < 70:
        fluency_feedback = "Your speech is not fluent. Focus on reducing hesitations and improving rhythm."
    elif 70 <= fluency_score < 90:
        fluency_feedback = "Good fluency, but strive for smoother transitions between words."
    else:
        fluency_feedback = "Excellent fluency! Your speech flows naturally."

    # Pronunciation Feedback
    if pronunciation_score < 70:
        pronunciation_feedback = "Your pronunciation needs improvement. Try practicing individual sounds and word stresses."
    elif 70 <= pronunciation_score < 90:
        pronunciation_feedback = "Your pronunciation is quite good, but there’s room for slight improvement."
    else:
        pronunciation_feedback = "Excellent pronunciation! Keep it up."

    return pace_feedback, fluency_feedback, pronunciation_feedback

def run_assessment():
    video_file = "output.avi"
    audio_file = "output.wav"
    duration = 30  # seconds

    # No recording here — just process the saved files
    transcript = video_to_text(video_file)
    grammar_score, grammar_issues = analyze_grammar(transcript)
    speaking_rate_value = speaking_rate(transcript, duration)
    filler_word_count, total_pause_time = analyze_pause_filler(audio_file, transcript)
    pronunciation_score, accuracy_score, fluency_score, completeness_score = analyze_pronunciation(audio_file, transcript)
    overall_score = calculate_overall_score(grammar_score, speaking_rate_value, filler_word_count, total_pause_time,
                                            pronunciation_score, accuracy_score, fluency_score)
    pace_feedback, fluency_feedback, pronunciation_feedback = provide_feedback(speaking_rate_value, fluency_score, pronunciation_score)
    posture_score, gesture_score, eye_contact_score = analyze_body_language(video_file)

    return {
        "Transcript": transcript,
        "Grammar Score": grammar_score,
        "Grammar Issues": grammar_issues,
        "Speaking Rate (WPM)": speaking_rate_value,
        "Filler Word Count": filler_word_count,
        "Total Pause Time": total_pause_time,
        "Pronunciation Score": pronunciation_score,
        "Accuracy": accuracy_score,
        "Fluency": fluency_score,
        "Completeness": completeness_score,
        "Overall Score": overall_score,
        "Pace Feedback": pace_feedback,
        "Fluency Feedback": fluency_feedback,
        "Pronunciation Feedback": pronunciation_feedback,
        "Posture Score": posture_score,
        "Eye Contact Score": eye_contact_score
    }