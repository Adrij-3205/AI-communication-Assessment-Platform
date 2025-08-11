from flask import Flask, render_template, request, jsonify
import os
from process import run_assessment

app = Flask(__name__)
last_results = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_assessment", methods=["POST"])
def start_assessment():
    global last_results

    if "video" not in request.files:
        return jsonify({"status": "No video uploaded"})

    video_file = request.files["video"]
    save_path = os.path.join("uploads", "uploaded.webm")
    os.makedirs("uploads", exist_ok=True)
    video_file.save(save_path)

    # Convert WebM to AVI/WAV for analysis
    os.system(f'ffmpeg -i "{save_path}" -ar 16000 -ac 1 "output.wav" -y')
    os.system(f'ffmpeg -i "{save_path}" "output.avi" -y')

    last_results = run_assessment()
    return jsonify({"status": "Assessment completed"})

@app.route("/get_results")
def get_results():
    if not last_results:
        return jsonify({"status": "No results available yet"})
    return jsonify(last_results)

if __name__ == "__main__":
    app.run(debug=True)
