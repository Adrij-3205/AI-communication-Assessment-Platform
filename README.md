# AI Communication Assessment Platform

This project is a web-based AI tool to assess communication skills using video and audio analysis. It records the user from their browser, processes the data, and provides scores and feedback on grammar, pronunciation, speaking rate, filler words, and body language.

## Folder Structure

Place the files in the following structure:

- app.py  (Flask backend)
- process.py  (AI/ML processing logic)
- requirements.txt  (Python dependencies)
- /templates
    - index.html  (Main HTML page)
- /static
    - style.css  (CSS styling)
    - script.js  (Frontend JavaScript)
- /uploads  (Will be created automatically to store temporary recordings)

## Setup Instructions

1. Open a terminal in the project folder.

2. Create a virtual environment:
    python -m venv venv

3. Activate the virtual environment:
- Windows:
  ```
  venv\Scripts\activate
  ```


4. Install dependencies:
    pip install -r requirements.txt


5. Make sure FFmpeg is installed and added to your system PATH.

6. Run the application:
  python app.py

7. Open a web browser and go to:
  http://127.0.0.1:5000
