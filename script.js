let webcamStream;
let mediaRecorder;
let recordedChunks = [];
let resultsData = {};

function startAssessment() {
    document.getElementById("results").style.display = "none";
    recordedChunks = [];

    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
    .then(stream => {
        webcamStream = stream;
        document.getElementById("webcam").srcObject = stream;

        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = function(e) {
            if (e.data.size > 0) recordedChunks.push(e.data);
        };
        mediaRecorder.start();

        let timeLeft = 30;
        document.getElementById("countdown").innerText = `Time left: ${timeLeft}s`;
        let timer = setInterval(() => {
            timeLeft--;
            document.getElementById("countdown").innerText = `Time left: ${timeLeft}s`;
            if (timeLeft <= 0) {
                clearInterval(timer);
                stopAssessment();
            }
        }, 1000);
    })
    .catch(err => console.error("Error accessing webcam:", err));
}

function stopAssessment() {
    mediaRecorder.stop();
    webcamStream.getTracks().forEach(track => track.stop());

    mediaRecorder.onstop = function() {
        let blob = new Blob(recordedChunks, { type: 'video/webm' });
        let formData = new FormData();
        formData.append("video", blob, "assessment.webm");

        fetch("/start_assessment", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => alert(data.status));
    };
}

function viewResults() {
    fetch("/get_results")
        .then(res => res.json())
        .then(data => {
            if (data.status) {
                alert(data.status);
                return;
            }
            resultsData = data;
            let html = "<h2>Results</h2>";
            for (let key in data) {
                let value = data[key];
                let cls = "";
                if (typeof value === "number") {
                    if (value >= 80) cls = "good";
                    else if (value >= 50) cls = "average";
                    else cls = "poor";
                }
                html += `<div class="card"><strong>${key}:</strong> <span class="score ${cls}">${value}</span></div>`;
            }
            document.getElementById("results").innerHTML = html;
            document.getElementById("results").style.display = "block";
        });
}

function downloadPDF() {
    if (!resultsData || Object.keys(resultsData).length === 0) {
        alert("Please view results first.");
        return;
    }
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    doc.setFontSize(18);
    doc.text("Communication Assessment Results", 20, 20);
    doc.setFontSize(12);
    let y = 30;
    for (let key in resultsData) {
        doc.text(`${key}: ${resultsData[key]}`, 20, y);
        y += 8;
    }
    doc.save("assessment_results.pdf");
}
