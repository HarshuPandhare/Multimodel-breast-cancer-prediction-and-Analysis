import os
import re
import time
import json
import uuid
import joblib
import pandas as pd
import torch
import torch.nn as nn
from functools import wraps
from contextlib import contextmanager
from flask import Flask, render_template, request, send_from_directory, jsonify, send_file, redirect, url_for, session, flash
from torchvision import models, transforms
from PIL import Image
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from google import genai
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import hmac
import threading

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true",
)

database_initialized = False
database_initialization_lock = threading.Lock()

CLINICAL_MODEL_PATH = "models/breast_cancer_model.pkl"
IMAGE_MODEL_PATH = "models/breast_cancer_efficientnet.pth"
UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

if not os.path.exists(CLINICAL_MODEL_PATH):
    raise FileNotFoundError("Clinical model not found: " + CLINICAL_MODEL_PATH)

clinical_package = joblib.load(CLINICAL_MODEL_PATH)
clinical_model = clinical_package["model"]
target_encoder = clinical_package["target_encoder"]
clinical_accuracy = clinical_package["accuracy"]

if not os.path.exists(IMAGE_MODEL_PATH):
    raise FileNotFoundError("Image model not found: " + IMAGE_MODEL_PATH)

image_model = models.efficientnet_b0(weights=None)
number_of_features = image_model.classifier[1].in_features
image_model.classifier[1] = nn.Linear(number_of_features, 2)

checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=device)
image_model.load_state_dict(checkpoint["model_state_dict"])
image_model = image_model.to(device)
image_model.eval()

image_classes = checkpoint["classes"] if "classes" in checkpoint else ["benign", "malignant"]
print("Image classes:", image_classes)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Gemini AI Agent Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")

gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=genai.types.HttpOptions(
                timeout=60_000,
                retry_options=genai.types.HttpRetryOptions(attempts=1),
            ),
        )
        print(f"Gemini AI agent initialized with model: {GEMINI_MODEL}")
    except Exception as e:
        print(f"Warning: Failed to initialize Gemini client: {e}")


def encode_marker(value):
    value = str(value).strip().lower()
    if value in ["positive", "pos", "1"]:
        return 1
    if value in ["negative", "neg", "0"]:
        return 0
    raise ValueError("Invalid biomarker value.")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def connect_database():
    database_url = os.getenv("DATABASE_URL")
    import psycopg
    if database_url:
        return psycopg.connect(database_url)

    connection_options = {
        "host": os.getenv("PGHOST", "localhost"),
        "port": os.getenv("PGPORT", "5432"),
        "dbname": os.getenv("PGDATABASE"),
        "user": os.getenv("PGUSER"),
        "password": os.getenv("PGPASSWORD"),
    }
    missing_options = [key for key, value in connection_options.items() if not value]
    if missing_options:
        raise RuntimeError(
            "Set DATABASE_URL or the PostgreSQL settings: "
            + ", ".join("PG" + key.upper() for key in missing_options)
            + "."
        )

    return psycopg.connect(**connection_options)


@contextmanager
def database_connection():
    connection = connect_database()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def ensure_database():
    global database_initialized
    if database_initialized:
        return

    with database_initialization_lock:
        if database_initialized:
            return

        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS app_users (
                        id BIGSERIAL PRIMARY KEY,
                        username TEXT NOT NULL UNIQUE,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('admin', 'patient')),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_reports (
                        id TEXT PRIMARY KEY,
                        patient_id BIGINT NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
                        payload JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS medical_requests (
                        id BIGSERIAL PRIMARY KEY,
                        patient_id BIGINT NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
                        message TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'Open',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS analysis_reports_patient_created_idx
                    ON analysis_reports (patient_id, created_at DESC)
                """)

                admin_username = os.getenv("ADMIN_USERNAME")
                admin_password = os.getenv("ADMIN_PASSWORD")
                if not admin_username or not admin_password:
                    raise RuntimeError("Set ADMIN_USERNAME and ADMIN_PASSWORD before starting the app.")
                admin_username = admin_username.strip()
                if not admin_username:
                    raise RuntimeError("ADMIN_USERNAME cannot be empty.")
                if len(admin_password) < 12:
                    raise RuntimeError("ADMIN_PASSWORD must be at least 12 characters.")
                cursor.execute(
                    """INSERT INTO app_users (username, password_hash, role)
                       VALUES (%s, %s, 'admin') ON CONFLICT (username) DO NOTHING""",
                    (admin_username, generate_password_hash(admin_password))
                )
                cursor.execute("SELECT role FROM app_users WHERE username = %s", (admin_username,))
                if cursor.fetchone()[0] != "admin":
                    raise RuntimeError("ADMIN_USERNAME is already assigned to a non-admin account.")

        database_initialized = True


def get_csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = uuid.uuid4().hex
    return session["csrf_token"]


@app.context_processor
def inject_template_helpers():
    return {"csrf_token": get_csrf_token}


@app.before_request
def prepare_request():
    ensure_database()
    if request.method == "POST":
        expected_token = session.get("csrf_token", "")
        supplied_token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
        if (not expected_token or not supplied_token or
                not hmac.compare_digest(str(expected_token), str(supplied_token))):
            return "Invalid or missing CSRF token.", 400


def require_role(role):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if session.get("role") != role:
                if not session.get("user_id"):
                    return redirect(url_for("login"))
                return redirect(url_for("patient_portal" if session.get("role") == "patient" else "home"))
            return view(*args, **kwargs)
        return wrapped
    return decorator


def get_json_payload(value):
    return json.loads(value) if isinstance(value, str) else value


def load_patient_analysis_history(patient_id):
    with database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT id, payload, created_at FROM analysis_reports
                   WHERE patient_id = %s ORDER BY created_at DESC LIMIT 50""",
                (patient_id,)
            )
            return [
                {"id": row[0], "payload": get_json_payload(row[1]), "created_at": row[2]}
                for row in cursor.fetchall()
            ]


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = image_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = image_classes[predicted.item()]
    confidence_value = confidence.item() * 100

    class_probabilities = []
    for i, class_name in enumerate(image_classes):
        class_probabilities.append({
            "class": class_name,
            "probability": round(probabilities[0][i].item() * 100, 2)
        })

    return predicted_class, round(confidence_value, 2), class_probabilities


def build_gemini_prompt(patient_data, clinical_result, image_result):
    """Build a structured prompt for the Gemini AI agent."""
    prompt = """You are a medical AI documentation assistant specializing in breast cancer analysis.
Based on the following patient data and ML model predictions, generate a comprehensive clinical documentation report.

IMPORTANT: You must respond with ONLY valid JSON, no markdown formatting, no code blocks, no extra text.
Use the exact JSON structure shown below.

PATIENT DATA:
- Age at Diagnosis: {age}
- ER Status: {er}
- PR Status: {pr}
- HER2 Status: {her2}
- Tumor Size: {tumor_size} mm
- Positive Lymph Nodes: {lymph_nodes}

ML MODEL PREDICTIONS:
- Clinical/Biomarker Model Prediction (PAM50 Subtype): {clinical_prediction}
- Clinical Model Confidence: {clinical_confidence}%
""".format(
        age=patient_data.get("age", "N/A"),
        er=patient_data.get("er_status", "N/A"),
        pr=patient_data.get("pr_status", "N/A"),
        her2=patient_data.get("her2_status", "N/A"),
        tumor_size=patient_data.get("tumor_size", "N/A"),
        lymph_nodes=patient_data.get("lymph_nodes", "N/A"),
        clinical_prediction=clinical_result.get("prediction", "N/A"),
        clinical_confidence=clinical_result.get("confidence", "N/A")
    )

    if image_result:
        prompt += """- Ultrasound Image Model Prediction: {image_prediction}
- Image Model Confidence: {image_confidence}%
""".format(
            image_prediction=image_result.get("prediction", "N/A"),
            image_confidence=image_result.get("confidence", "N/A")
        )

    prompt += """
Respond ONLY with this JSON structure (no markdown, no code fences):
{
    "summary": "A 2-3 sentence executive summary of the patient case and key findings.",
    "risk_assessment": "Detailed risk assessment based on the biomarkers and model predictions. Include risk level (Low/Moderate/High/Very High) and reasoning.",
    "subtype_analysis": "Explanation of the predicted PAM50 subtype, its characteristics, typical prognosis, and treatment response patterns.",
    "biomarker_interpretation": "Interpretation of ER, PR, and HER2 status combination and what it means for treatment selection.",
    "recommended_actions": [
        "First recommended action with explanation",
        "Second recommended action with explanation",
        "Third recommended action with explanation",
        "Fourth recommended action with explanation",
        "Fifth recommended action with explanation"
    ],
    "additional_tests": [
        "First additional test recommendation",
        "Second additional test recommendation",
        "Third additional test recommendation"
    ],
    "treatment_considerations": "Overview of potential treatment approaches based on the molecular subtype and biomarker profile.",
    "follow_up_plan": "Suggested follow-up schedule and monitoring plan.",
    "confidence_notes": "Notes about the AI model confidence levels and any caveats about the predictions."
}

IMPORTANT GUIDELINES:
- This is for research/educational purposes only - always include appropriate disclaimers
- Be thorough and evidence-based in your analysis
- Consider the interplay between all biomarkers and clinical features
- Provide actionable, specific recommendations
- Do NOT wrap the response in markdown code blocks or any other formatting
"""
    return prompt


def call_gemini_agent(prompt):
    """Call the Gemini AI agent with fallback model support."""
    if not gemini_client:
        raise ValueError("Gemini AI agent not configured. Check your GEMINI_API_KEY in .env")

    models_to_try = [GEMINI_MODEL, GEMINI_FALLBACK_MODEL]
    last_error = None

    for model_name in models_to_try:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            response_text = response.text
            if not response_text or not response_text.strip():
                raise RuntimeError(f"Gemini model {model_name} returned an empty response.")
            return response_text
        except Exception as error:
            last_error = error
            print(f"Gemini model {model_name} failed: {error}")

    raise RuntimeError(
        f"Gemini report generation failed for both configured models. Last error: {last_error}"
    ) from last_error


def parse_gemini_response(response_text):
    """Parse the JSON response from Gemini, handling markdown code fences."""
    cleaned = response_text.strip()

    # Remove markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {
            "summary": response_text[:500],
            "risk_assessment": "Could not parse structured response from AI agent.",
            "subtype_analysis": "",
            "biomarker_interpretation": "",
            "recommended_actions": ["Please regenerate the report for structured recommendations."],
            "additional_tests": [],
            "treatment_considerations": "",
            "follow_up_plan": "",
            "confidence_notes": "Response parsing failed - raw AI response was preserved in summary."
        }


def generate_docx_report(patient_data, clinical_result, image_result, ai_analysis, report_id):
    """Generate a professional .docx report with all analysis data."""
    doc = Document()

    # Title
    title = doc.add_heading("Breast Cancer AI Analysis Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in title.runs:
        run.font.color.rgb = RGBColor(37, 99, 235)

    doc.add_paragraph(f"Report ID: {report_id}")
    doc.add_paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph("")

    # Patient Information
    doc.add_heading("Patient Information", level=1)
    table = doc.add_table(rows=6, cols=2)
    table.style = "Light Grid Accent 1"
    fields = [
        ("Age at Diagnosis", f"{patient_data.get('age', 'N/A')}"),
        ("ER Status", f"{patient_data.get('er_status', 'N/A')}"),
        ("PR Status", f"{patient_data.get('pr_status', 'N/A')}"),
        ("HER2 Status", f"{patient_data.get('her2_status', 'N/A')}"),
        ("Tumor Size", f"{patient_data.get('tumor_size', 'N/A')} mm"),
        ("Positive Lymph Nodes", f"{patient_data.get('lymph_nodes', 'N/A')}")
    ]
    for i, (label, value) in enumerate(fields):
        table.rows[i].cells[0].text = label
        table.rows[i].cells[1].text = str(value)

    doc.add_paragraph("")

    # ML Predictions
    doc.add_heading("ML Model Predictions", level=1)

    doc.add_heading("Clinical/Biomarker Model", level=2)
    doc.add_paragraph(f"Predicted PAM50 Subtype: {clinical_result.get('prediction', 'N/A')}")
    doc.add_paragraph(f"Model Confidence: {clinical_result.get('confidence', 'N/A')}%")

    if image_result:
        doc.add_heading("Ultrasound Image Model", level=2)
        doc.add_paragraph(f"Image Prediction: {image_result.get('prediction', 'N/A')}")
        doc.add_paragraph(f"Model Confidence: {image_result.get('confidence', 'N/A')}%")

    doc.add_paragraph("")

    # AI Agent Analysise
    doc.add_heading("AI Agent Analysis & Documentation", level=1)

    if ai_analysis.get("summary"):
        doc.add_heading("Executive Summary", level=2)
        doc.add_paragraph(ai_analysis["summary"])

    if ai_analysis.get("risk_assessment"):
        doc.add_heading("Risk Assessment", level=2)
        doc.add_paragraph(ai_analysis["risk_assessment"])

    if ai_analysis.get("subtype_analysis"):
        doc.add_heading("Subtype Analysis", level=2)
        doc.add_paragraph(ai_analysis["subtype_analysis"])

    if ai_analysis.get("biomarker_interpretation"):
        doc.add_heading("Biomarker Interpretation", level=2)
        doc.add_paragraph(ai_analysis["biomarker_interpretation"])

    if ai_analysis.get("treatment_considerations"):
        doc.add_heading("Treatment Considerations", level=2)
        doc.add_paragraph(ai_analysis["treatment_considerations"])

    if ai_analysis.get("recommended_actions"):
        doc.add_heading("Recommended Actions", level=2)
        for action in ai_analysis["recommended_actions"]:
            doc.add_paragraph(action, style="List Bullet")

    if ai_analysis.get("additional_tests"):
        doc.add_heading("Additional Tests Recommended", level=2)
        for test in ai_analysis["additional_tests"]:
            doc.add_paragraph(test, style="List Bullet")

    if ai_analysis.get("follow_up_plan"):
        doc.add_heading("Follow-Up Plan", level=2)
        doc.add_paragraph(ai_analysis["follow_up_plan"])

    if ai_analysis.get("confidence_notes"):
        doc.add_heading("Confidence & Limitations", level=2)
        doc.add_paragraph(ai_analysis["confidence_notes"])

    # Disclaimer
    doc.add_paragraph("")
    disclaimer = doc.add_paragraph()
    disclaimer_run = disclaimer.add_run(
        "DISCLAIMER: This report is generated by an AI system for research and educational purposes only. "
        "It should not be used as a substitute for professional medical diagnosis or treatment. "
        "Always consult with qualified healthcare professionals for medical decisions."
    )
    disclaimer_run.font.size = Pt(9)
    disclaimer_run.font.italic = True
    disclaimer_run.font.color.rgb = RGBColor(128, 128, 128)

    # Save
    filepath = os.path.join(REPORT_FOLDER, f"breast_cancer_report_{report_id}.docx")
    doc.save(filepath)
    return filepath


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("patient_portal" if session.get("role") == "patient" else "home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id, username, password_hash, role FROM app_users WHERE username = %s",
                    (username,)
                )
                user = cursor.fetchone()

        if user and check_password_hash(user[2], password):
            session.clear()
            session["user_id"] = user[0]
            session["username"] = user[1]
            session["role"] = user[3]
            get_csrf_token()
            return redirect(url_for("patient_portal" if user[3] == "patient" else "home"))

        flash("Username or password is incorrect.", "error")

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
@require_role("admin")
def home():
    clinical_prediction = None
    clinical_confidence = None
    clinical_probabilities = None
    image_prediction = None
    image_confidence = None
    image_probabilities = None
    uploaded_image = None
    error = None
    report = None
    analysis_id = None
    ai_analysis = None
    patients = []
    selected_patient = None
    medical_requests = []
    analysis_history = []

    with database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, username FROM app_users WHERE role = 'patient' ORDER BY username"
            )
            patients = [{"id": row[0], "username": row[1]} for row in cursor.fetchall()]

            selected_patient_id = request.form.get("patient_id") if request.method == "POST" else request.args.get("patient_id")
            if selected_patient_id is None:
                selected_patient_id = session.get("selected_patient_id")

            try:
                selected_patient_id = int(selected_patient_id) if selected_patient_id else None
            except (TypeError, ValueError):
                selected_patient_id = None

            if selected_patient_id:
                cursor.execute(
                    "SELECT id, username FROM app_users WHERE id = %s AND role = 'patient'",
                    (selected_patient_id,)
                )
                patient_row = cursor.fetchone()
                if patient_row:
                    selected_patient = {"id": patient_row[0], "username": patient_row[1]}
                    session["selected_patient_id"] = patient_row[0]
                else:
                    session.pop("selected_patient_id", None)
            else:
                session.pop("selected_patient_id", None)

            cursor.execute("""
                SELECT r.id, r.patient_id, u.username, r.message, r.status, r.created_at
                FROM medical_requests r
                JOIN app_users u ON u.id = r.patient_id
                ORDER BY r.created_at DESC
                LIMIT 50
            """)
            medical_requests = [
                {"id": row[0], "patient_id": row[1], "username": row[2],
                 "message": row[3], "status": row[4], "created_at": row[5]}
                for row in cursor.fetchall()
            ]

    if request.method == "POST":
        try:
            if not selected_patient:
                raise ValueError("Select a patient before running an analysis.")

            er_status = request.form.get("er_status")
            pr_status = request.form.get("pr_status")
            her2_status = request.form.get("her2_status")
            age = float(request.form.get("age"))
            tumor_size = float(request.form.get("tumor_size"))
            lymph_nodes = float(request.form.get("lymph_nodes"))

            if age <= 0:
                raise ValueError("Age must be greater than 0.")
            if tumor_size < 0:
                raise ValueError("Tumor size cannot be negative.")
            if lymph_nodes < 0:
                raise ValueError("Lymph node count cannot be negative.")

            input_data = pd.DataFrame({
                "er_status": [encode_marker(er_status)],
                "pr_status": [encode_marker(pr_status)],
                "her2_status": [encode_marker(her2_status)],
                "age_at_diagnosis": [age],
                "tumor_size": [tumor_size],
                "lymph_nodes_examined_positive": [lymph_nodes]
            })

            prediction_encoded = clinical_model.predict(input_data)[0]
            clinical_prediction = target_encoder.inverse_transform([prediction_encoded])[0]

            probability_array = clinical_model.predict_proba(input_data)[0]
            clinical_confidence = round(float(max(probability_array)) * 100, 2)

            clinical_probabilities = []
            for class_name, probability in zip(target_encoder.classes_, probability_array):
                clinical_probabilities.append({
                    "class": class_name,
                    "probability": round(float(probability) * 100, 2)
                })
            clinical_probabilities.sort(key=lambda x: x["probability"], reverse=True)

            report = {
                "ER": er_status,
                "PR": pr_status,
                "HER2": her2_status,
                "Age": age,
                "Tumor Size": tumor_size,
                "Lymph Nodes": lymph_nodes
            }

            image_file = request.files.get("image")
            if image_file:
                if image_file.filename == "":
                    raise ValueError("Please select an image.")
                if not allowed_file(image_file.filename):
                    raise ValueError("Only PNG, JPG and JPEG images are allowed.")

                filename = str(int(time.time())) + "_" + secure_filename(image_file.filename)
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                image_file.save(image_path)
                uploaded_image = "/uploads/" + filename

                image_prediction, image_confidence, image_probabilities = predict_image(image_path)

            analysis_id = uuid.uuid4().hex
            patient_data = {
                "er_status": er_status,
                "pr_status": pr_status,
                "her2_status": her2_status,
                "age": age,
                "tumor_size": tumor_size,
                "lymph_nodes": lymph_nodes,
            }
            clinical_result = {
                "prediction": str(clinical_prediction),
                "confidence": clinical_confidence,
                "probabilities": clinical_probabilities,
            }
            image_result = None
            if image_prediction:
                image_result = {
                    "prediction": image_prediction,
                    "confidence": image_confidence,
                    "probabilities": image_probabilities,
                }
            saved_payload = {
                "patient_data": patient_data,
                "clinical_result": clinical_result,
                "image_result": image_result,
                "report": report,
                "uploaded_image": uploaded_image,
                "ai_analysis": None,
            }
            with database_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """INSERT INTO analysis_reports (id, patient_id, payload)
                           VALUES (%s, %s, %s::jsonb)""",
                        (analysis_id, selected_patient["id"], json.dumps(saved_payload))
                    )
        except Exception as e:
            error = str(e)
            analysis_id = None

    if selected_patient:
        analysis_history = load_patient_analysis_history(selected_patient["id"])
        if request.method == "GET" and analysis_history:
            requested_analysis_id = request.args.get("analysis_id")
            saved_analysis = next(
                (item for item in analysis_history if item["id"] == requested_analysis_id),
                analysis_history[0]
            )
            analysis_id = saved_analysis["id"]
            payload = saved_analysis["payload"]
            patient_data = payload.get("patient_data", {})
            clinical_result = payload.get("clinical_result", {})
            image_result = payload.get("image_result")
            report = payload.get("report") or {
                "ER": patient_data.get("er_status"),
                "PR": patient_data.get("pr_status"),
                "HER2": patient_data.get("her2_status"),
                "Age": patient_data.get("age"),
                "Tumor Size": patient_data.get("tumor_size"),
                "Lymph Nodes": patient_data.get("lymph_nodes"),
            }
            clinical_prediction = clinical_result.get("prediction")
            clinical_confidence = clinical_result.get("confidence")
            clinical_probabilities = clinical_result.get("probabilities")
            image_prediction = image_result.get("prediction") if image_result else None
            image_confidence = image_result.get("confidence") if image_result else None
            image_probabilities = image_result.get("probabilities") if image_result else None
            uploaded_image = payload.get("uploaded_image")
            ai_analysis = payload.get("ai_analysis")

    return render_template(
        "index.html",
        clinical_prediction=clinical_prediction,
        clinical_confidence=clinical_confidence,
        clinical_probabilities=clinical_probabilities,
        image_prediction=image_prediction,
        image_confidence=image_confidence,
        image_probabilities=image_probabilities,
        uploaded_image=uploaded_image,
        error=error,
        report=report,
        clinical_accuracy=round(clinical_accuracy * 100, 2),
        patients=patients,
        selected_patient=selected_patient,
        medical_requests=medical_requests,
        analysis_id=analysis_id,
        ai_analysis=ai_analysis,
        analysis_history=analysis_history,
    )


@app.route("/admin/patients", methods=["POST"])
@require_role("admin")
def create_patient():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    if not username or len(username) > 100:
        flash("Enter a username of up to 100 characters.", "error")
        return redirect(url_for("home"))
    if len(password) < 12:
        flash("Patient passwords must be at least 12 characters.", "error")
        return redirect(url_for("home"))

    try:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO app_users (username, password_hash, role)
                       VALUES (%s, %s, 'patient') RETURNING id""",
                    (username, generate_password_hash(password))
                )
                patient_id = cursor.fetchone()[0]
    except Exception as error:
        if getattr(error, "sqlstate", None) == "23505" or getattr(error, "pgcode", None) == "23505":
            flash("That username is already in use.", "error")
            return redirect(url_for("home"))
        raise

    flash("Patient account created. Share the username and password securely.", "success")
    return redirect(url_for("home", patient_id=patient_id))


@app.route("/admin/requests/<int:request_id>/close", methods=["POST"])
@require_role("admin")
def close_medical_request(request_id):
    with database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE medical_requests SET status = 'Reviewed' WHERE id = %s",
                (request_id,)
            )
    return redirect(url_for("home"))


@app.route("/patient")
@require_role("patient")
def patient_portal():
    with database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT id, payload, created_at FROM analysis_reports
                   WHERE patient_id = %s ORDER BY created_at DESC""",
                (session["user_id"],)
            )
            reports = [
                {"id": row[0], "payload": get_json_payload(row[1]), "created_at": row[2]}
                for row in cursor.fetchall()
            ]
            cursor.execute(
                """SELECT message, status, created_at FROM medical_requests
                   WHERE patient_id = %s ORDER BY created_at DESC""",
                (session["user_id"],)
            )
            medical_requests = [
                {"message": row[0], "status": row[1], "created_at": row[2]}
                for row in cursor.fetchall()
            ]
    return render_template(
        "patient.html",
        reports=reports,
        medical_requests=medical_requests,
        username=session.get("username"),
    )


@app.route("/patient/requests", methods=["POST"])
@require_role("patient")
def request_medical_attention():
    message = request.form.get("message", "").strip()
    if not message or len(message) > 1000:
        flash("Enter a message of up to 1,000 characters.", "error")
    else:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO medical_requests (patient_id, message) VALUES (%s, %s)",
                    (session["user_id"], message)
                )
        flash("Your request has been sent to the care team.", "success")
    return redirect(url_for("patient_portal"))


@app.route("/generate-report", methods=["POST"])
@require_role("admin")
def generate_report():
    """API endpoint to generate AI documentation report."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        analysis_id = data.get("analysis_id")
        selected_patient_id = session.get("selected_patient_id")
        if not analysis_id or not selected_patient_id:
            return jsonify({"error": "Select a patient and run an analysis first."}), 400

        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT payload FROM analysis_reports
                       WHERE id = %s AND patient_id = %s""",
                    (analysis_id, selected_patient_id)
                )
                row = cursor.fetchone()
        if not row:
            return jsonify({"error": "Analysis not found for the selected patient."}), 404

        saved_payload = get_json_payload(row[0])
        patient_data = saved_payload["patient_data"]
        clinical_result = saved_payload["clinical_result"]
        image_result = saved_payload["image_result"]

        # Build prompt and call Gemini
        prompt = build_gemini_prompt(patient_data, clinical_result, image_result)
        raw_response = call_gemini_agent(prompt)
        ai_analysis = parse_gemini_response(raw_response)

        # Generate .docx report
        generate_docx_report(patient_data, clinical_result, image_result, ai_analysis, analysis_id)
        saved_payload["ai_analysis"] = ai_analysis
        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE analysis_reports SET payload = %s::jsonb
                       WHERE id = %s AND patient_id = %s""",
                    (json.dumps(saved_payload), analysis_id, selected_patient_id)
                )

        return jsonify({
            "success": True,
            "report_id": analysis_id,
            "ai_analysis": ai_analysis,
            "download_url": f"/download-report/{analysis_id}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download-report/<report_id>")
def download_report(report_id):
    """Download a generated .docx report."""
    if not session.get("user_id"):
        return redirect(url_for("login"))
    if session.get("role") not in ("admin", "patient"):
        return redirect(url_for("login"))

    # Sanitize report_id to prevent path traversal
    safe_id = secure_filename(report_id)
    if safe_id != report_id:
        return "Report not found", 404

    with database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT patient_id FROM analysis_reports WHERE id = %s",
                (safe_id,)
            )
            report_row = cursor.fetchone()
    if not report_row:
        return "Report not found", 404
    if session.get("role") == "patient" and report_row[0] != session.get("user_id"):
        return "Report not found", 404

    filename = f"breast_cancer_report_{safe_id}.docx"
    filepath = os.path.join(REPORT_FOLDER, filename)

    if not os.path.exists(filepath):
        return "Report not found", 404

    return send_file(filepath, as_attachment=True, download_name=filename)


@app.route("/uploads/<filename>")
@require_role("admin")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
        host="127.0.0.1",
        port=5000,
    )