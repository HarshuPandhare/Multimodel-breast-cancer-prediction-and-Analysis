# 🎗️ Multimodal Breast Cancer Prediction & Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Google Gemini"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Pillow"/>
</p>

<p align="center">
  <strong>An AI-powered multimodal application for breast cancer prediction, image analysis, and automated report generation.</strong>
</p>

<p align="center">
  Combining text-based data and medical images with machine learning, deep learning, and generative AI.
</p>

---

## 📌 Overview

**Multimodal Breast Cancer Prediction and Analysis** is a machine learning and deep learning project designed to analyze two different types of input: **textual data and medical images**.

The application provides a web-based interface where users can submit supported text data or upload an image for model-based analysis. It also integrates Google's Gemini AI to generate an AI-assisted report based on the application's available analysis results.

By combining multiple data modalities with AI-generated reporting, the project demonstrates how machine learning, computer vision, web development, and generative AI can work together in a single application.

## ✨ Key Features

| Feature | Description |
|---|---|
| 📝 Text-Based Analysis | Processes supported text inputs for prediction using the application's text-analysis workflow. |
| 🖼️ Image-Based Prediction | Uses a trained deep learning image model to analyze uploaded images. |
| 🧠 Deep Learning | Uses PyTorch and Torchvision for image preprocessing and model inference. |
| 🤖 AI Report Generation | Integrates Google Gemini to generate AI-assisted reports. |
| 🌐 Web Application | Provides a Flask-powered interface for interacting with the models. |
| 📊 Model Evaluation | Includes training visualization assets such as accuracy, loss, and confusion matrix plots. |
| 📄 Report Support | Uses Python tooling for generating and handling documentation and reports. |

## 🛠️ Technology Stack

<p>
  <img src="https://skillicons.dev/icons?i=python,flask,pytorch,sklearn,html,css,git,github,vscode" alt="Technology stack"/>
</p>

- **Programming Language:** Python
- **Backend:** Flask
- **Deep Learning:** PyTorch, Torchvision
- **Machine Learning:** Scikit-learn
- **Image Processing:** Pillow
- **Generative AI:** Google Gemini API
- **Data Processing:** NumPy, Pandas
- **Model Persistence:** Joblib
- **Report Generation:** Python-docx
- **Configuration:** Pydantic, python-dotenv

## 🏗️ Project Architecture

```text
                    ┌─────────────────────────┐
                    │       User Interface    │
                    │      Flask Web App      │
                    └────────────┬────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   │                           │
          ┌────────▼────────┐        ┌─────────▼────────┐
          │   Text Input    │        │   Image Upload   │
          └────────┬────────┘        └─────────┬────────┘
                   │                           │
          ┌────────▼────────┐        ┌─────────▼────────┐
          │ Text Prediction │        │ Image Prediction │
          │    Pipeline     │        │  PyTorch Model   │
          └────────┬────────┘        └─────────┬────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 │
                       ┌─────────▼─────────┐
                       │  Analysis Results │
                       └─────────┬─────────┘
                                 │
                       ┌─────────▼─────────┐
                       │  Gemini AI Report │
                       └─────────┬─────────┘
                                 │
                       ┌─────────▼─────────┐
                       │  Results & Report │
                       └───────────────────┘
```

*Conceptual architecture: the exact processing flow depends on the implementation in the application.*

## 📂 Project Structure

```text
ML_project/
│
├── app.py                  # Flask application and prediction routes
├── train.py                # Model training workflow
├── train2.py               # Additional training workflow
├── train3.py               # Additional training workflow
├── requirements.txt        # Python dependencies
├── .gitignore              # Git exclusion rules
│
├── models/                 # Saved machine learning/deep learning models
├── templates/              # HTML templates for the web interface
├── data/                   # Dataset files (kept local if excluded)
├── uploads/                # Uploaded images and related files
├── reports/                # Generated or saved reports
│
├── accuracy_graph.png      # Model accuracy visualization
├── loss_graph.png          # Model loss visualization
└── confusion_matrix.png    # Classification evaluation visualization
```

## 🚀 Getting Started

Follow these steps to run the project locally.

### 1. Clone the repository

```bash
git clone https://github.com/HarshuPandhare/Multimodel-breast-cancer-prediction-and-Analysis.git
```

### 2. Navigate to the project directory

```bash
cd Multimodel-breast-cancer-prediction-and-Analysis
```

### 3. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** PyTorch installation can depend on your operating system, Python version, and whether you intend to use CPU or CUDA. Follow the official PyTorch installation instructions if the default installation does not work.

### 5. Configure environment variables

Create a `.env` file in the project root and add your Google Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Use the exact environment variable name expected by `app.py`. If the application uses a different name, update this example accordingly.

Obtain an API key through [Google AI Studio](https://aistudio.google.com/).

**Security:** Never commit your `.env` file or expose API keys in your source code or public repository.

### 6. Run the application

```bash
python app.py
```

Open the local address displayed in your terminal, commonly:

```text
http://127.0.0.1:5000
```

## 🧪 How to Use

1. Start the Flask application.
2. Open the application in your browser.
3. Choose the available text-based analysis or image-based prediction workflow.
4. Enter the required text data or upload a supported image.
5. Submit the input for model inference.
6. Review the prediction and any confidence information provided by the application.
7. Use the AI report-generation feature to request a report, if available in the interface.
8. Review or download the generated report if the application supports that option.

*The available inputs, outputs, and report options depend on the application's implemented features.*

## 📈 Model Evaluation

The repository includes visualizations for examining model performance:

- **Accuracy Graph:** Shows accuracy across training epochs, if recorded by the training workflow.
- **Loss Graph:** Helps visualize the model's optimization progress.
- **Confusion Matrix:** Helps inspect correct and incorrect classifications across supported classes.

These visualizations can help assess model behavior. They do not, by themselves, establish clinical accuracy or suitability for medical use.

## 🔐 Privacy & Security

- Keep API credentials in environment variables.
- Exclude `.env`, virtual environments, and other sensitive files from Git.
- Avoid committing identifiable patient information or private medical images.
- Review the retention and handling of uploaded images and generated reports before deployment.
- Use secure configuration and appropriate access controls before exposing the application publicly.

## ⚠️ Medical Disclaimer

**This project is intended for educational and research purposes only.**

Its predictions and AI-generated reports are experimental and must not be used as a substitute for professional medical advice, diagnosis, screening, or treatment. Model outputs may be incorrect, incomplete, or misleading.

Always consult a qualified healthcare professional for medical interpretation and decisions. The application should undergo appropriate clinical validation and regulatory review before any real-world clinical use.

## 🔮 Future Enhancements

- Integrate a more unified multimodal prediction pipeline.
- Improve model evaluation using appropriate validation and independent test datasets.
- Add explainability techniques for image predictions.
- Enhance report generation with structured prediction results and model limitations.
- Add authentication, secure file handling, and privacy controls.
- Deploy the application using a secure, production-ready infrastructure.

## 👨‍💻 Author

**Harshvardhan Pandhare**

GitHub: [@HarshuPandhare](https://github.com/HarshuPandhare)

Project Repository: [Multimodel Breast Cancer Prediction and Analysis](https://github.com/HarshuPandhare/Multimodel-breast-cancer-prediction-and-Analysis)

---

<p align="center">
  <strong>Built with Python, Machine Learning, Deep Learning, and Generative AI.</strong>
</p>

<p align="center">
  ⭐ If you find this project interesting, consider starring the repository!
</p>
