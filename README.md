---
title: MedClinic - SOAP Notes & Product Assistant
emoji: 🏥
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
license: mit
---

# 🏥 MedClinic - v2.0: Enhanced Medical AI Assistant

**AI-Powered SOAP Notes + Medicine Information Assistant using MedGemma**

[![MedGemma Impact Challenge](https://img.shields.io/badge/MedGemma-Impact%20Challenge-blue)](https://kaggle.com/competitions/med-gemma-impact-challenge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What's New (v2.0)

MedClinic has been **upgraded from a single-purpose SOAP note generator** into a **dual-mode AI medical assistant**:

| Feature | v1.0 (Original) | v2.0 (New) |
|---------|-----------------|-----------|
| **SOAP Notes** | ✅ Text & audio input | ✅ Enhanced + maintained |
| **Medicine Info** | ❌ None | ✅ NEW - Medicine Details + chat |
| **Safety Guardrails** | Basic disclaimer | ✅ Advanced safety checks |
| **Chat Interface** | ❌ None | ✅ NEW - Q&A on Medications |
| **Local Processing** | ✅ 100% local | ✅ 100% local |

---

## 🚀 Live Demo

**Try it now:** [https://huggingface.co/spaces/YOUR-USERNAME/medclinic-v2](https://huggingface.co/spaces/YOUR-USERNAME/medclinic-v2)

---

## ✨ Features

### Mode 1: SOAP Note Generation (Original)
**Clinical documentation for healthcare providers**

- 📝 **Text Input**: Paste clinical transcripts → Get structured SOAP notes
- 🎤 **Audio Input**: Upload audio recordings → Auto-transcribe + generate SOAP
- 🔄 **Workflow Actions**: Generates follow-ups, test orders, prescriptions
- ⚡ **Fast**: 5-30 seconds per visit (GPU/CPU)
- 🔒 **Private**: All processing on-device

**Perfect for:**
- Clinical documentation automation
- Provider efficiency
- Workflow integration

---

### Mode 2: Medicine Information Assistant (NEW)
**Educational information about common medications**

- 💊 **Medicine Query**: Choose common medications or type custom names to get key details
- 📋 **Structured Output**: 
  - Medicine name & description
  - Indications & dosage guidelines
  - Side effects & warnings
  - Clinical pharmacology notes & mechanism
- 💬 **Smart Chat**: Ask follow-up questions about the medicine
- 🛡️ **Safety First**: Blocks diagnostic/treatment advice, educational-only responses
- 📚 **No Medical Advice**: Always provides appropriate disclaimers

**Perfect for:**
- Healthcare professionals reviewing common drugs
- Patient education and training
- Quick offline reference for dosages, side effects, and pharmacology

---

## 📖 Quick Start

### Via Browser (Recommended)

1. **Visit the demo**: [HuggingFace Spaces link above]
2. **Choose a mode** at the top (SOAP Notes or Medicine Info)
3. **SOAP Mode**: Paste transcript or upload audio
4. **Medicine Info Mode**: Select or type a medication name to fetch its details and ask questions

### Local Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/medclinic.git
cd medclinic

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
python main.py

# 5. Open browser to http://localhost:8000
```

---

## 🎓 How to Use

### SOAP Notes Mode
```
1. Select "SOAP Notes" tab
2. Choose "Text" or "Audio" input
3. Paste clinical transcript or upload MP3/WAV file
4. Click "Generate SOAP"
5. View structured SOAP note + workflow actions
6. Copy and integrate into your system
```

### Medicine Info Mode
```
1. Select "Medicine Info" tab
2. Choose a medication from the dropdown or type a custom medication name
3. Click "Get Medicine Details"
4. View medicine details cards
5. Ask questions in the chat interface
6. Get educational medicine information + safety notes
```

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Frontend (UI)  │  ← Browser interface with dual modes
│  index.html     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI       │
│  Backend API    │
└────────┬────────┘
         │
    ┌────┴──────────┬─────────────┐
    ▼               ▼             ▼
┌──────────┐  ┌─────────────┐  ┌──────────┐
│MedGemma  │  │ Whisper     │  │SafetyMod │
│(SOAP)    │  │(transcribe) │  │(checks)  │
└──────────┘  └─────────────┘  └──────────┘
    (SOAP)        (SOAP)        (MEDICINE)
```

**Models Used:**
- **MedGemma-2B**: SOAP note generation & Medicine Details (if loaded)
- **Whisper-base**: Audio transcription
- **Local Rules Engine**: Pharmacological details fallback when offline
- **Safety Checker**: Medical guideline enforcement

---

## 🔒 Safety & Compliance

### Medical Safety Guardrails

**Medicine Information Mode includes:**
- ✅ Diagnostic keyword detection & blocking
- ✅ Treatment advice prevention
- ✅ Automatic safety disclaimers
- ✅ Educational-only responses
- ❌ Never prescribes, diagnoses, or treats

**Example (Blocked):**
```
User: "How do I treat a patient with this drug?"
System: ⚠️ I cannot provide treatment guidance. 
        Please consult a qualified healthcare professional.
```

**Example (Allowed):**
```
User: "What is this medicine used for?"
System: ✅ This medicine is used for [educational info].
        For clinical guidance, consult a healthcare provider.
```

### SOAP Notes Mode

- 📋 Clinical disclaimer on every note
- ✍️ Designed for provider review before use
- 🔍 Not a substitute for clinical judgment
- ✅ All processing local (no data stored)

---

## 📋 Models & Performance

| Component | Model | Size | Speed (GPU) | Speed (CPU) | Accuracy |
|-----------|-------|------|-------------|------------|----------|
| SOAP Gen | MedGemma-2B | 5GB RAM | ~3-5s | ~15-30s | Good |
| Transcribe | Whisper-base | 140MB | ~2-5s | ~10-15s | Excellent |
| Medicine Info | MedGemma / Offline | <1GB / Local | ~1-3s | ~1-5s | Excellent |

---

## 📝 API Endpoints

### SOAP Mode (Original)

```bash
# Generate SOAP from transcript
POST /api/visit/summarize
{
  "transcript": "Patient reports 3 days of cough..."
}

# Transcribe audio
POST /api/transcribe
Content-Type: multipart/form-data
file: <audio_file>
```

### Medicine Info Mode (NEW)

```bash
# Get medicine details
POST /api/product/analyze
Content-Type: application/json
{
  "name": "Amoxicillin"
}

# Chat about medicine
POST /api/product/chat
{
  "message": "What is this medicine used for?",
  "product_info": {...},
  "chat_history": [...]
}
```

---

## ⚙️ Configuration

### Environment Variables

```bash
PORT=8000                    # Server port
MODEL_SIZE=base              # Whisper model (base, small, medium, large)
DEVICE=auto                  # Model device (auto, cuda, cpu)
MAX_CHAT_HISTORY=10          # Max conversation memory
```

### Docker Deployment

```bash
# Build image
docker build -t medclinic:v2 .

# Run container
docker run -p 8000:8000 medclinic:v2
```

---

## 🧪 Testing

```bash
# Run all automated API tests
python test_api.py

# Test health endpoint manually
curl http://localhost:8000/health
```

---

## ⚠️ Important Disclaimers

### SOAP Notes Mode
**Clinical Disclaimer:**
MedClinic is an AI-assisted documentation tool for clinician support only. It is **NOT a substitute for professional medical judgment**. All generated notes should be reviewed and edited by a qualified healthcare provider before use in clinical care.

### Medicine Info Mode
**Medicine Information Disclaimer:**
This information is for educational purposes only about medications. It is **NOT medical advice, diagnosis, or treatment guidance**. Always consult with a qualified healthcare professional before using any medical product or starting any medication.

---

## 🤝 Contributing

```bash
# Fork the repo
git clone https://github.com/YOUR-USERNAME/medclinic.git

# Make changes
git checkout -b feature/my-feature

# Submit pull request
git push origin feature/my-feature
```

---

## 📚 Additional Resources

- 📖 [SOAP Notes Explained](https://en.wikipedia.org/wiki/SOAP_note)
- 🏥 [Medical Device Standards](https://www.fda.gov/medical-devices/)
- 🔬 [MedGemma Research](https://arxiv.org/abs/2406.02056)
- 🎙️ [Whisper Documentation](https://github.com/openai/whisper)

---

## 📄 License

MIT License - Free for research, education, and commercial use.

---

## Version History

- **v2.0** (May 2026) - Added Product Assistant mode, safety checks, chat interface
- **v1.0** (2026) - Original SOAP note generator with Whisper transcription

---

**Built with ❤️ for the MedGemma Impact Challenge 2026**

Questions? Open an issue on GitHub or visit our demo space!
