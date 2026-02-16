"""
MedClinic Voice Scribe & Workflow Agent - FastAPI Backend

Uses MedGemma (Google's open medical LLM) for SOAP note generation
and OpenAI Whisper for speech-to-text transcription.
All models run locally for privacy, speed, and Edge AI deployment.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json
import os
import tempfile

# ML/AI imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL MODEL INITIALIZATION (lazy-loaded for performance)
# ============================================================================

# Ensure ffmpeg is available for Whisper (fix for missing system ffmpeg)
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + ffmpeg_dir
    logger.info(f"Added ffmpeg to PATH via imageio_ffmpeg: {ffmpeg_dir}")
except ImportError:
    logger.warning("imageio-ffmpeg not found, relying on system PATH for ffmpeg")
except Exception as e:
    logger.error(f"Failed to setup ffmpeg via imageio_ffmpeg: {e}")

class ModelCache:
    """Global cache for MedGemma and Whisper models."""
    _medgemma_tokenizer = None
    _medgemma_model = None
    _whisper_model = None
    
    @classmethod
    def get_medgemma_tokenizer(cls):
        """Lazy-load MedGemma tokenizer."""
        if cls._medgemma_tokenizer is None:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("transformers library not available")
            logger.info("Loading MedGemma tokenizer...")
            # Using the official MedGemma model from Google
            cls._medgemma_tokenizer = AutoTokenizer.from_pretrained(
                "google/medgemma-2b",
                trust_remote_code=True
            )
        return cls._medgemma_tokenizer
    
    @classmethod
    def get_medgemma_model(cls):
        """Lazy-load MedGemma model."""
        if cls._medgemma_model is None:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("transformers library not available")
            logger.info("Loading MedGemma model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._medgemma_model = AutoModelForCausalLM.from_pretrained(
                "google/medgemma-2b",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            if device == "cpu":
                cls._medgemma_model.to(device)
            logger.info(f"MedGemma loaded on {device}")
        return cls._medgemma_model
    
    @classmethod
    def get_whisper_model(cls, model_size: str = "base"):
        """Lazy-load Whisper model for speech-to-text."""
        if cls._whisper_model is None:
            if not WHISPER_AVAILABLE:
                raise RuntimeError("openai-whisper library not available")
            logger.info(f"Loading Whisper ({model_size}) model...")
            cls._whisper_model = whisper.load_model(model_size)
        return cls._whisper_model

app = FastAPI(
    title="MedClinic Voice Scribe API",
    description="Backend for medical visit transcription and SOAP note generation using MedGemma",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = current_dir
if os.path.exists(os.path.join(static_dir, "index.html")):
    app.add_route("/", lambda request: FileResponse(os.path.join(static_dir, "index.html")))
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ---------------------------------------------------------------------------
# Pydantic models (clear, typed, and ready to extend)
# ---------------------------------------------------------------------------

class TranscriptRequest(BaseModel):
    """Request body: raw transcript text."""
    transcript: str = Field(
        ..., 
        min_length=1,
        example="Patient reports 3 days of productive cough and low-grade fever. Denies chest pain. Vital signs: BP 120/80, HR 98, Temp 38.2C. Lungs: clear to auscultation."
    )

class AudioTranscriptRequest(BaseModel):
    """Metadata for audio file processing."""
    language: str = "en"


class VisitRequest(BaseModel):
    """Request body: raw transcript text (for now, plain text)."""
    transcript: str = Field(
        ..., 
        min_length=1,
        example="Pt reports 3 days of cough, low‑grade fever; no chest pain."
    )


class SOAPNote(BaseModel):
    """Structured SOAP note. Plan is a list of discrete steps (easier to present/action)."""
    subjective: str
    objective: str
    assessment: str
    plan: List[str]


class WorkflowAction(BaseModel):
    """Single actionable workflow item the UI can turn into tasks/messages/orders."""
    action_type: str
    description: str
    timing: Optional[str] = None
    priority: str = "normal"


class VisitResponse(BaseModel):
    transcript: str
    soap_note: SOAPNote
    actions: List[WorkflowAction]
    generated_at: str


class TranscriptResponse(BaseModel):
    """Response from audio transcription endpoint."""
    transcript: str
    audio_duration_seconds: float
    model_used: str
    generated_at: str


# ---------------------------------------------------------------------------
# MedGemma SOAP Generation (using actual LLM inference)
# ---------------------------------------------------------------------------

def generate_soap_with_medgemma(transcript: str) -> Dict[str, Any]:
    """
    Use MedGemma to generate structured SOAP note from clinical transcript.
    
    Falls back to rule-based if MedGemma unavailable.
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers not available, using rule-based fallback")
        return generate_soap_fallback(transcript)
    
    try:
        tokenizer = ModelCache.get_medgemma_tokenizer()
        model = ModelCache.get_medgemma_model()
        
        # Construct a prompt that asks MedGemma to generate a SOAP note
        prompt = f"""Generate a structured medical SOAP note from the following clinical transcript.

TRANSCRIPT:
{transcript}

Respond in JSON format with the following structure:
{{
    "subjective": "Patient's reported symptoms and history",
    "objective": "Vital signs, exam findings, measurements",
    "assessment": "Clinical impression and differential diagnoses",
    "plan": ["Treatment/action 1", "Treatment/action 2", ...]
}}

SOAP NOTE:"""
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logger.info("Running MedGemma inference...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode output
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                json_str = response_text[json_start:json_end]
                soap_data = json.loads(json_str)
                
                # Ensure plan is a list
                if isinstance(soap_data.get("plan"), str):
                    soap_data["plan"] = [soap_data["plan"]]
                
                logger.info("Successfully generated SOAP note with MedGemma")
                return {"soap_note": soap_data, "actions": generate_workflow_actions(soap_data)}
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse MedGemma JSON output: {e}, using fallback")
                return generate_soap_fallback(transcript)
        else:
            logger.warning("No JSON found in MedGemma output, using fallback")
            return generate_soap_fallback(transcript)
            
    except Exception as e:
        logger.error(f"MedGemma inference failed: {e}, using fallback", exc_info=True)
        return generate_soap_fallback(transcript)


def generate_workflow_actions(soap_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate workflow actions from SOAP data."""
    actions: List[Dict[str, Any]] = []
    
    assessment_text = (soap_data.get("assessment") or "").lower()
    plan_text = " ".join([str(p).lower() for p in soap_data.get("plan", [])])
    
    # Determine urgency and follow-up timing
    if any(w in assessment_text for w in ["urgent", "severe", "critical", "emergency"]):
        follow_up_timing = "24 hours"
        priority = "urgent"
    elif any(w in assessment_text for w in ["chronic", "stable"]):
        follow_up_timing = "2-4 weeks"
        priority = "normal"
    else:
        follow_up_timing = "1-2 weeks"
        priority = "normal"
    
    actions.append({
        "action_type": "follow_up",
        "description": f"Schedule follow-up to reassess symptoms and treatment response.",
        "timing": follow_up_timing,
        "priority": priority,
    })
    
    if any(w in plan_text for w in ["lab", "test", "imaging", "x-ray", "cbc", "bloodwork"]):
        actions.append({
            "action_type": "test_order",
            "description": "Order diagnostic tests suggested in plan.",
            "timing": "within 3 days",
            "priority": "normal",
        })
    
    if any(w in plan_text for w in ["prescribe", "prescription", "medication", "antibiotic"]):
        actions.append({
            "action_type": "prescription",
            "description": "Create e-prescription per visit discussion.",
            "timing": "immediate",
            "priority": "normal",
        })
    
    actions.append({
        "action_type": "patient_message",
        "description": f"Patient follow-up message: Schedule reassessment in {follow_up_timing}. Follow care plan instructions.",
        "timing": "immediate",
        "priority": "normal",
    })
    
    return actions


def generate_soap_fallback(transcript: str) -> Dict[str, Any]:
    """
    Rule-based (no-ML) fallback generator that returns a dict with keys:
    - soap_note: dict matching `SOAPNote`
    - actions: list of dicts matching `WorkflowAction`
    """
    text = (transcript or "").strip()
    text_l = text.lower()

    # SUBJECTIVE
    subjective_parts: List[str] = []
    symptom_keywords = [
        "cough", "fever", "pain", "headache", "dizzy", "nausea",
        "shortness of breath", "sob", "swelling", "rash", "weakness",
    ]
    found_symptoms = [k for k in symptom_keywords if k in text_l]
    
    if found_symptoms:
        subjective_parts.append("Symptoms: " + ", ".join(found_symptoms))
    
    if "day" in text_l or "week" in text_l or "month" in text_l:
        subjective_parts.append("Duration: reported in transcript")
    
    if not subjective_parts:
        subjective_parts.append("Chief complaint described in transcript; details recorded above.")
    
    subjective = " ".join(subjective_parts)

    # OBJECTIVE
    objective_parts: List[str] = ["Exam: basic office exam documented."]
    
    if "blood pressure" in text_l or "bp " in text_l:
        objective_parts.append("Blood pressure noted in transcript (value not parsed).")
    if "heart rate" in text_l or "pulse" in text_l:
        objective_parts.append("Heart rate noted in transcript (value not parsed).")
    if "temp" in text_l or "fever" in text_l:
        objective_parts.append("Temperature reported by patient or measured in clinic.")
    
    exam_findings = ["swelling", "erythema", "redness", "wheezes", "rales", "murmur"]
    found_findings = [f for f in exam_findings if f in text_l]
    
    if found_findings:
        objective_parts.append("Exam findings: " + ", ".join(found_findings))
    
    objective = " ".join(objective_parts)

    # ASSESSMENT
    assessment_suggestions: List[str] = []
    differentials: List[str] = []
    
    if "fever" in text_l or "temperature" in text_l or "chills" in text_l:
        assessment_suggestions.append("Likely viral or bacterial infection (site per history).")
        differentials.extend(["viral infection", "bacterial infection", "COVID-19/other respiratory"])
    
    if "cough" in text_l or "wheeze" in text_l or "shortness of breath" in text_l or "sob" in text_l:
        assessment_suggestions.append("Respiratory etiology; consider bronchitis, asthma exacerbation, or pneumonia.")
        differentials.extend(["acute bronchitis", "asthma exacerbation", "pneumonia"])
    
    if "pain" in text_l or "injury" in text_l or "sprain" in text_l:
        assessment_suggestions.append("Musculoskeletal pain/strain.")
        differentials.extend(["soft tissue strain", "sprain", "fracture (if high‑risk trauma)"])
    
    if not assessment_suggestions:
        assessment_suggestions.append("Symptoms consistent with a common primary care presentation; further workup as below.")
        differentials.append("symptom-based differential (see plan)")
    
    assessment = " ".join(assessment_suggestions) + " Differential: " + ", ".join(dict.fromkeys(differentials))

    # PLAN
    plan_steps: List[str] = []
    plan_steps.append("Provide symptomatic care: rest, hydration, acetaminophen/ibuprofen as needed (per allergies).")
    
    if any(w in text_l for w in ["prescribe", "prescription", "start", "give", "medication", "antibiotic"]):
        plan_steps.append("Prescribe medication as discussed in visit (see e-prescription).")
    
    if any(w in text_l for w in ["lab", "cbc", "xr", "x-ray", "imaging", "ct", "culture", "covid"]):
        plan_steps.append("Order indicated diagnostic tests (CBC, imaging, or pathogen testing as clinically indicated).")
    
    if any(w in text_l for w in ["refer", "referral", "specialist", "cardio", "ortho", "ent"]):
        plan_steps.append("Arrange referral to specialist as discussed (coordinate with scheduling).")
    
    if any(w in text_l for w in ["urgent", "severe", "worse", "emergency"]):
        plan_steps.append("Arrange urgent follow-up within 48 hours or advise ED if symptoms worsen.")
        follow_up_timing = "48 hours"
        priority = "urgent"
    elif any(w in text_l for w in ["chronic", "follow-up", "routine"]):
        plan_steps.append("Schedule routine follow-up in 2–4 weeks to reassess symptoms and response to treatment.")
        follow_up_timing = "2–4 weeks"
        priority = "normal"
    else:
        plan_steps.append("Routine follow-up in ~2 weeks or sooner if symptoms worsen.")
        follow_up_timing = "2 weeks"
        priority = "normal"
    
    if len(plan_steps) == 0:
        plan_steps.append("Provide symptomatic care and schedule reassessment if no improvement.")

    # WORKFLOW ACTIONS
    actions: List[Dict[str, Any]] = []
    
    actions.append({
        "action_type": "follow_up",
        "description": "Schedule follow-up to reassess symptoms and treatment response.",
        "timing": follow_up_timing,
        "priority": priority,
    })
    
    if any("order indicated diagnostic tests" in s for s in plan_steps) or any(
        w in text_l for w in ["lab", "cbc", "x-ray", "xr", "imaging", "ct", "culture"]
    ):
        actions.append({
            "action_type": "test_order",
            "description": "Order diagnostic tests suggested in plan (e.g., CBC, imaging, pathogen testing).",
            "timing": "within 3 days",
            "priority": "normal",
        })
    
    if any(w in text_l for w in ["prescribe", "prescription", "antibiotic", "azithro", "amoxi", "ibuprofen"]):
        actions.append({
            "action_type": "prescription",
            "description": "Create e-prescription per visit discussion.",
            "timing": "immediate",
            "priority": "normal",
        })
    
    sms_lines: List[str] = []
    sms_lines.append("Thanks for your visit today. ")
    sms_lines.append("Follow the plan: ")
    sms_lines.append(" ".join(plan_steps[:2]))
    sms_lines.append(f"Please book follow-up in {follow_up_timing} or sooner if worse.")
    
    patient_message = " ".join(sms_lines).strip()
    if len(patient_message) > 320:
        patient_message = patient_message[:317] + "..."
    
    actions.append({
        "action_type": "patient_message",
        "description": patient_message,
        "timing": "immediate",
        "priority": "normal",
    })

    soap_note = {
        "subjective": subjective,
        "objective": objective,
        "assessment": assessment,
        "plan": plan_steps,
    }
    
    return {"soap_note": soap_note, "actions": actions}


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "0.1.0"
    }


@app.post("/api/visit/summarize", response_model=VisitResponse)
async def summarize_visit(request: VisitRequest):
    """
    Accepts { "transcript": "..." } and returns structured SOAP + actions.
    Uses MedGemma for inference, falls back to rule-based if unavailable.
    """
    try:
        if not request.transcript or not request.transcript.strip():
            raise HTTPException(
                status_code=400,
                detail="transcript must be a non-empty string"
            )
        
        logger.info(f"Processing transcript of length {len(request.transcript)}")
        
        out = generate_soap_with_medgemma(request.transcript)
        
        soap = SOAPNote(**out["soap_note"])
        actions = [WorkflowAction(**a) for a in out["actions"]]
        
        response = VisitResponse(
            transcript=request.transcript,
            soap_note=soap,
            actions=actions,
            generated_at=datetime.utcnow().isoformat() + "Z",
        )
        
        logger.info("Successfully generated SOAP note and actions")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing transcript: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/api/transcribe", response_model=TranscriptResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file (mp3, wav, m4a, etc.) and returns transcribed text.
    Uses OpenAI's Whisper model for speech-to-text.
    """
    try:
        if not WHISPER_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Whisper model not available. Install with: pip install openai-whisper"
            )
        
        # Accept any audio file - Whisper is flexible
        # Just check it's not empty
        if not file.content_type or "audio" not in file.content_type.lower():
            if file.filename and not any(f in file.filename.lower() for f in ['.mp3', '.wav', '.m4a', '.ogg', '.webm', '.flac', '.aac']):
                logger.warning(f"Unusual file type for: {file.filename}, attempting transcription anyway")
            # Still allow it - Whisper can handle most formats
        
        logger.info(f"Transcribing audio file: {file.filename}")
        
        # Save uploaded file temporarily with proper extension detection
        file_ext = ".wav"
        if file.filename:
            for ext in ['.mp3', '.wav', '.m4a', '.ogg', '.webm', '.flac', '.aac', '.opus']:
                if file.filename.lower().endswith(ext):
                    file_ext = ext
                    break
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Load Whisper model and transcribe
            whisper_model = ModelCache.get_whisper_model(model_size="base")
            result = whisper_model.transcribe(tmp_path)
            transcript = result["text"]
            duration = result.get("duration", 0)
            
            logger.info(f"Successfully transcribed audio. Length: {len(transcript)} chars")
            
            return TranscriptResponse(
                transcript=transcript,
                audio_duration_seconds=duration,
                model_used="openai/whisper-base",
                generated_at=datetime.utcnow().isoformat() + "Z",
            )
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MedClinic Voice Scribe API - Powered by MedGemma",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "summarize": "/api/visit/summarize (POST)",
            "transcribe": "/api/transcribe (POST with audio file)",
        },
        "models": {
            "medgemma": "google/medgemma-2b",
            "whisper": "openai/whisper-base",
        }
    }


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("MedClinic Voice Scribe API starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("MedClinic Voice Scribe API shutting down...")




if __name__ == "__main__":
    import uvicorn
    import os
    
    # Support both local (port 8000) and Hugging Face Spaces (port 7860)
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting MedClinic on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
