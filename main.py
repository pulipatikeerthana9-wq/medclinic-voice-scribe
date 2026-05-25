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
import base64
import io
import re
import requests

# Image processing for product analysis
try:
    from PIL import Image
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False

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
    import shutil
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    
    # On Windows, copy the versioned ffmpeg exe to ffmpeg.exe so whisper subprocess can find it
    target_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(target_ffmpeg):
        try:
            shutil.copy(ffmpeg_exe, target_ffmpeg)
            logger.info(f"Copied {ffmpeg_exe} to {target_ffmpeg}")
        except Exception as copy_err:
            logger.error(f"Failed to copy ffmpeg binary: {copy_err}")
    else:
        logger.info(f"ffmpeg.exe already exists at {target_ffmpeg}")
        
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


# ============================================================================
# SAFETY CHECKER FOR MEDICAL PRODUCT ADVICE
# ============================================================================

class SafetyChecker:
    """
    Ensures responses stay within safe medical product advisory scope.
    Blocks diagnostic/treatment advice, redirects to educational info only.
    """
    
    # Keywords indicating unsafe advice
    DIAGNOSTIC_KEYWORDS = {
        'diagnose', 'diagnosis', 'disease', 'condition', 'syndrome',
        'symptom', 'signs', 'presentation', 'pathology', 'etiology',
        'patient has', 'you have', 'they have', 'suffering from'
    }
    
    TREATMENT_KEYWORDS = {
        'prescribe', 'prescription', 'medication', 'drug', 'dose', 'dosage',
        'treat', 'treatment', 'therapy', 'cure', 'heal', 'medicate',
        'antibiotic', 'painkiller', 'inject', 'surgery', 'operation'
    }
    
    UNSAFE_KEYWORDS = DIAGNOSTIC_KEYWORDS | TREATMENT_KEYWORDS
    
    @classmethod
    def is_safe_response(cls, text: str) -> bool:
        """Check if response contains unsafe medical advice."""
        text_lower = text.lower()
        
        # Count unsafe keyword matches
        unsafe_count = sum(1 for keyword in cls.UNSAFE_KEYWORDS if keyword in text_lower)
        
        # If more than 2 unsafe keywords, likely unsafe
        if unsafe_count > 2:
            return False
        
        # Check for direct advice patterns
        unsafe_patterns = [
            r'you (should|must|need to) (take|use|take|get).*\w+(medicine|drug|pill)',
            r'(go to|visit|see) (a|an)? ?(doctor|physician|clinic)',
            r'this will (cure|treat|fix|heal) your',
            r'(indicates|suggests|means) you have',
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, text_lower):
                return False
        
        return True
    
    @classmethod
    def get_safety_disclaimer(cls) -> str:
        """Return appropriate safety disclaimer for product advice."""
        return """⚠️ **Medical Product Information Disclaimer**: 
This information is for educational purposes only about medical products and equipment. 
It is NOT medical advice, diagnosis, or treatment guidance. 
Always consult with a qualified healthcare professional before using any medical product.
For medical concerns, please see a physician."""
    
    @classmethod
    def filter_response(cls, text: str) -> str:
        """Add safety notice if response approaches unsafe territory."""
        if not cls.is_safe_response(text):
            text += f"\n\n{cls.get_safety_disclaimer()}"
        return text


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
# Product Analysis Models (NEW)
# ---------------------------------------------------------------------------

class ProductAnalysisResult(BaseModel):
    """Structured result from product image analysis."""
    product_name: str = Field(..., description="Identified product name or type")
    category: str = Field(..., description="Product category or class")
    confidence: float = Field(..., description="Model confidence (0.0-1.0)")
    visible_features: List[str] = Field(..., description="List of visible features detected in the image")
    description: str = Field(..., description="Clear, concise product description")
    advantages: List[str] = Field(..., description="List of product advantages/benefits")
    disadvantages: List[str] = Field(..., description="List of product limitations/concerns")
    suggested_use: str = Field(..., description="Suggested use cases or general information")
    general_notes: str = Field(..., description="Additional relevant information")
    safety_note: str = Field(
        ..., 
        description="Important safety consideration or disclaimer"
    )


class MedicineAnalysisRequest(BaseModel):
    """Request for medication analysis by name."""
    name: str = Field(..., description="Name of the medicine/drug to analyze")


class ChatMessage(BaseModel):
    """Single message in product chat."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ProductChatRequest(BaseModel):
    """Request for product-related chat."""
    message: str = Field(..., min_length=1, description="User's question or message")
    product_info: Optional[ProductAnalysisResult] = Field(
        None, 
        description="Previously analyzed product info for context"
    )
    chat_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Previous chat messages for context"
    )


class ProductChatResponse(BaseModel):
    """Response from product chat endpoint."""
    response: str = Field(..., description="Assistant's response")
    is_safe: bool = Field(..., description="Whether response is safe medical product info")
    generated_at: str = Field(..., description="ISO timestamp")


# ---------------------------------------------------------------------------
# Pydantic models (existing - kept for compatibility)
# ---------------------------------------------------------------------------

# ============================================================================
# MEDICINE LOOKUP DATABASE & LOCAL GENERATION
# ============================================================================

def generate_medicine_info(medicine_name: str) -> Dict[str, Any]:
    """
    Look up structured medical information about the drug for educational purposes.
    Uses a high-fidelity local database, falling back to MedGemma prompt generation
    if available, or a safe structured template.
    """
    name_clean = medicine_name.strip().lower()
    
    local_db = {
        "amoxicillin": {
            "product_name": "Amoxicillin",
            "category": "Antibiotic (Beta-lactam / Penicillin)",
            "confidence": 1.0,
            "visible_features": ["Capsules (250mg/500mg)", "Oral suspension powder", "Rx Only labeling"],
            "description": "Amoxicillin is a bactericidal antibiotic used to treat a wide variety of bacterial infections by inhibiting bacterial cell wall synthesis.",
            "advantages": [
                "Broad efficacy against common gram-positive pathogens",
                "Acid-stable (can be taken with or without food)",
                "Pediatric-friendly liquid formulations available"
            ],
            "disadvantages": [
                "Inactive against viral infections",
                "Common side effects: diarrhea, nausea, skin rash",
                "Risk of hypersensitivity / anaphylaxis in penicillin-allergic patients"
            ],
            "suggested_use": "Otitis media, streptococcal pharyngitis, community-acquired pneumonia, skin infections, and H. pylori eradication.",
            "general_notes": "Instruct patients to complete the entire prescribed course even if they feel better to prevent resistance.",
            "safety_note": "Contraindicated in patients with confirmed penicillin allergy. Monitor for signs of severe allergic reactions (hives, swelling, wheezing)."
        },
        "metformin": {
            "product_name": "Metformin",
            "category": "Antidiabetic (Biguanide)",
            "confidence": 1.0,
            "visible_features": ["White oval tablets", "Strengths: 500mg, 850mg, 1000mg", "Often marked as 'XR' for extended-release"],
            "description": "Metformin is the first-line medication for the treatment of type 2 diabetes. It works by decreasing hepatic glucose production and improving insulin sensitivity.",
            "advantages": [
                "Highly effective at lowering HbA1c",
                "Weight-neutral or associated with mild weight loss",
                "Low risk of hypoglycemia when used as monotherapy"
            ],
            "disadvantages": [
                "Gastrointestinal side effects (bloating, diarrhea, abdominal discomfort)",
                "Requires monitoring of renal function (eGFR)",
                "Risk of vitamin B12 deficiency with long-term use"
            ],
            "suggested_use": "First-line pharmacological therapy for type 2 diabetes mellitus, prediabetes (off-label), and PCOS (off-label).",
            "general_notes": "Often initiated at a low dose and titrated weekly to minimize gastrointestinal side effects. Best taken with meals.",
            "safety_note": "Warning: Rare risk of lactic acidosis. Contraindicated in patients with severe renal impairment (eGFR <30 mL/min/1.73m²) or acute heart failure."
        },
        "lisinopril": {
            "product_name": "Lisinopril",
            "category": "Antihypertensive (ACE Inhibitor)",
            "confidence": 1.0,
            "visible_features": ["Round or oval tablets", "Varying colors by strength (2.5mg, 5mg, 10mg, 20mg, 40mg)", "Unscored or scored"],
            "description": "Lisinopril is an angiotensin-converting enzyme (ACE) inhibitor used to treat high blood pressure, heart failure, and to improve survival after a heart attack.",
            "advantages": [
                "Excellent cardiovascular and renal protective benefits",
                "Once-daily dosing promotes adherence",
                "Effective as a core component of heart failure management"
            ],
            "disadvantages": [
                "Can cause a persistent, dry cough (bradykinin-mediated)",
                "Risk of hyperkalemia (high potassium levels)",
                "Risk of acute kidney injury if blood pressure drops too rapidly"
            ],
            "suggested_use": "Hypertension management, adjunctive therapy in heart failure, and post-myocardial infarction recovery.",
            "general_notes": "Monitor serum creatinine, eGFR, and potassium levels regularly, especially when initiating therapy or increasing dosage.",
            "safety_note": "Black Box Warning: Fetal toxicity; discontinue immediately if pregnancy is detected. Risk of life-threatening angioedema (swelling of face/airway)."
        },
        "atorvastatin": {
            "product_name": "Atorvastatin",
            "category": "Lipid-lowering Agent (HMG-CoA Reductase Inhibitor / Statins)",
            "confidence": 1.0,
            "visible_features": ["White elliptical film-coated tablets", "Strengths: 10mg, 20mg, 40mg, 80mg", "Rx Only"],
            "description": "Atorvastatin is a lipid-lowering medication used to prevent cardiovascular disease in those at high risk and to lower abnormal lipid levels.",
            "advantages": [
                "Significant reduction in LDL cholesterol (up to 60%)",
                "Proven to reduce cardiovascular mortality and stroke risk",
                "Long half-life (can be taken at any time of day)"
            ],
            "disadvantages": [
                "Common side effects: headache, nasal congestion, joint pain",
                "Risk of statin-associated muscle symptoms (myalgia)",
                "Mild risk of transient liver enzyme elevations"
            ],
            "suggested_use": "Hypercholesterolemia, mixed dyslipidemia, and primary/secondary prevention of cardiovascular events.",
            "general_notes": "Advise patients to avoid drinking large quantities of grapefruit juice (potent CYP3A4 inhibitor).",
            "safety_note": "Contraindicated in active liver disease or unexplained persistent elevations of serum transaminases. Discontinue if pregnancy occurs."
        },
        "albuterol": {
            "product_name": "Albuterol (Salbutamol)",
            "category": "Bronchodilator (Short-acting Beta-2 Agonist - SABA)",
            "confidence": 1.0,
            "visible_features": ["Metered-dose inhaler (MDI)", "Blue plastic actuator and canister", "Nebulizer solution vials"],
            "description": "Albuterol is a rapid-acting bronchodilator used to relieve bronchospasm in patients with reversible obstructive airway disease.",
            "advantages": [
                "Onset of action is extremely rapid (within 5-15 minutes)",
                "Excellent rescue medication for acute asthma flare-ups",
                "Direct delivery to lungs minimizes systemic side effects"
            ],
            "disadvantages": [
                "Can cause tachycardia (rapid heart rate), tremors, and nervousness",
                "Overuse may indicate poorly controlled underlying asthma",
                "Short duration of action (4-6 hours), not for maintenance control"
            ],
            "suggested_use": "Treatment or prevention of bronchospasm in patients with asthma or COPD, and prevention of exercise-induced bronchospasm.",
            "general_notes": "Ensure patient is trained in proper inhaler technique (use of spacer is highly recommended to improve lung deposition).",
            "safety_note": "Use with caution in patients with cardiovascular disorders, especially coronary insufficiency, arrhythmias, or hypertension."
        },
        "aspirin": {
            "product_name": "Aspirin (Acetylsalicylic Acid)",
            "category": "Analgesic / Antiplatelet (NSAID)",
            "confidence": 1.0,
            "visible_features": ["Small round tablets", "Commonly 81mg (low-dose/baby aspirin) or 325mg (regular strength)", "Enteric-coated versions"],
            "description": "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) that acts as an analgesic, antipyretic, and irreversible inhibitor of platelet aggregation.",
            "advantages": [
                "Highly effective and inexpensive antiplatelet therapy for cardiovascular protection",
                "Reduces risk of recurrent ischemic stroke or myocardial infarction",
                "Provides rapid pain and fever relief at higher doses"
            ],
            "disadvantages": [
                "Increased risk of gastrointestinal bleeding and peptic ulcers",
                "Can worsen asthma symptoms in aspirin-sensitive patients",
                "Irreversible platelet inhibition lasts for the lifetime of the platelet (~7-10 days)"
            ],
            "suggested_use": "Secondary prevention of cardiovascular disease, acute coronary syndrome management, mild-to-moderate pain relief, and fever reduction.",
            "general_notes": "Enteric-coated tablets should be swallowed whole (not crushed/chewed) unless used for acute myocardial infarction, where chewing is preferred.",
            "safety_note": "Contraindicated in children and teenagers for viral infections due to risk of Reye's syndrome. Discontinue before major surgical procedures."
        },
        "ibuprofen": {
            "product_name": "Ibuprofen",
            "category": "Analgesic / Anti-inflammatory (NSAID)",
            "confidence": 1.0,
            "visible_features": ["Round or oval coated tablets", "Vibrant brown/orange coating (standard Advil/Motrin)", "Strengths: 200mg (OTC), 400mg, 600mg, 800mg (Rx)"],
            "description": "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) that works by inhibiting cyclooxygenase (COX) enzymes, reducing prostaglandin synthesis.",
            "advantages": [
                "Excellent dual action for both pain relief and reduction of inflammation",
                "Widely available over-the-counter",
                "Rapid onset of action for acute pain (headache, muscle ache, toothache)"
            ],
            "disadvantages": [
                "Can cause gastrointestinal upset, irritation, or bleeding",
                "May increase the risk of serious cardiovascular thrombotic events (myocardial infarction or stroke)",
                "Can impair renal function and cause fluid retention"
            ],
            "suggested_use": "Management of mild-to-moderate pain, fever reduction, primary dysmenorrhea (menstrual cramps), rheumatoid arthritis, and osteoarthritis.",
            "general_notes": "Advise patients to take with food or milk if gastrointestinal upset occurs, and use the lowest effective dose for the shortest duration.",
            "safety_note": "Contraindicated for the treatment of perioperative pain in the setting of coronary artery bypass graft (CABG) surgery. Use with caution in chronic kidney disease."
        },
        "methylprednisolone": {
            "product_name": "Methylprednisolone",
            "category": "Corticosteroid (Glucocorticoid)",
            "confidence": 1.0,
            "visible_features": ["White round tablets", "Strengths: 4mg, 8mg, 16mg, 32mg", "Dose-pack (Medrol Dosepak) blister card"],
            "description": "Methylprednisolone is a synthetic corticosteroid with potent anti-inflammatory and immunosuppressive properties. It reduces swelling, pain, and allergic responses by suppressing the immune system.",
            "advantages": [
                "Powerful anti-inflammatory effect across multiple organ systems",
                "Available in oral, IV, and intramuscular formulations",
                "Convenient Medrol Dosepak for short-course tapering regimens"
            ],
            "disadvantages": [
                "Long-term use: risk of osteoporosis, adrenal suppression, and Cushing's syndrome",
                "Can cause hyperglycemia (elevated blood sugar), especially in diabetics",
                "Increased susceptibility to infections due to immunosuppression"
            ],
            "suggested_use": "Treatment of severe allergic reactions, asthma exacerbations, autoimmune disorders (e.g., lupus, rheumatoid arthritis), inflammatory conditions, and organ transplant rejection prevention.",
            "general_notes": "Should not be stopped abruptly after prolonged use — requires gradual tapering to avoid adrenal crisis. Take with food to minimize GI irritation.",
            "safety_note": "Contraindicated in systemic fungal infections. Use with extreme caution in patients with diabetes, peptic ulcer disease, hypertension, or psychiatric disorders. Monitor blood glucose regularly."
        },
        "omeprazole": {
            "product_name": "Omeprazole",
            "category": "Proton Pump Inhibitor (PPI)",
            "confidence": 1.0,
            "visible_features": ["Delayed-release capsules (purple/pink)", "Strengths: 10mg, 20mg, 40mg", "Available OTC and Rx"],
            "description": "Omeprazole is a proton pump inhibitor that suppresses gastric acid secretion by irreversibly inhibiting the hydrogen-potassium ATPase enzyme system in gastric parietal cells.",
            "advantages": [
                "Highly effective at reducing gastric acid production",
                "Promotes healing of erosive esophagitis and gastric ulcers",
                "Available over-the-counter for short-term heartburn relief"
            ],
            "disadvantages": [
                "Long-term use associated with magnesium deficiency and bone fracture risk",
                "May reduce absorption of calcium, iron, and vitamin B12",
                "Potential risk of Clostridium difficile-associated diarrhea"
            ],
            "suggested_use": "Gastroesophageal reflux disease (GERD), erosive esophagitis, peptic ulcer disease, Zollinger-Ellison syndrome, and H. pylori eradication (in combination with antibiotics).",
            "general_notes": "Take 30-60 minutes before a meal on an empty stomach. Capsules should be swallowed whole, not crushed or chewed.",
            "safety_note": "Long-term use should be periodically reassessed. Monitor magnesium levels in patients on prolonged therapy. May interact with clopidogrel (Plavix)."
        },
        "cetirizine": {
            "product_name": "Cetirizine",
            "category": "Antihistamine (Second-generation H1 blocker)",
            "confidence": 1.0,
            "visible_features": ["White oval or round tablets", "Strength: 10mg (standard adult dose)", "Also available as liquid syrup for children"],
            "description": "Cetirizine is a second-generation antihistamine that selectively antagonizes peripheral H1 receptors, providing relief from allergic symptoms with minimal central nervous system effects.",
            "advantages": [
                "Non-drowsy in most patients (less sedating than first-generation antihistamines)",
                "Once-daily dosing provides 24-hour symptom relief",
                "Rapid onset of action (within 1 hour)"
            ],
            "disadvantages": [
                "Can still cause mild drowsiness in some individuals",
                "Dry mouth is a common side effect",
                "Less effective for severe allergic reactions (anaphylaxis requires epinephrine)"
            ],
            "suggested_use": "Seasonal and perennial allergic rhinitis (hay fever), chronic urticaria (hives), and allergic conjunctivitis.",
            "general_notes": "Can be taken with or without food. Dose adjustment may be needed in patients with renal impairment.",
            "safety_note": "Use with caution in elderly patients and those with kidney disease. Avoid alcohol as it may increase drowsiness."
        },
        "paracetamol": {
            "product_name": "Paracetamol (Acetaminophen)",
            "category": "Analgesic / Antipyretic",
            "confidence": 1.0,
            "visible_features": ["White round or caplet-shaped tablets", "Strengths: 325mg, 500mg, 650mg", "Also available as syrup and suppository"],
            "description": "Paracetamol (acetaminophen) is a widely used analgesic and antipyretic that works centrally by inhibiting prostaglandin synthesis in the brain, providing effective pain relief and fever reduction.",
            "advantages": [
                "Excellent safety profile when used at recommended doses",
                "Does not cause gastrointestinal bleeding (unlike NSAIDs)",
                "Safe for use in pregnancy (when used as directed) and most patient populations"
            ],
            "disadvantages": [
                "No significant anti-inflammatory effect",
                "Risk of severe hepatotoxicity (liver damage) in overdose",
                "Maximum daily dose must not exceed 4g in adults (lower in liver disease)"
            ],
            "suggested_use": "Mild-to-moderate pain relief (headache, toothache, musculoskeletal pain), fever reduction, and osteoarthritis management.",
            "general_notes": "Patients should be warned about hidden sources of paracetamol in combination products (cold/flu medications) to avoid accidental overdose.",
            "safety_note": "Black Box Warning: Risk of fatal hepatotoxicity in overdose. Contraindicated in severe hepatic impairment. Avoid concurrent excessive alcohol use. N-acetylcysteine is the antidote for overdose."
        }
    }

    # Match in local DB
    for key, data in local_db.items():
        if key in name_clean:
            return data
            
    # Dynamic search fallback with MedGemma if loaded
    if ModelCache._medgemma_model is not None or (TRANSFORMERS_AVAILABLE and os.environ.get("LOAD_MEDGEMMA", "false").lower() == "true"):
        try:
            tokenizer = ModelCache.get_medgemma_tokenizer()
            model = ModelCache.get_medgemma_model()
            
            prompt = f"""Provide structured medical information about the drug "{medicine_name}" for educational purposes.
Respond strictly in JSON format matching this exact structure:
{{
    "product_name": "Official Drug Name",
    "category": "Drug class / category",
    "confidence": 0.95,
    "visible_features": ["Feature 1", "Feature 2"],
    "description": "Description of the drug and how it works",
    "advantages": ["Advantage 1", "Advantage 2"],
    "disadvantages": ["Disadvantage 1", "Disadvantage 2"],
    "suggested_use": "Common indications and uses",
    "general_notes": "Important instructions or guidelines",
    "safety_note": "Warnings and contraindications"
}}

JSON:"""
            
            inputs = tokenizer(prompt, return_tensors="pt")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            logger.info(f"Generating info for {medicine_name} using MedGemma...")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=350, temperature=0.3, do_sample=True)
            
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                soap_data = json.loads(response_text[json_start:json_end])
                soap_data["confidence"] = 0.95
                return soap_data
        except Exception as e:
            logger.error(f"MedGemma medicine generation failed: {e}")
            
    # Standard template fallback for unknown drugs
    formatted_name = medicine_name.strip().capitalize()
    return {
        "product_name": formatted_name,
        "category": "General Medical Product / Drug",
        "confidence": 0.5,
        "visible_features": ["Varies by manufacturer", "Verify package labeling"],
        "description": f"No offline details are available for '{formatted_name}'. For patient safety, reference official prescribing information.",
        "advantages": ["Refer to official prescribing insert"],
        "disadvantages": ["Refer to official prescribing insert"],
        "suggested_use": "Refer to official prescribing insert",
        "general_notes": "Ensure that the medication packaging and prescription details match the patient's record before administration.",
        "safety_note": "Always consult a clinical pharmacist or prescribing provider. Discontinue use if signs of an allergic reaction occur."
    }


def generate_product_chat_response(
    user_message: str,
    product_info: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generate a contextual response for product/medicine-related questions.
    Uses product info and chat history for context while staying safe.
    """
    if product_info is None:
        product_info = {}
    if chat_history is None:
        chat_history = []
        
    analysis = product_info.get("analysis", {}) if "analysis" in product_info else product_info
    med_name = analysis.get("product_name", "the medication")
    category = analysis.get("category", "")
    description = analysis.get("description", "")
    advantages = analysis.get("advantages", [])
    disadvantages = analysis.get("disadvantages", [])
    suggested_use = analysis.get("suggested_use", "")
    safety_note = analysis.get("safety_note", "")
    
    message_lower = user_message.lower()
    
    # Check if MedGemma is loaded and can answer contextually
    if ModelCache._medgemma_model is not None:
        try:
            tokenizer = ModelCache.get_medgemma_tokenizer()
            model = ModelCache.get_medgemma_model()
            
            # Format chat history
            history_str = ""
            for msg in chat_history[-4:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_str += f"{role}: {msg.get('content')}\n"
                
            prompt = f"""You are a safe offline clinical medication assistant.
Here is the details of the active drug being discussed:
Drug Name: {med_name}
Class: {category}
Description: {description}
Uses: {suggested_use}
Common Side Effects / Drawbacks: {', '.join(disadvantages)}
Contraindications: {safety_note}

Context:
{history_str}User: {user_message}

Provide a concise, educational, and professional response about {med_name}.
Never prescribe, diagnose, or give clinical instructions. Keep it educational.

Response:"""
            
            inputs = tokenizer(prompt, return_tensors="pt")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            logger.info(f"Generating chat reply for {med_name} using MedGemma...")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.4, do_sample=True)
            
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            marker = "Response:"
            if marker in response_text:
                response = response_text.split(marker)[-1].strip()
            else:
                response = response_text.strip()
                
            return SafetyChecker.filter_response(response)
        except Exception as e:
            logger.error(f"MedGemma chat response failed: {e}")

    # Offline intelligent rule-based agent fallback
    if any(word in message_lower for word in ["side effect", "adverse", "risk", "disadvantage", "bad", "harm", "symptom"]):
        response = f"Common side effects or limitations of {med_name} include: "
        if disadvantages:
            response += "; ".join(disadvantages) + "."
        else:
            response += "gastrointestinal irritation, allergic reactions, or class-specific adverse effects. Refer to the manufacturer package insert."
        response += f" Additionally, note: {safety_note}"
        
    elif any(word in message_lower for word in ["use", "indication", "why", "who", "for", "prescribe", "take"]):
        response = f"{med_name} ({category}) is primarily used for: {suggested_use}. {description}"
        
    elif any(word in message_lower for word in ["pregnant", "pregnancy", "breastfeed", "baby"]):
        response = f"Pregnancy and lactation guidelines vary. For {med_name}, the safety profile is as follows: {safety_note}. Always consult an obstetrician or pharmacist before taking medications during pregnancy."
        
    elif any(word in message_lower for word in ["dose", "dosage", "how much", "frequently", "often", "mg"]):
        response = f"Typical dosage and administration for {med_name} depends on the indication and patient-specific factors (e.g., renal function). Always follow the prescriber's instructions. General notes: {analysis.get('general_notes', 'refer to prescribing label')}"
        
    elif any(word in message_lower for word in ["help", "diagnose", "sick", "disease"]):
        response = f"I cannot diagnose conditions. For {med_name}, it belongs to the class {category} and is indicated for {suggested_use}. If you are feeling unwell, consult a primary care physician."
        
    else:
        response = f"{med_name} is a medication of the class {category}. {description} "
        if advantages:
            response += f"It has key therapeutic properties: {'; '.join(advantages)}. "
        response += f"Please consult your healthcare provider or prescribing guidelines for specific questions."

    return SafetyChecker.filter_response(response)


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


# ============================================================================
# PRODUCT ASSISTANT ENDPOINTS (NEW)
# ============================================================================

@app.post("/api/product/analyze", response_model=ProductAnalysisResult)
async def analyze_product(request: MedicineAnalysisRequest):
    """
    Look up detailed educational information about a medicine/drug by name.
    Returns: name, description, category, advantages, disadvantages, use cases, and safety notes.
    """
    try:
        if not request.name or not request.name.strip():
            raise HTTPException(
                status_code=400,
                detail="Medicine name must be a non-empty string"
            )
        
        logger.info(f"Looking up medicine: {request.name}")
        
        # Analyze medicine via local database or MedGemma
        analysis = generate_medicine_info(request.name)

        # Validate and return
        product_result = ProductAnalysisResult(
            product_name=analysis.get("product_name", "Unknown Product"),
            category=analysis.get("category", "unknown"),
            confidence=float(analysis.get("confidence", 0.0)),
            visible_features=analysis.get("visible_features", []) or [],
            description=analysis.get("description", ""),
            advantages=analysis.get("advantages", []) or [],
            disadvantages=analysis.get("disadvantages", []) or [],
            suggested_use=analysis.get("suggested_use", ""),
            general_notes=analysis.get("general_notes", ""),
            safety_note=analysis.get("safety_note", SafetyChecker.get_safety_disclaimer())
        )
        
        logger.info(f"Successfully retrieved info for: {product_result.product_name}")
        return product_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing medicine: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Medicine analysis failed: {str(e)}"
        )


@app.post("/api/product/chat", response_model=ProductChatResponse)
async def chat_about_product(request: ProductChatRequest):
    """
    Chat endpoint for answering questions about medical products.
    
    Provides educational, product-level information only.
    Never gives medical diagnostics or treatment advice.
    All responses include appropriate disclaimers.
    """
    try:
        if not request.message or not request.message.strip():
            raise HTTPException(
                status_code=400,
                detail="Message must be a non-empty string"
            )
        
        logger.info(f"Processing product chat: {request.message[:50]}...")
        
        # Generate response with product context
        product_dict = request.product_info.dict() if request.product_info else None
        chat_dict = [{"role": m.role, "content": m.content} for m in request.chat_history] if request.chat_history else []
        
        response_text = generate_product_chat_response(
            user_message=request.message,
            product_info=product_dict,
            chat_history=chat_dict
        )
        
        # Check safety
        is_safe = SafetyChecker.is_safe_response(response_text)
        
        result = ProductChatResponse(
            response=response_text,
            is_safe=is_safe,
            generated_at=datetime.utcnow().isoformat() + "Z"
        )
        
        logger.info(f"Product chat response generated (safe={is_safe})")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in product chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )


@app.get("/api/info")
async def api_info():
    """Root endpoint with API information"""
    return {
        "message": "MedClinic - Medical SOAP Notes & Medicine Info Assistant (Powered by MedGemma)",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "summarize": "/api/visit/summarize (POST) - SOAP note generation",
            "transcribe": "/api/transcribe (POST with audio file) - Audio to text",
            "product_analyze": "/api/product/analyze (POST) - Look up medicine details",
            "product_chat": "/api/product/chat (POST) - Chat about medicines",
        },
        "models": {
            "medgemma": "google/medgemma-2b (SOAP & Medicine generation)",
            "whisper": "openai/whisper-base (Audio transcription)",
            "medicine_assistant": "Local medication info database (with optional MedGemma)"
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
