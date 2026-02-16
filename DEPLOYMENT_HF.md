# 🚀 Deploy MedClinic to Hugging Face Spaces - Step by Step

## ✅ Prerequisites

- [ ] Hugging Face account (sign up at <https://huggingface.co> - **FREE**)
- [ ] Git installed on your computer
- [ ] All project files ready

**Total cost: $0.00** ✓

---

## 📋 Step-by-Step Deployment Guide

### Step 1: Sign Up for Hugging Face (FREE)

1. Go to <https://huggingface.co>
2. Click "Sign Up" (top right)
3. Create account with email or GitHub
4. Verify your email
5. Go to <https://huggingface.co/spaces>

**No credit card needed!**

---

### Step 2: Create a New Space

1. Click **"Create new Space"** button
2. Fill in details:
   - **Space name**: `medclinic-voice-scribe` (or your choice)
   - **License**: MIT
   - **Select SDK**: Choose **"Docker"** (important!)
   - **Space hardware**: CPU basic (FREE tier)
   - **Visibility**: Public (required for free tier)

3. Click **"Create Space"**

---

### Step 3: Prepare Files for Upload

You need to upload these files from your `medclinic` folder:

**Required Files:**

```
✓ Dockerfile          → Rename Dockerfile.hf to Dockerfile
✓ README.md           → Rename README_SPACE.md to README.md  
✓ main.py             → As is
✓ index.html          → As is
✓ requirements.txt    → As is
✓ examples_sample_transcript_1.txt → As is
✓ examples_sample_transcript_2.txt → As is
```

**Files to EXCLUDE** (don't upload):

```
✗ venv/               (too large)
✗ __pycache__/        (not needed)
✗ server.log          (logs)
✗ Dockerfile          (use Dockerfile.hf instead)
✗ docker-compose.yml  (not for HF)
✗ setup.bat/setup.sh  (not needed)
✗ test_api.py         (development only)
```

---

### Step 4: Upload Files to Your Space

**Option A: Web Upload (Easiest)**

1. In your Space page, click **"Files and versions"** tab
2. Click **"Add file"** → **"Upload files"**
3. Drag and drop the required files (see list above)
4. **IMPORTANT**:
   - Rename `Dockerfile.hf` → `Dockerfile`
   - Rename `README_SPACE.md` → `README.md`
5. Click **"Commit changes to main"**

**Option B: Git (Advanced)**

```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/medclinic-voice-scribe
cd medclinic-voice-scribe

# Copy files from your medclinic folder
cp ../medclinic/main.py .
cp ../medclinic/index.html .
cp ../medclinic/requirements.txt .
cp ../medclinic/Dockerfile.hf ./Dockerfile
cp ../medclinic/README_SPACE.md ./README.md
cp ../medclinic/examples_sample_transcript_*.txt .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

---

### Step 5: Wait for Build (5-10 minutes)

1. After uploading, HF Spaces will automatically start building
2. Watch the **"Building"** status in your Space
3. You'll see logs showing:
   - Docker image building
   - Dependencies installing
   - Models downloading (MedGemma + Whisper = ~5GB)

**First build takes 5-10 minutes due to model downloads**

1. When complete, status will change to **"Running"** ✓

---

### Step 6: Test Your Live Demo

1. Once status is "Running", your Space URL will be active:

   ```
   https://huggingface.co/spaces/YOUR_USERNAME/medclinic-voice-scribe
   ```

2. Test the app:
   - Click "Load Sample Transcript"
   - Click "Generate SOAP"
   - Verify SOAP note appears

3. Test audio upload:
   - Switch to "Audio" tab
   - Upload a WAV/MP3 file
   - Verify transcription + SOAP generation

---

### Step 7: Get Your Public URL

Your live demo URL is:

```
https://YOUR_USERNAME-medclinic-voice-scribe.hf.space
```

Or the full URL:

```
https://huggingface.co/spaces/YOUR_USERNAME/medclinic-voice-scribe
```

**Copy this URL - you'll need it for:**

- Competition submission writeup
- README.md links
- Video demo

---

## 🐛 Troubleshooting

### Build Failed?

**Error: "Out of memory"**

- Solution: Models are too large for free tier. Try reducing model size in main.py:

  ```python
  # In main.py, line 89:
  cls._whisper_model = whisper.load_model("tiny")  # Instead of "base"
  ```

**Error: "Port 7860 not responding"**

- Solution: Check Dockerfile has correct port (7860, not 8000)
- Verify: `CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]`

**Error: "Health check failed"**

- Solution: Increase health check timeout in Dockerfile
- Or remove health check temporarily

### App is Slow?

- **First load**: Takes 30-60s (models loading into memory)
- **Subsequent requests**: Should be faster
- **CPU tier**: Inference takes ~30s per request
- **To speed up**: Upgrade to GPU tier (still free with community account!)

### Models Not Downloading?

If MedGemma won't download:

```python
# In main.py, add offline mode fallback:
# Line 183-184:
if not TRANSFORMERS_AVAILABLE:
    logger.warning("Using rule-based fallback")
    return generate_soap_fallback(transcript)
```

---

## 🎉 Success Checklist

- [ ] Space created on Hugging Face
- [ ] All files uploaded correctly
- [ ] Build completed successfully
- [ ] Status shows "Running"
- [ ] Live demo URL works
- [ ] Sample transcript generates SOAP note
- [ ] Audio upload works
- [ ] Copy public URL for competition submission

---

## 🔗 Next Steps After Deployment

1. **Update your README.md** with live demo link
2. **Add URL to WRITEUP.md** (competition submission)
3. **Create video demo** showing the live URL
4. **Share on social media** (LinkedIn, Twitter) for exposure
5. **Test from different devices** (mobile, tablet)

---

## 💡 Pro Tips

### Upgrade to GPU (FREE!)

1. In your Space settings
2. Change hardware to "CPU upgrade" or "T4 small" (if available)
3. Inference will be 5x faster (30s → 5s)

### Add Analytics

Add visitor counter to README:

```markdown
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=YOUR_USERNAME.medclinic-voice-scribe)
```

### Enable Gradio Interface (Optional)

If you want a fancier UI, consider adding Gradio wrapper (still free!)

---

## 📞 Need Help?

1. Check HF Spaces documentation: <https://huggingface.co/docs/hub/spaces>
2. HF Discord: <https://discord.gg/hugging-face>
3. Or respond here and I'll help debug!

---

**🎊 Congratulations! Your MedClinic is now LIVE and FREE forever!**

Your demo link: `https://YOUR_USERNAME-medclinic-voice-scribe.hf.space`

**Add this to your competition submission and you're one step closer to winning!** 🏆
