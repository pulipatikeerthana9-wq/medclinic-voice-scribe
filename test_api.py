#!/usr/bin/env python
"""
MedClinic Test Suite
Tests API endpoints and model integration
"""

import requests
import json
import time
import sys

# Ensure UTF-8 output on Windows to support emojis in console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

API_BASE = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("\n🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE}/health")
        assert response.status_code == 200
        print("✓ Health check passed")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test API info endpoint"""
    print("\n🔍 Testing API info endpoint...")
    try:
        response = requests.get(f"{API_BASE}/api/info")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        print(f"✓ API info endpoint passed (version: {data['version']})")
        return True
    except Exception as e:
        print(f"✗ API info endpoint failed: {e}")
        return False

def test_soap_generation():
    """Test SOAP note generation with sample transcript"""
    print("\n🔍 Testing SOAP generation...")
    
    sample_transcript = """
    Patient is a 45-year-old male presenting with 3 days of cough and low-grade fever. 
    Reports productive cough with clear sputum. Denies chest pain or shortness of breath. 
    Vital signs: BP 120/80, HR 88, RR 16, Temp 38.1C. 
    Lungs clear to auscultation bilaterally. 
    Assessment: Likely viral respiratory infection. 
    Plan: Supportive care, follow-up in 1 week if symptoms persist.
    """
    
    try:
        response = requests.post(
            f"{API_BASE}/api/visit/summarize",
            json={"transcript": sample_transcript},
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"✗ API returned status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
        
        data = response.json()
        
        # Validate response structure
        assert "soap_note" in data, "Missing soap_note in response"
        assert "subjective" in data["soap_note"], "Missing subjective"
        assert "objective" in data["soap_note"], "Missing objective"
        assert "assessment" in data["soap_note"], "Missing assessment"
        assert "plan" in data["soap_note"], "Missing plan"
        assert "actions" in data, "Missing actions"
        
        print("✓ SOAP generation passed")
        print(f"\n  Subjective: {data['soap_note']['subjective'][:100]}...")
        print(f"  Assessment: {data['soap_note']['assessment'][:100]}...")
        print(f"  Plan items: {len(data['soap_note']['plan'])}")
        print(f"  Actions: {len(data['actions'])}")
        
        return True
    except requests.exceptions.Timeout:
        print("✗ Request timed out (model might still be loading)")
        return False
    except Exception as e:
        print(f"✗ SOAP generation failed: {e}")
        return False

def test_empty_transcript():
    """Test validation: empty transcript should fail"""
    print("\n🔍 Testing input validation...")
    try:
        response = requests.post(
            f"{API_BASE}/api/visit/summarize",
            json={"transcript": ""},
        )
        
        assert response.status_code in (400, 422), f"Expected 400 or 422, got {response.status_code}"
        print("✓ Input validation passed")
        return True
    except Exception as e:
        print(f"✗ Input validation failed: {e}")
        return False

def test_medicine_analysis():
    """Test medicine analysis endpoint"""
    print("\n🔍 Testing medicine analysis...")
    try:
        response = requests.post(
            f"{API_BASE}/api/product/analyze",
            json={"name": "Amoxicillin"},
            timeout=10
        )
        if response.status_code != 200:
            print(f"✗ Medicine API returned status {response.status_code}: {response.text}")
            return False
            
        data = response.json()
        assert data["product_name"] == "Amoxicillin", f"Expected Amoxicillin, got {data['product_name']}"
        assert "category" in data
        assert "description" in data
        assert "safety_note" in data
        print("✓ Medicine analysis passed")
        return True
    except Exception as e:
        print(f"✗ Medicine analysis failed: {e}")
        return False

def test_medicine_chat():
    """Test medicine chat endpoint"""
    print("\n🔍 Testing medicine chat...")
    try:
        # Load mock product info
        mock_info = {
            "product_name": "Amoxicillin",
            "category": "Antibiotic",
            "confidence": 1.0,
            "visible_features": [],
            "description": "Amoxicillin is an antibiotic.",
            "advantages": [],
            "disadvantages": [],
            "suggested_use": "Bacterial infections",
            "general_notes": "Take with water",
            "safety_note": "Contraindicated in penicillin allergy"
        }
        response = requests.post(
            f"{API_BASE}/api/product/chat",
            json={
                "message": "What are the precautions for Amoxicillin?",
                "product_info": mock_info,
                "chat_history": []
            },
            timeout=10
        )
        if response.status_code != 200:
            print(f"✗ Medicine chat returned status {response.status_code}: {response.text}")
            return False
            
        data = response.json()
        assert "response" in data
        assert "is_safe" in data
        print("✓ Medicine chat passed")
        return True
    except Exception as e:
        print(f"✗ Medicine chat failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("MedClinic API Test Suite")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
    except:
        print("\n❌ ERROR: Cannot connect to MedClinic server")
        print(f"   Make sure the server is running at {API_BASE}")
        print("\n   Start the server with: python main.py")
        sys.exit(1)
    
    tests = [
        test_health_check,
        test_root_endpoint,
        test_empty_transcript,
        test_medicine_analysis,
        test_medicine_chat,
        test_soap_generation,  # This might take time on first run (model loading)
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All tests passed! MedClinic is ready to use.")
        print(f"\n📱 Open browser: http://localhost:8000")
        print(f"📚 API docs: http://localhost:8000/docs")
    else:
        print(f"❌ {total - passed} test(s) failed")
        sys.exit(1)
    
    print("=" * 60)

if __name__ == "__main__":
    main()
