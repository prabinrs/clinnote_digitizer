import streamlit as st
import pytesseract
from PIL import Image
import json
import os
from datetime import datetime
import uuid
import re
import asyncio
import requests
from typing import Dict, List, Optional

# Configure page
st.set_page_config(
    page_title="Clinical Notes Digitizer with DeepSeek",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'processed_notes' not in st.session_state:
    st.session_state.processed_notes = []

class DeepSeekClinicalProcessor:
    def __init__(self, vllm_server_url="http://localhost:8000"):
        self.output_folder = "clinical_notes_json"
        self.vllm_server_url = vllm_server_url
        self.ensure_output_folder()
        
    def ensure_output_folder(self):
        """Create output folder if it doesn't exist"""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def extract_text_from_image(self, image):
        """Extract text from uploaded image using OCR"""
        try:
            # Configure pytesseract for better medical text recognition
            custom_config = r'--oem 3 --psm 6'
            extracted_text = pytesseract.image_to_string(image, config=custom_config)
            return extracted_text.strip()
        except Exception as e:
            st.error(f"Error in OCR: {str(e)}")
            return None
    
    def check_vllm_server(self):
        """Check if vLLM server is running"""
        try:
            response = requests.get(f"{self.vllm_server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def call_deepseek_model(self, prompt: str, max_tokens: int = 2000) -> Optional[str]:
        """Call DeepSeek model via vLLM server"""
        try:
            payload = {
                "model": "deepseek-ai/deepseek-coder-33b-instruct",  # Adjust model name as needed
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
                "stop": ["</json>", "<|end|>"]
            }
            
            response = requests.post(
                f"{self.vllm_server_url}/v1/completions",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['text'].strip()
            else:
                st.error(f"vLLM API error: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"Error calling DeepSeek model: {str(e)}")
            return None
    
    def create_clinical_analysis_prompt(self, raw_text: str) -> str:
        """Create a comprehensive prompt for clinical text analysis"""
        prompt = f"""You are a medical AI assistant specialized in analyzing clinical notes. Please analyze the following clinical text and extract structured information in JSON format.

Clinical Text:
{raw_text}

Please extract and structure the following information in valid JSON format:

```json
{{
  "patient_info": {{
    "patient_id": "extracted or generated ID",
    "age": "if mentioned",
    "gender": "if mentioned",
    "date_of_visit": "if mentioned"
  }},
  "chief_complaint": "main reason for visit",
  "history_of_present_illness": "detailed description",
  "medications": [
    {{
      "name": "medication name",
      "dosage": "if mentioned",
      "frequency": "if mentioned",
      "route": "if mentioned"
    }}
  ],
  "allergies": ["list of allergies if mentioned"],
  "vital_signs": {{
    "blood_pressure": "if mentioned",
    "heart_rate": "if mentioned",
    "temperature": "if mentioned",
    "respiratory_rate": "if mentioned",
    "oxygen_saturation": "if mentioned",
    "weight": "if mentioned",
    "height": "if mentioned"
  }},
  "physical_examination": {{
    "general": "general appearance",
    "systems": {{
      "cardiovascular": "findings",
      "respiratory": "findings",
      "neurological": "findings",
      "gastrointestinal": "findings",
      "other": "other findings"
    }}
  }},
  "diagnoses": [
    {{
      "primary": "primary diagnosis",
      "icd10_code": "if known",
      "confidence": "high/medium/low"
    }}
  ],
  "procedures": [
    {{
      "name": "procedure name",
      "date": "if mentioned",
      "provider": "if mentioned"
    }}
  ],
  "laboratory_results": [
    {{
      "test_name": "name of test",
      "value": "result value",
      "reference_range": "if mentioned",
      "abnormal": "true/false"
    }}
  ],
  "plan": {{
    "medications_prescribed": ["list of new medications"],
    "follow_up": "follow-up instructions",
    "referrals": ["specialist referrals if any"],
    "lifestyle_recommendations": ["recommendations"]
  }},
  "provider_info": {{
    "provider_name": "if mentioned",
    "specialty": "if mentioned",
    "facility": "if mentioned"
  }}
}}
```

Important guidelines:
1. Only include information that is explicitly mentioned or can be reasonably inferred
2. Use "not mentioned" or null for missing information
3. Be accurate with medical terminology
4. Maintain patient confidentiality principles
5. Provide valid JSON format only

<json>"""

        return prompt
    
    def process_with_deepseek(self, raw_text: str) -> Dict:
        """Process clinical text using DeepSeek model"""
        if not self.check_vllm_server():
            st.error("❌ vLLM server is not running. Please start the server first.")
            return self.fallback_processing(raw_text)
        
        prompt = self.create_clinical_analysis_prompt(raw_text)
        
        with st.spinner("🧠 Processing with DeepSeek model..."):
            response = self.call_deepseek_model(prompt)
        
        if response:
            try:
                # Extract JSON from response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    clinical_data = json.loads(json_str)
                    
                    # Convert to FHIR-like structure
                    fhir_document = self.convert_to_fhir_structure(clinical_data, raw_text)
                    return fhir_document
                else:
                    st.error("Failed to extract valid JSON from model response")
                    return self.fallback_processing(raw_text)
                    
            except json.JSONDecodeError as e:
                st.error(f"JSON parsing error: {str(e)}")
                st.error(f"Raw response: {response[:500]}...")
                return self.fallback_processing(raw_text)
        else:
            return self.fallback_processing(raw_text)
    
    def convert_to_fhir_structure(self, clinical_data: Dict, raw_text: str) -> Dict:
        """Convert clinical data to FHIR DocumentReference structure"""
        document_id = str(uuid.uuid4())
        
        fhir_document = {
            "resourceType": "Bundle",
            "id": document_id,
            "type": "document",
            "timestamp": datetime.now().isoformat(),
            "entry": [
                {
                    "resource": {
                        "resourceType": "DocumentReference",
                        "id": document_id,
                        "status": "current",
                        "type": {
                            "coding": [{
                                "system": "http://loinc.org",
                                "code": "11506-3",
                                "display": "Progress note"
                            }]
                        },
                        "subject": {
                            "reference": f"Patient/{clinical_data.get('patient_info', {}).get('patient_id', 'unknown')}"
                        },
                        "date": datetime.now().isoformat(),
                        "content": [{
                            "attachment": {
                                "contentType": "text/plain",
                                "data": raw_text
                            }
                        }]
                    }
                }
            ],
            "clinical_analysis": clinical_data,
            "raw_text": raw_text,
            "processing_metadata": {
                "processed_by": "DeepSeek via vLLM",
                "processing_timestamp": datetime.now().isoformat(),
                "model": "deepseek-ai/deepseek-coder-33b-instruct"
            }
        }
        
        # Add patient resource if patient info is available
        patient_info = clinical_data.get('patient_info', {})
        if patient_info.get('patient_id'):
            patient_resource = {
                "resource": {
                    "resourceType": "Patient",
                    "id": patient_info.get('patient_id'),
                    "gender": patient_info.get('gender', 'unknown').lower() if patient_info.get('gender') else 'unknown'
                }
            }
            
            if patient_info.get('age'):
                try:
                    birth_year = datetime.now().year - int(patient_info.get('age'))
                    patient_resource["resource"]["birthDate"] = f"{birth_year}-01-01"
                except:
                    pass
            
            fhir_document["entry"].append(patient_resource)
        
        return fhir_document
    
    def fallback_processing(self, raw_text: str) -> Dict:
        """Fallback processing when vLLM is not available"""
        document_id = str(uuid.uuid4())
        
        return {
            "resourceType": "Bundle",
            "id": document_id,
            "type": "document",
            "timestamp": datetime.now().isoformat(),
            "entry": [{
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": document_id,
                    "status": "current",
                    "type": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": "11506-3",
                            "display": "Progress note"
                        }]
                    },
                    "subject": {
                        "reference": "Patient/unknown"
                    },
                    "date": datetime.now().isoformat(),
                    "content": [{
                        "attachment": {
                            "contentType": "text/plain",
                            "data": raw_text
                        }
                    }]
                }
            }],
            "raw_text": raw_text,
            "processing_metadata": {
                "processed_by": "Fallback processor",
                "processing_timestamp": datetime.now().isoformat(),
                "note": "vLLM server was not available"
            }
        }
    
    def save_to_json(self, processed_data: Dict) -> Optional[str]:
        """Save processed data to JSON file"""
        filename = f"clinical_note_{processed_data['id']}.json"
        filepath = os.path.join(self.output_folder, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            return filepath
        except Exception as e:
            st.error(f"Error saving JSON: {str(e)}")
            return None

# Initialize processor
@st.cache_resource
def load_processor():
    return DeepSeekClinicalProcessor()

processor = load_processor()

# Main Streamlit UI
st.title("🏥 Clinical Notes Digitizer with DeepSeek")
st.markdown("**Advanced Solution**: Using vLLM + DeepSeek for clinical text understanding")

# Server status check
server_status = processor.check_vllm_server()
if server_status:
    st.success("✅ vLLM Server: Connected")
else:
    st.warning("⚠️ vLLM Server: Not connected (will use fallback processing)")

# Sidebar for configuration
st.sidebar.title("Configuration")

st.sidebar.markdown("### vLLM Server")
server_url = st.sidebar.text_input("Server URL", value="http://localhost:8000")
processor.vllm_server_url = server_url

st.sidebar.markdown("### OCR Settings")
ocr_language = st.sidebar.selectbox("OCR Language", ["eng", "eng+fra", "eng+spa"])

st.sidebar.markdown("### Processing Options")
max_tokens = st.sidebar.slider("Max Tokens", 500, 4000, 2000)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, 0.1)

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Clinical Note")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload a handwritten clinical note image"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Clinical Note", use_column_width=True)
        
        # Process button
        if st.button("🔍 Process Clinical Note with DeepSeek", type="primary"):
            with st.spinner("Processing clinical note..."):
                # Extract text using OCR
                extracted_text = processor.extract_text_from_image(image)
                
                if extracted_text:
                    st.success("✅ OCR extraction completed!")
                    
                    # Display extracted text
                    with st.expander("📄 View Extracted Text", expanded=False):
                        st.text_area("Raw OCR Output", extracted_text, height=200, disabled=True)
                    
                    # Process with DeepSeek
                    processed_data = processor.process_with_deepseek(extracted_text)
                    
                    # Save to JSON
                    saved_path = processor.save_to_json(processed_data)
                    
                    if saved_path:
                        st.success(f"✅ Processed data saved to: `{saved_path}`")
                        st.session_state.processed_notes.append(processed_data)
                    
                else:
                    st.error("❌ Failed to extract text from image")

with col2:
    st.header("DeepSeek Analysis Results")
    
    if st.session_state.processed_notes:
        latest_note = st.session_state.processed_notes[-1]
        
        # Display clinical analysis if available
        if 'clinical_analysis' in latest_note:
            clinical_data = latest_note['clinical_analysis']
            
            # Patient Information
            if clinical_data.get('patient_info'):
                st.subheader("👤 Patient Information")
                patient = clinical_data['patient_info']
                col_a, col_b = st.columns(2)
                with col_a:
                    if patient.get('patient_id'):
                        st.markdown(f"**ID:** {patient['patient_id']}")
                    if patient.get('age'):
                        st.markdown(f"**Age:** {patient['age']}")
                with col_b:
                    if patient.get('gender'):
                        st.markdown(f"**Gender:** {patient['gender']}")
                    if patient.get('date_of_visit'):
                        st.markdown(f"**Visit Date:** {patient['date_of_visit']}")
            
            # Chief Complaint
            if clinical_data.get('chief_complaint'):
                st.subheader("🎯 Chief Complaint")
                st.markdown(clinical_data['chief_complaint'])
            
            # Vital Signs
            if clinical_data.get('vital_signs'):
                st.subheader("📊 Vital Signs")
                vitals = clinical_data['vital_signs']
                cols = st.columns(3)
                
                vital_items = [(k, v) for k, v in vitals.items() if v and v != "not mentioned"]
                for i, (key, value) in enumerate(vital_items):
                    with cols[i % 3]:
                        st.metric(key.replace('_', ' ').title(), value)
            
            # Medications
            if clinical_data.get('medications'):
                st.subheader("💊 Medications")
                for med in clinical_data['medications']:
                    if med.get('name') and med['name'] != "not mentioned":
                        med_info = med['name']
                        if med.get('dosage'):
                            med_info += f" - {med['dosage']}"
                        if med.get('frequency'):
                            med_info += f" ({med['frequency']})"
                        st.markdown(f"• {med_info}")
            
            # Diagnoses
            if clinical_data.get('diagnoses'):
                st.subheader("🔬 Diagnoses")
                for diag in clinical_data['diagnoses']:
                    if diag.get('primary') and diag['primary'] != "not mentioned":
                        diag_text = diag['primary']
                        if diag.get('confidence'):
                            confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(diag['confidence'], "⚪")
                            diag_text += f" {confidence_color}"
                        st.markdown(f"• {diag_text}")
            
            # Laboratory Results
            if clinical_data.get('laboratory_results'):
                st.subheader("🧪 Laboratory Results")
                for lab in clinical_data['laboratory_results']:
                    if lab.get('test_name') and lab['test_name'] != "not mentioned":
                        lab_info = f"**{lab['test_name']}:** {lab.get('value', 'N/A')}"
                        if lab.get('abnormal') == "true":
                            lab_info += " ⚠️"
                        st.markdown(lab_info)
            
            # Plan
            if clinical_data.get('plan'):
                st.subheader("📋 Treatment Plan")
                plan = clinical_data['plan']
                
                if plan.get('medications_prescribed'):
                    st.markdown("**New Medications:**")
                    for med in plan['medications_prescribed']:
                        if med != "not mentioned":
                            st.markdown(f"• {med}")
                
                if plan.get('follow_up') and plan['follow_up'] != "not mentioned":
                    st.markdown(f"**Follow-up:** {plan['follow_up']}")
        
        # JSON download
        st.subheader("📥 Export Data")
        json_str = json.dumps(latest_note, indent=2)
        st.download_button(
            label="Download FHIR Bundle JSON",
            data=json_str,
            file_name=f"clinical_bundle_{latest_note['id']}.json",
            mime="application/json"
        )
        
        # Display full JSON
        with st.expander("🔍 View Full FHIR Bundle"):
            st.json(latest_note)
    
    else:
        st.info("👆 Upload and process a clinical note to see DeepSeek analysis")

# Footer section
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Notes Processed", len(st.session_state.processed_notes))

with col2:
    if os.path.exists(processor.output_folder):
        json_files = len([f for f in os.listdir(processor.output_folder) if f.endswith('.json')])
        st.metric("JSON Files Saved", json_files)

with col3:
    status_color = "🟢" if server_status else "🔴"
    st.metric("vLLM Status", f"{status_color}")

with col4:
    if st.button("🗑️ Clear All Data"):
        st.session_state.processed_notes = []
        st.success("All data cleared!")

# Setup Instructions
with st.expander("🚀 vLLM Setup Instructions"):
    st.markdown("""
    ## Setting up vLLM with DeepSeek
    
    ### 1. Install vLLM
    ```bash
    pip install vllm
    ```
    
    ### 2. Start vLLM Server with DeepSeek
    ```bash
    # Option 1: DeepSeek Coder (recommended for clinical notes)
    python -m vllm.entrypoints.openai.api_server \
        --model deepseek-ai/deepseek-coder-33b-instruct \
        --tensor-parallel-size 1 \
        --host 0.0.0.0 \
        --port 8000
    
    # Option 2: DeepSeek Chat (alternative)
    python -m vllm.entrypoints.openai.api_server \
        --model deepseek-ai/deepseek-llm-67b-chat \
        --tensor-parallel-size 2 \
        --host 0.0.0.0 \
        --port 8000
    ```
    
    ### 3. Hardware Requirements
    - **DeepSeek-Coder-33B**: ~66GB GPU memory (A100 80GB or 2x A6000)
    - **DeepSeek-LLM-67B**: ~134GB GPU memory (2x A100 80GB)
    
    ### 4. Alternative: Smaller Models
    If you have limited GPU memory, try:
    ```bash
    # DeepSeek Coder 6.7B (fits on 16GB GPU)
    python -m vllm.entrypoints.openai.api_server \
        --model deepseek-ai/deepseek-coder-6.7b-instruct \
        --host 0.0.0.0 \
        --port 8000
    ```
    
    ### 5. CPU-Only Option (Slow)
    ```bash
    pip install vllm[cpu]
    # Then use smaller models with --enforce-eager flag
    ```
    """)

st.markdown("**Powered by vLLM + DeepSeek** - Advanced clinical text understanding with local models")