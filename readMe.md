# Clinical Notes Digitizer - A streamlit App

## Key Features
1. OCR Text Extraction: Uses Tesseract OCR to extract text from uploaded images
2. Clinical NLP Processing: Extracts medical entities like medications, conditions, vital signs, and procedures
3. FHIR-like Structure: Converts processed data into a DocumentReference-like FHIR format
4. JSON Storage: Saves all processed notes as JSON files in a local folder
5. Interactive UI: Clean Streamlit interface with real-time processing

## Architecture Compoentns: 
- Frontend : Streamlit web interface 
- OCR: Tesseract (might need to replace with cloud OCR for better handwriting recognition)
- NLP : Locall LLM servered using vllm 

- Storage: Local Json files in `Clinical_notes_json` folder in FHIR format 
- output : FHIR Document Reference format 

## To Run the Demo: 
1. install Dependencies: 
```bash
pip install -r requirments.txt  
```

2. Install Tesseract OCR 
- Linux 
``` 
Sudo apt-get install tesseract-ocr
```
- windows : downlaod from github releases 

- macos: 
```
brew install tesseract
``` 

3. Start vLLM server (Run based on the hardaware)
``` bash
python -m vllm.entrypoints.openai.api_server \
--model deepseek-ai/deepseek-model-name \
--host 0.0.0.0 \
--port 8000
```
4. Run the Application 
```bash 
streamlit run app.py
```

## Hardware Requirements: 
- FOr Deep Seek 33B model - it will need 66GB GPU memory (A100 80GB or 2 x A6000) 
- Deepseek 7B model - will need 14GB GPU memoery (RTX 3090)
## Future Enhancement for Production 

Replace the demo compoent with: 

1. Better OCR: 
    - AWS Textract, google vision API, or Azure Form Recognizer 

2. Better LLMs : 
    - Llama 3, phimed or Qwen or bigger sized deepseek with medical fine-turning (if local)
    - chatGPT or gemini or claudeai 
3. Clinical NLP
    -  

4. FHIR validation: 

5. Database or FHIR store : 
Replace SON files with proper database stroage or FHIR store 