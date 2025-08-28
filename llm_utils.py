# filepath: c:\Users\divya\OneDrive\Desktop\testle\llm_utils.py
import subprocess
import json
import re

def call_gemma_llm(raw_text: str):
    """
    Send OCR text to Gemma LLM for structured JSON output with all required fields.
    """
    prompt = f"""
You are a marksheet parser. Extract ALL student information from the following OCR text.
Required fields to extract:

- Student Name
- Father's Name
- Mother's Name (if present)
- Roll Number
- Registration Number
- Course / Class
- College Name
- University
- Board
- All subjects with:
    - Subject Name
    - Marks Obtained
    - Maximum Marks
    - Confidence per subject
- Total Marks
- Percentage
- Division
- Date of Issue
- Overall OCR Confidence

Return the output in this exact JSON format ONLY (no extra text):
{{
  "student_name": "",
  "father_name": "",
  "mother_name": "",
  "roll_number": "",
  "registration_number": "",
  "course": "",
  "college_name": "",
  "university": "",
  "board": "",
  "subjects": [
    {{"name": "", "marks_obtained": "", "max_marks": "", "confidence": ""}}
  ],
  "total_marks": "",
  "percentage": "",
  "division": "",
  "date_of_issue": "",
  "confidence_overall": ""
}}

Rules:
- Use OCR confidence values if available.
- Leave fields empty if missing; do NOT guess.
- Only JSON output. Must be valid JSON.
- Ensure subject marks include obtained, min, max, and confidence.
- Student Name: should only contain the name, nothing else
- Roll Number: only numbers
- Subjects: extract as rows from table, do not guess

OCR Text:
{raw_text}
"""
    # Call Gemma API / LLM client here
    # Example (replace with actual Gemma call):
    # response = gemma_client.generate(prompt)
    # return response.text

    return prompt  # for testing, returns the prompt string


    process = subprocess.run(
        ["ollama", "run", "gemma:2b"],
        input=prompt,
        text=True,
        capture_output=True
    )

    response = process.stdout.strip()
    # Extract JSON from response using regex
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            json_data = json.loads(json_str)
        except json.JSONDecodeError:
            json_data = {"error": "Invalid JSON from LLM", "raw_output": response}
    else:
        json_data = {"error": "No JSON found in LLM output", "raw_output": response}

    return json_data