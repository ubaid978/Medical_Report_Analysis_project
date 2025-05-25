from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import pdfplumber
import base64
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.output_parsers import StrOutputParser
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_to_text(image_path: str) -> str:
    image_base64 = encode_image(image_path)
    chat = ChatOpenAI(model="gpt-4o-mini")
    response = chat.invoke([
        HumanMessage(content=[
            {"type": "text", "text": "Extract all readable text from this image."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            }
        ])
    ])
    return response.content

def pdfextract_text(pdf_path: str) -> str:
    all_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                all_text += page.extract_text() or ''
                all_text += '\n'
    except Exception as e:
        return f"Error reading PDF: {e}"
    return all_text

def extract_csv(csv_path: str) -> str:
    try:
        df = pd.read_csv(csv_path)
        return df.to_string()
    except Exception as e:
        return f"Error reading CSV: {e}"

def extract_excel(excel_path: str) -> str:
    try:
        df = pd.read_excel(excel_path)
        return df.to_string()
    except Exception as e:
        return f"Error reading Excel: {e}"

# Serve index.html at root route
@app.get("/")
def read_index():
    return FileResponse("static/new.html")

# LangChain template
medical_report_analysis_template = """
You are an expert medical analyst. Given the following medical report, analyze it in detail based on the sections below:

Medical Report:
{medical_report}

---

1. Patient Information: Extract patient details and medical history.
2. Reason for Examination: Summarize the patient's chief complaint or symptoms.
3. Clinical Findings: List key clinical or physical examination findings.
4. Diagnostic Tests: Enumerate and describe tests performed with results.
5. Results and Interpretation: Highlight normal and abnormal results; explain medical terms.
6. Diagnosis: State the diagnosis or possible differential diagnoses.
7. Treatment Recommendations: Summarize suggested treatments or next steps.
8. Prognosis and Follow-Up: Describe prognosis and any follow-up advice.
9. Additional Notes: Note any disclaimers or uncertainties.
10. Summary and Key Takeaways: Provide a concise summary for the patient.

Please provide a detailed, clear, and organized analysis based on the above sections.
"""

prompt = PromptTemplate(
    input_variables=["medical_report"],
    template=medical_report_analysis_template
)
llm = ChatOpenAI(model='gpt-3.5-turbo')
chain = prompt | llm | StrOutputParser()

# Utility functions: encode_image, image_to_text, etc.
# (Keep your previously defined helper functions here...)

@app.post("/extract")
async def extract_content(file: UploadFile = File(...)):
    filename = file.filename
    extension = os.path.splitext(filename)[-1].lower()
    temp_path = f"temp_{filename}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        if extension in [".jpg", ".jpeg", ".png"]:
            extracted_text = image_to_text(temp_path)
        elif extension == ".pdf":
            extracted_text = pdfextract_text(temp_path)
        elif extension == ".csv":
            extracted_text = extract_csv(temp_path)
        elif extension in [".xls", ".xlsx"]:
            extracted_text = extract_excel(temp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        analyzed_report = chain.invoke({'medical_report': extracted_text})
        return PlainTextResponse(analyzed_report)

    finally:
        os.remove(temp_path)


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
