from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from prompts import EXTRACTION_PROMPT,RESUME_EXTRACTION_PROMPT,ANALYZER_PROMPT,EVALUATOR_PROMPT
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel, Field
from agent_state import AgentState
from dotenv import load_dotenv
from typing import List
import logging
import pathlib
import fitz
import os
import re

_ = load_dotenv()

api_key = os.environ.get("NVIDIA_API_KEY")
if not api_key:
    raise ValueError("NVIDIA_API_KEY is missing! Please set it in the Hugging Face Settings secrets.")

class JDSchema(BaseModel):
    hard_skills: List[str] = Field(description="Technical skills like Python, React, etc.")
    soft_skills: List[str] = Field(description="Interpersonal skills like Mentorship or Agile.")
    min_years_experience: int = Field(description="The minimum years of experience required.")
    required_degree: str = Field(description="Minimum education like BS, MS, or PhD.")

class ResumeSchema(BaseModel):
    hard_skills: List[str] = Field(description="List of technical skills found.")
    total_years_exp: float = Field(description="Total years of work experience calculated from dates.")
    top_achievements: List[str] = Field(description="Quantifiable accomplishments.")
    education: str = Field(description="Highest degree and major.")
    seniority_level: str = Field(description="Inferred seniority based on titles.")

class FinalEvaluation(BaseModel):
    match_score: float = Field(description="Score from 0 to 100")
    suggestions: List[str] = Field(description="3-5 specific bullet point edits for the resume")

def clean_jd_text(raw_jd: str):
    if not raw_jd:
        return ""
    fluff_patterns = [r"About the company.*", r"Equal Opportunity Employer.*",r"Apply now.*",r"Share this job.*"]
    for pattern in fluff_patterns:
        raw_jd = re.sub(pattern, "", raw_jd, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'\s+', ' ', raw_jd).strip()
    return cleaned

def clean_extracted_text(text):
    # Remove excessive newlines
    text = re.sub(r'\n+', '\n', text)
    # Remove weird ligatures or non-ascii characters often found in PDFs
    text = text.encode("ascii", "ignore").decode()
    # Normalize spacing
    text = re.sub(r' +', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_file):
    try:
        file_path = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        cleaned_text = clean_extracted_text(text)
        return cleaned_text
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def extraction_node(state: AgentState):
    print(">>> ENTERING EXTRACTION NODE")
    llm_jd = ChatNVIDIA(model="openai/gpt-oss-120b",api_key=api_key, temperature=0,top_p=1,max_completion_tokens=16384).with_structured_output(JDSchema)
    llm_resume = ChatNVIDIA(model="openai/gpt-oss-120b",api_key=api_key, temperature=0,top_p=1,max_completion_tokens=16384).with_structured_output(ResumeSchema)

    # 2. Process JD
    jd_messages = [SystemMessage(content=EXTRACTION_PROMPT), HumanMessage(content=state['jd_raw'])]
    jd_skills = llm_jd.invoke(jd_messages)
    #print("Extracted JD Skills:", jd_skills)
    
    # 3. Process Resume
    resume_messages = [SystemMessage(content=RESUME_EXTRACTION_PROMPT), HumanMessage(content=state['resume_text'])]
    resume_skills = llm_resume.invoke(resume_messages)
    #print("Extracted Resume Skills:", resume_skills)

    print("LEAVING EXTRACTION NODE >>>")
    return {
        "extracted_jd": jd_skills, 
        "extracted_resume": resume_skills
    }

def analyzer_node(state: AgentState):
    print(">>> ENTERING ANALYZER NODE")
    llm = ChatNVIDIA(model="openai/gpt-oss-120b", api_key=api_key, temperature=0, top_p=1, max_completion_tokens=16384)

    formatted_prompt = ANALYZER_PROMPT.format(jd_extracted=state['extracted_jd'],resume_extracted=state['extracted_resume'])
    analysis_response = llm.invoke([HumanMessage(content=formatted_prompt)])
    
    #print("Analysis Result:", analysis_response.content)
    print("LEAVING ANALYZER NODE >>>")
    return {"gap_analysis": analysis_response.content}

def evaluator_node(state: AgentState):
    print(">>> ENTERING EVALUATOR NODE")
    llm = ChatNVIDIA(model="openai/gpt-oss-120b", api_key=api_key, temperature=0, top_p=1, max_completion_tokens=16384).with_structured_output(FinalEvaluation)
    
    evaluator_messages = [SystemMessage(content=EVALUATOR_PROMPT), HumanMessage(content=state['gap_analysis'])]
    result = llm.invoke(evaluator_messages)
    print("LEAVING EVALUATOR NODE >>>")
    return {
        "match_score": result.match_score,
        "suggestions": result.suggestions
    }