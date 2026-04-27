from typing import Annotated, TypedDict, List

class AgentState(TypedDict):
    jd_raw: str
    resume_text: str
    extracted_jd: dict
    extracted_resume: dict 
    gap_analysis: str
    match_score: float
    suggestions: List[str]