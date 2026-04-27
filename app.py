import os
import re
import gradio as gr
from utils import extract_text_from_pdf, clean_jd_text
from nodes import create_analyzer_graph 

# Initialize graph
analyzer = create_analyzer_graph()

# Extract basic info
def extract_personal_info(text):
    name = text.split("\n")[0] if text else "Unknown"

    email = re.findall(r'\S+@\S+', text)
    phone = re.findall(r'\b\d{10}\b', text)

    return {
        "name": name,
        "email": email[0] if email else "Not found",
        "phone": phone[0] if phone else "Not found"
    }

# Main logic
def process_analysis(jd_raw, pdf_file):
    resume_text = extract_text_from_pdf(pdf_file) 
    jd_text = clean_jd_text(jd_raw)

    personal = extract_personal_info(resume_text)
    
    initial_state = {
        "jd_raw": jd_text,
        "resume_text": resume_text,
        "extracted_skills": {},
        "gap_analysis": "",
        "match_score": 0.0,
        "suggestions": []
    }
    
    final_output = analyzer.invoke(initial_state)
    
    score = final_output.get("match_score", 0)
    analysis = final_output.get("gap_analysis", "No analysis found.")
    suggestions = "\n\n".join(final_output.get("suggestions", []))

    return (
        score,
        analysis,
        suggestions,
        f"{personal['name']}",
        f"{personal['email']}",
        f"{personal['phone']}"
    )


with gr.Blocks(
    title="Resume Analyzer",
    css="""
    .gradio-container {
        max-width: 950px;
        margin: auto;
        padding-top: 20px;
        padding-bottom: 0px !important;
    }

    body {
        background: #f7fbff;
        font-family: Inter, sans-serif;
        color: #374151;
    }

    .gr-row {
        gap: 20px;
    }

    .gr-button {
        background: #a7c7e7 !important;
        color: #1f2937 !important;
        border-radius: 6px !important;
    }

    /* 🔒 strong blur */
    .blur-text {
        filter: blur(12px);
        user-select: none;
        pointer-events: none;
        font-size: 14px;
        color: #6b7280;
    }

    .details-box {
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 12px;
    margin-top: 10px;
    background: #1f2937 !important;
}

    footer {
        display: none !important;
    }
    """
) as demo:

    gr.Markdown("Resume Analyzer")

    with gr.Row():
        with gr.Column():

            jd_input = gr.Textbox(label="Job Description", lines=8)

            with gr.Group():

                file_input = gr.File(
                    label="Upload Resume (PDF)",
                    file_types=[".pdf"]
                )
            with gr.Column(visible=False, elem_classes="details-box") as blur_section:
                    gr.Markdown("Personal Details")

                    name_out = gr.Markdown(elem_classes="blur-text")
                    email_out = gr.Markdown(elem_classes="blur-text")
                    phone_out = gr.Markdown(elem_classes="blur-text")
            run_btn = gr.Button("🔍 Run Analysis", variant="primary")

            

        # 👉 RIGHT
        with gr.Column():

            score_out = gr.Number(label="Match Score %")

            with gr.Tabs():
                with gr.TabItem("Gap Analysis"):
                    analysis_out = gr.Markdown()

                with gr.TabItem("How to Improve"):
                    suggestions_out = gr.Markdown()


    # 👇 show after run
    def run_all(jd_raw, pdf_file):
        score, analysis, suggestions, name, email, phone = process_analysis(jd_raw, pdf_file)

        return (
            score,
            analysis,
            suggestions,
            name,
            email,
            phone,
            gr.update(visible=True)
        )

    run_btn.click(
        fn=run_all,
        inputs=[jd_input, file_input],
        outputs=[
            score_out,
            analysis_out,
            suggestions_out,
            name_out,
            email_out,
            phone_out,
            blur_section
        ]
    )

demo.launch(share=True)