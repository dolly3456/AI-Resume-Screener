import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Page configuration for professional look
st.set_page_config(page_title="Enterprise AI Resume Screener", page_icon="🎯", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .metric-box {background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);}
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("🎯 Enterprise AI Resume Screener & Parser")
st.markdown("##### *Advanced HR Analytics Platform for Instant Candidate Shortlisting*")
st.write("---")

# Layout splitting into two columns for better look
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 🛠️ Input Requirements")
    job_description = st.text_area("📋 Paste Job Description (JD):", height=220, placeholder="Enter requirements (e.g., Python, Machine Learning, SQL, Communication...)")
    uploaded_file = st.file_uploader("📂 Upload Candidate Resume (PDF format):", type=["pdf"])

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text_content = page.extract_text()
        if text_content:
            text += text_content + " "
    return text

# Function to extract keywords dynamically for analysis
def analyze_keywords(resume, jd):
    # Basic skill keyword extraction logic using regex tokenization
    jd_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', jd.lower()))
    resume_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', resume.lower()))
    
    # Filtering common short words (stop words quick alternative)
    common_filter = {'and', 'the', 'for', 'with', 'from', 'that', 'this', 'candidate', 'experience', 'required', 'working'}
    jd_keywords = {w for w in jd_words if len(w) > 2 and w not in common_filter}
    
    matched = jd_keywords.intersection(resume_words)
    missing = jd_keywords.difference(resume_words)
    
    return list(matched)[:10], list(missing)[:10]

# Analytics & Logic execution
with col2:
    st.markdown("### 📊 Live ATS Analytics")
    
    if st.button("🚀 Execute AI Screening", type="primary", use_container_width=True):
        if not job_description:
            st.error("⚠️ Please provide a Job Description to match against.")
        elif not uploaded_file:
            st.error("⚠️ Please upload a resume PDF file.")
        else:
            with st.spinner("AI Engine parsing text and running vector matching..."):
                # 1. Parse PDF
                resume_text = extract_text_from_pdf(uploaded_file)
                
                if not resume_text.strip():
                    st.error("❌ Unable to extract text. Please ensure the PDF is not an image scan.")
                else:
                    # 2. Vectorization & Similarity
                    text_corpus = [resume_text, job_description]
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(text_corpus)
                    
                    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                    match_percentage = round(similarity_matrix[0][0] * 100, 2)
                    
                    # 3. Keyword Match Analysis
                    matched_skills, missing_skills = analyze_keywords(resume_text, job_description)
                    
                    # --- Presentation Grid ---
                    st.toast("Analysis complete!", icon='🎉')
                    
                    # Metric dashboard
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.metric(label="ATS Score", value=f"{match_percentage}%")
                    with m_col2:
                        status = "Shortlist" if match_percentage >= 70 else ("Review" if match_percentage >= 40 else "Reject")
                        st.metric(label="HR Recommendation", value=status)
                    
                    st.progress(int(match_percentage) if match_percentage <= 100 else 100)
                    
                    # Advanced Insights section
                    st.markdown("#### 🔍 Keyword Gap Analysis")
                    
                    tabs1, tabs2 = st.tabs(["✅ Matched Keywords", "⚠️ Missing Core Keywords"])
                    with tabs1:
                        if matched_skills:
                            st.write(", ".join([f"`{skill.upper()}`" for skill in matched_skills]))
                        else:
                            st.caption("No significant keywords matched.")
                            
                    with tabs2:
                        if missing_skills:
                            st.write(", ".join([f"`{skill.upper()}`" for skill in missing_skills]))
                        else:
                            st.caption("Excellent! No major keywords missing.")
                    
                    # Dynamic Feedback Report for Download
                    report_text = f"--- AI RESUME SCREENING REPORT ---\n\nMatch Score: {match_percentage}%\nRecommendation: {status}\n\nMatched Skills: {', '.join(matched_skills)}\nMissing Skills: {', '.join(missing_skills)}"
                    
                    st.download_button(
                        label="📥 Download HR Evaluation Report",
                        data=report_text,
                        file_name="Resume_Screening_Report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
                    if match_percentage >= 70:
                        st.balloons()