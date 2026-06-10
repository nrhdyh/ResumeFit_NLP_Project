import re
import string
from io import StringIO

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="ResumeFit - NLP Resume Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .card {
        padding: 22px;
        border-radius: 16px;
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.04);
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .small-text {
        color: #6b7280;
        font-size: 15px;
    }

    .skill-pill {
        display: inline-block;
        padding: 7px 12px;
        margin: 4px;
        border-radius: 30px;
        background-color: #ecfdf5;
        color: #047857;
        font-size: 14px;
        border: 1px solid #a7f3d0;
    }

    .missing-pill {
        display: inline-block;
        padding: 7px 12px;
        margin: 4px;
        border-radius: 30px;
        background-color: #fef2f2;
        color: #b91c1c;
        font-size: 14px;
        border: 1px solid #fecaca;
    }

    .info-pill {
        display: inline-block;
        padding: 7px 12px;
        margin: 4px;
        border-radius: 30px;
        background-color: #eff6ff;
        color: #1d4ed8;
        font-size: 14px;
        border: 1px solid #bfdbfe;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATASET LOADING
# ============================================================
@st.cache_data
def load_resume_dataset():
    try:
        df = pd.read_csv("Resume.csv")
        df = df[["Resume_str", "Category"]].dropna()
        df["Resume_str"] = df["Resume_str"].astype(str)
        df["Category"] = df["Category"].astype(str)
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["Resume_str", "Category"])


# ============================================================
# JOB DESCRIPTION TEMPLATES
# ============================================================
JOB_DESCRIPTIONS = {
    "INFORMATION-TECHNOLOGY": """
    We are looking for an information technology candidate with experience in software development,
    programming, database management, networking, cybersecurity, technical support, system administration,
    troubleshooting, cloud computing, web development, API integration, and IT project support.
    """,

    "HR": """
    We are looking for a human resource candidate with experience in recruitment, payroll,
    employee relations, training, onboarding, performance management, HR policies, staff records,
    communication, conflict resolution, and administrative support.
    """,

    "ACCOUNTANT": """
    We are looking for an accountant with experience in financial reporting, bookkeeping,
    auditing, taxation, accounts payable, accounts receivable, budgeting, payroll, bank reconciliation,
    accounting software, and financial analysis.
    """,

    "FINANCE": """
    We are looking for a finance candidate with experience in financial analysis, budgeting,
    forecasting, investment analysis, risk management, reporting, accounting, financial planning,
    Microsoft Excel, and business decision support.
    """,

    "ENGINEERING": """
    We are looking for an engineering candidate with experience in project engineering,
    technical design, process improvement, quality control, maintenance, manufacturing,
    problem solving, safety standards, documentation, and engineering analysis.
    """,

    "TEACHER": """
    We are looking for a teacher with experience in lesson planning, classroom management,
    student assessment, curriculum development, teaching, communication, learning activities,
    educational technology, and student support.
    """,

    "HEALTHCARE": """
    We are looking for a healthcare candidate with experience in patient care, medical records,
    clinical support, healthcare administration, treatment planning, safety procedures,
    communication, health assessment, and medical documentation.
    """,

    "SALES": """
    We are looking for a sales candidate with experience in customer relationship management,
    sales strategy, lead generation, negotiation, product presentation, communication,
    target achievement, account management, and business development.
    """,

    "DESIGNER": """
    We are looking for a designer with experience in graphic design, visual communication,
    Adobe Photoshop, Adobe Illustrator, branding, layout design, creative concepts,
    typography, digital media, and user interface design.
    """,

    "BUSINESS-DEVELOPMENT": """
    We are looking for a business development candidate with experience in market research,
    client relationship management, sales planning, partnership development, proposal writing,
    negotiation, strategic planning, communication, and revenue growth.
    """,

    "DIGITAL-MEDIA": """
    We are looking for a digital media candidate with experience in content creation,
    social media management, digital marketing, video editing, online campaigns,
    branding, analytics, communication, and creative media production.
    """,

    "FITNESS": """
    We are looking for a fitness candidate with experience in personal training,
    exercise planning, fitness assessment, health coaching, nutrition guidance,
    client motivation, safety, and wellness programs.
    """,

    "CHEF": """
    We are looking for a chef with experience in food preparation, kitchen operations,
    menu planning, food safety, cooking techniques, inventory control, hygiene standards,
    and team coordination.
    """,

    "ADVOCATE": """
    We are looking for an advocate or legal candidate with experience in legal research,
    case preparation, drafting legal documents, client consultation, litigation support,
    compliance, negotiation, and legal communication.
    """,

    "CONSULTANT": """
    We are looking for a consultant with experience in business analysis, client advisory,
    problem solving, project management, research, strategy development, process improvement,
    communication, and reporting.
    """,

    "PUBLIC-RELATIONS": """
    We are looking for a public relations candidate with experience in media relations,
    corporate communication, press releases, event coordination, branding, stakeholder communication,
    crisis communication, and content writing.
    """,

    "BANKING": """
    We are looking for a banking candidate with experience in customer service,
    financial products, loan processing, risk assessment, compliance, account management,
    banking operations, and financial documentation.
    """,

    "ARTS": """
    We are looking for an arts candidate with experience in creative production,
    visual arts, design concepts, artistic planning, exhibitions, creative communication,
    media production, and project coordination.
    """,

    "AVIATION": """
    We are looking for an aviation candidate with experience in airline operations,
    aircraft safety, customer service, flight operations, aviation regulations,
    scheduling, documentation, and safety compliance.
    """,

    "AGRICULTURE": """
    We are looking for an agriculture candidate with experience in crop production,
    farm management, agricultural operations, soil management, irrigation, livestock,
    sustainability, and agricultural technology.
    """,

    "APPAREL": """
    We are looking for an apparel candidate with experience in fashion design,
    garment production, textile knowledge, merchandising, quality control, retail,
    inventory management, and customer service.
    """,

    "CONSTRUCTION": """
    We are looking for a construction candidate with experience in site management,
    project coordination, safety procedures, building materials, construction planning,
    quality control, documentation, and team supervision.
    """,

    "AUTOMOBILE": """
    We are looking for an automobile candidate with experience in vehicle maintenance,
    automotive repair, diagnostics, mechanical systems, customer service, safety procedures,
    and technical documentation.
    """,

    "BPO": """
    We are looking for a BPO candidate with experience in customer support,
    call center operations, communication, data entry, client handling, problem solving,
    service quality, and process documentation.
    """
}


# ============================================================
# SKILL DICTIONARY
# ============================================================
SKILLS = [
    "python", "java", "sql", "mysql", "html", "css", "javascript", "react",
    "node.js", "php", "firebase", "api", "database", "web development",
    "software development", "machine learning", "deep learning", "nlp",
    "data analysis", "data visualization", "power bi", "excel", "tableau",
    "tensorflow", "scikit-learn", "streamlit", "flask", "django",

    "communication", "teamwork", "leadership", "problem solving",
    "project management", "time management", "critical thinking",
    "customer service", "training", "documentation", "reporting",

    "recruitment", "payroll", "employee relations", "onboarding",
    "performance management", "hr policies",

    "accounting", "bookkeeping", "auditing", "taxation", "budgeting",
    "financial analysis", "bank reconciliation", "accounts payable",
    "accounts receivable",

    "sales", "marketing", "negotiation", "lead generation",
    "business development", "market research", "client relationship",

    "graphic design", "photoshop", "illustrator", "branding",
    "typography", "layout design", "ui design",

    "teaching", "lesson planning", "classroom management",
    "student assessment", "curriculum development",

    "patient care", "medical records", "clinical support",
    "healthcare administration",

    "quality control", "maintenance", "manufacturing", "safety",
    "engineering analysis", "technical design"
]


# ============================================================
# NLP PREPROCESSING
# ============================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text.strip()

    except Exception:
        return ""


def extract_skills(text):
    cleaned = clean_text(text)
    found = []

    for skill in SKILLS:
        if skill.lower() in cleaned:
            found.append(skill)

    return sorted(list(set(found)))


# ============================================================
# SIMILARITY AND MATCHING
# ============================================================
def calculate_similarity(resume_text, job_text):
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_text)

    if not resume_clean or not job_clean:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([resume_clean, job_clean])

    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)


def get_match_level(score):
    if score >= 65:
        return "High Match"
    elif score >= 35:
        return "Medium Match"
    else:
        return "Low Match"


def predict_match(score):
    if score >= 35:
        return "Match"
    return "Not Match"


def get_recommendation(level, missing_skills):
    if level == "High Match":
        return "This resume is highly suitable for the selected job description."
    elif level == "Medium Match":
        if missing_skills:
            return "This resume is moderately suitable. The candidate should improve the missing skills listed below."
        return "This resume is moderately suitable, but the job description may require more specific experience."
    else:
        return "This resume has low similarity with the job description. The candidate may not be suitable for this role."


def display_pills(items, pill_class):
    if not items:
        st.write("None detected.")
    else:
        html = ""
        for item in items:
            html += f"<span class='{pill_class}'>{item}</span>"
        st.markdown(html, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("📄 ResumeFit")
st.sidebar.caption("NLP-Based Resume Screening System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Resume Matcher",
        "Dataset Explorer",
        "Evaluation",
        "About Project"
    ]
)


df = load_resume_dataset()


# ============================================================
# HOME PAGE
# ============================================================
if page == "Home":
    st.markdown("<h1 class='main-title'>📄 ResumeFit</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>An NLP-Based Resume Screening and Job Matching System</p>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Project Overview")
    st.write(
        """
        ResumeFit is a Natural Language Processing application that compares resume text 
        with a job description. The system calculates a matching score, identifies matched 
        and missing skills, and provides a suitability recommendation.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Rows", len(df))
    col2.metric("Resume Categories", df["Category"].nunique() if not df.empty else 0)
    col3.metric("NLP Method", "TF-IDF + Cosine")

    st.markdown("### Main Features")
    st.write(
        """
        - Upload real resume PDF
        - Paste resume text manually
        - Select or paste job description
        - Calculate resume-job matching score
        - Extract matched and missing skills
        - Evaluate system using accuracy, precision, recall, and F1-score
        """
    )

    st.markdown("### NLP Flow")
    st.code(
        """
Resume / Job Description
        ↓
Text Cleaning
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity
        ↓
Match Score
        ↓
Matched Skills + Missing Skills
        ↓
Recommendation
        """,
        language="text"
    )


# ============================================================
# RESUME MATCHER PAGE
# ============================================================
elif page == "Resume Matcher":
    st.markdown("<h1 class='main-title'>🔍 Resume Matcher</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Upload or paste a resume, then compare it with a job description.</p>",
        unsafe_allow_html=True
    )

    resume_input_method = st.radio(
        "Choose Resume Input Method",
        ["Upload Resume PDF", "Paste Resume Text", "Use Sample Resume from Dataset"],
        horizontal=True
    )

    resume_text = ""

    if resume_input_method == "Upload Resume PDF":
        uploaded_resume = st.file_uploader(
            "Upload Resume PDF",
            type=["pdf"]
        )

        if uploaded_resume is not None:
            resume_text = extract_text_from_pdf(uploaded_resume)

            if resume_text:
                st.success("Resume text extracted successfully.")
                with st.expander("View Extracted Resume Text"):
                    st.text_area("Extracted Resume Text", resume_text, height=250)
            else:
                st.error(
                    "Unable to extract text from this PDF. "
                    "Please use a text-based PDF or paste the resume text manually."
                )

    elif resume_input_method == "Paste Resume Text":
        resume_text = st.text_area(
            "Paste Resume Text",
            height=280,
            placeholder="Paste candidate resume text here..."
        )

    else:
        if df.empty:
            st.error("Resume.csv not found or empty.")
        else:
            selected_category = st.selectbox(
                "Select Resume Category",
                sorted(df["Category"].unique())
            )

            sample_df = df[df["Category"] == selected_category].reset_index(drop=True)

            sample_index = st.number_input(
                "Select Sample Resume Number",
                min_value=0,
                max_value=max(len(sample_df) - 1, 0),
                value=0,
                step=1
            )

            resume_text = sample_df.loc[sample_index, "Resume_str"]

            with st.expander("View Selected Resume Text"):
                st.text_area("Sample Resume Text", resume_text, height=250)

    st.markdown("---")

    job_input_method = st.radio(
        "Choose Job Description Input Method",
        ["Use Job Template", "Paste Job Description"],
        horizontal=True
    )

    job_text = ""

    if job_input_method == "Use Job Template":
        selected_job = st.selectbox(
            "Select Job Category",
            sorted(JOB_DESCRIPTIONS.keys())
        )

        job_text = JOB_DESCRIPTIONS[selected_job]

        with st.expander("View Job Description Template"):
            st.text_area("Job Description", job_text, height=220)

    else:
        job_text = st.text_area(
            "Paste Job Description",
            height=260,
            placeholder="Paste job description here..."
        )

    st.markdown("---")

    if st.button("Analyze Match", use_container_width=True):
        if not resume_text.strip() or not job_text.strip():
            st.warning("Please provide both resume text and job description.")
        else:
            score = calculate_similarity(resume_text, job_text)
            level = get_match_level(score)

            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_text)

            matched_skills = sorted(list(set(resume_skills) & set(job_skills)))
            missing_skills = sorted(list(set(job_skills) - set(resume_skills)))
            extra_skills = sorted(list(set(resume_skills) - set(job_skills)))

            recommendation = get_recommendation(level, missing_skills)

            st.markdown("## Result")

            col1, col2, col3 = st.columns(3)

            col1.metric("Match Score", f"{score}%")
            col2.metric("Suitability Level", level)
            col3.metric("Missing Skills", len(missing_skills))

            st.progress(score / 100)

            st.markdown("### Matched Skills")
            display_pills(matched_skills, "skill-pill")

            st.markdown("### Missing Skills")
            display_pills(missing_skills, "missing-pill")

            st.markdown("### Extra Skills Found in Resume")
            display_pills(extra_skills[:20], "info-pill")

            st.markdown("### Recommendation")
            st.info(recommendation)

            result_df = pd.DataFrame({
                "Item": [
                    "Match Score",
                    "Suitability Level",
                    "Matched Skills",
                    "Missing Skills",
                    "Recommendation"
                ],
                "Result": [
                    f"{score}%",
                    level,
                    ", ".join(matched_skills) if matched_skills else "None",
                    ", ".join(missing_skills) if missing_skills else "None",
                    recommendation
                ]
            })

            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Result as CSV",
                data=csv,
                file_name="resume_match_result.csv",
                mime="text/csv",
                use_container_width=True
            )


# ============================================================
# DATASET EXPLORER PAGE
# ============================================================
elif page == "Dataset Explorer":
    st.markdown("<h1 class='main-title'>📁 Dataset Explorer</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>View and understand the resume dataset used in this project.</p>",
        unsafe_allow_html=True
    )

    if df.empty:
        st.error("Resume.csv not found. Please place Resume.csv in the same folder as app.py.")
    else:
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Resumes", len(df))
        col2.metric("Total Categories", df["Category"].nunique())
        col3.metric("Dataset Column Used", "Resume_str")

        st.markdown("### Resume Categories")

        category_count = df["Category"].value_counts().reset_index()
        category_count.columns = ["Category", "Count"]

        st.dataframe(category_count, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(category_count["Category"], category_count["Count"])
        ax.set_xlabel("Category")
        ax.set_ylabel("Number of Resumes")
        ax.set_title("Number of Resumes by Category")
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.markdown("### Preview Dataset")

        selected_category = st.selectbox(
            "Filter by Category",
            ["All"] + sorted(df["Category"].unique())
        )

        if selected_category == "All":
            preview_df = df.copy()
        else:
            preview_df = df[df["Category"] == selected_category]

        st.dataframe(preview_df.head(20), use_container_width=True)

        with st.expander("View One Resume Example"):
            if not preview_df.empty:
                st.text_area(
                    "Resume Text",
                    preview_df.iloc[0]["Resume_str"],
                    height=300
                )


# ============================================================
# EVALUATION PAGE
# ============================================================
elif page == "Evaluation":
    st.markdown("<h1 class='main-title'>📊 Model Evaluation</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Evaluate the matching model using resume categories and job description templates.</p>",
        unsafe_allow_html=True
    )

    if df.empty:
        st.error("Resume.csv not found. Please place Resume.csv in the same folder as app.py.")
    else:
        st.write(
            """
            For evaluation, the system compares resumes with job description templates.
            If the resume category is the same as the job template category, the actual label is 
            considered **Match**. If it is different, the actual label is considered **Not Match**.
            """
        )

        available_categories = sorted(list(set(df["Category"].unique()) & set(JOB_DESCRIPTIONS.keys())))

        selected_eval_category = st.selectbox(
            "Select Job Category for Evaluation",
            available_categories,
            index=available_categories.index("INFORMATION-TECHNOLOGY") if "INFORMATION-TECHNOLOGY" in available_categories else 0
        )

        sample_size = st.slider(
            "Number of resumes to evaluate",
            min_value=20,
            max_value=300,
            value=100,
            step=20
        )

        threshold = st.slider(
            "Match Threshold (%)",
            min_value=10,
            max_value=80,
            value=35,
            step=5
        )

        if st.button("Run Evaluation", use_container_width=True):
            positive_df = df[df["Category"] == selected_eval_category].copy()
            negative_df = df[df["Category"] != selected_eval_category].copy()

            half_size = sample_size // 2

            positive_sample = positive_df.sample(
                n=min(half_size, len(positive_df)),
                random_state=42
            )

            negative_sample = negative_df.sample(
                n=min(sample_size - len(positive_sample), len(negative_df)),
                random_state=42
            )

            eval_df = pd.concat([positive_sample, negative_sample], ignore_index=True)

            job_text = JOB_DESCRIPTIONS[selected_eval_category]

            scores = []
            actual_labels = []
            predicted_labels = []

            for _, row in eval_df.iterrows():
                score = calculate_similarity(row["Resume_str"], job_text)
                scores.append(score)

                actual = "Match" if row["Category"] == selected_eval_category else "Not Match"
                predicted = "Match" if score >= threshold else "Not Match"

                actual_labels.append(actual)
                predicted_labels.append(predicted)

            eval_df["Similarity Score (%)"] = scores
            eval_df["Actual Label"] = actual_labels
            eval_df["Predicted Label"] = predicted_labels

            accuracy = accuracy_score(actual_labels, predicted_labels)
            precision = precision_score(
                actual_labels,
                predicted_labels,
                pos_label="Match",
                zero_division=0
            )
            recall = recall_score(
                actual_labels,
                predicted_labels,
                pos_label="Match",
                zero_division=0
            )
            f1 = f1_score(
                actual_labels,
                predicted_labels,
                pos_label="Match",
                zero_division=0
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
            col2.metric("Precision", f"{precision * 100:.2f}%")
            col3.metric("Recall", f"{recall * 100:.2f}%")
            col4.metric("F1-Score", f"{f1 * 100:.2f}%")

            st.markdown("### Evaluation Metrics Chart")

            metrics = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            }

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(metrics.keys(), metrics.values())
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score")
            ax.set_title("Evaluation Metrics")
            st.pyplot(fig)

            st.markdown("### Confusion Matrix")

            cm = confusion_matrix(
                actual_labels,
                predicted_labels,
                labels=["Match", "Not Match"]
            )

            cm_df = pd.DataFrame(
                cm,
                index=["Actual Match", "Actual Not Match"],
                columns=["Predicted Match", "Predicted Not Match"]
            )

            st.dataframe(cm_df, use_container_width=True)

            st.markdown("### Prediction Result Table")

            show_df = eval_df[
                [
                    "Category",
                    "Similarity Score (%)",
                    "Actual Label",
                    "Predicted Label",
                    "Resume_str"
                ]
            ]

            st.dataframe(show_df, use_container_width=True)

            csv = show_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Evaluation Result as CSV",
                data=csv,
                file_name="evaluation_result.csv",
                mime="text/csv",
                use_container_width=True
            )


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================
elif page == "About Project":
    st.markdown("<h1 class='main-title'>ℹ️ About Project</h1>", unsafe_allow_html=True)

    st.markdown("### Problem Statement")
    st.write(
        """
        Resume screening is often performed manually by recruiters. This process can be 
        time-consuming and inconsistent, especially when many resumes need to be compared 
        against a job description. Different candidates may use different wording to describe 
        similar skills and experience. Therefore, this project applies Natural Language Processing 
        techniques to support resume-job matching.
        """
    )

    st.markdown("### Objectives")
    st.write(
        """
        1. To develop an NLP-based system that compares resume text with job description text.
        2. To calculate a resume-job matching score using TF-IDF and cosine similarity.
        3. To identify matched skills and missing skills from resume and job description text.
        4. To evaluate the system using accuracy, precision, recall, and F1-score.
        """
    )

    st.markdown("### Dataset")
    st.write(
        """
        The dataset used in this project is `Resume.csv`, which contains resume text and job 
        category labels. The main column used for NLP processing is `Resume_str`, while 
        `Category` is used for evaluation.
        """
    )

    st.markdown("### NLP Techniques Used")
    st.write(
        """
        - Text cleaning
        - Stopword removal
        - TF-IDF vectorization
        - Cosine similarity
        - Keyword-based skill extraction
        """
    )

    st.markdown("### Tools and Libraries")
    st.write(
        """
        - Python
        - Streamlit
        - Pandas
        - Scikit-learn
        - Matplotlib
        - PyPDF
        """
    )

    st.markdown("### Limitations")
    st.write(
        """
        - The system depends on text extracted from resumes.
        - Scanned image-based PDF resumes may not be readable without OCR.
        - TF-IDF focuses on word importance but does not fully understand semantic meaning.
        - The skill extraction feature depends on a predefined skill dictionary.
        """
    )

    st.markdown("### Future Work")
    st.write(
        """
        - Add OCR support for scanned resumes.
        - Use BERT or Sentence Transformers for better semantic matching.
        - Add ranking for multiple candidates.
        - Improve the skill dictionary with more industry-specific skills.
        - Add PDF report generation for matching results.
        """
    )
