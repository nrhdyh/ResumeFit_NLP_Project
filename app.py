import re
import string
import os

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pypdf import PdfReader

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ============================================================
# NLTK SETUP
# ============================================================
@st.cache_resource
def setup_nltk():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

    return True


setup_nltk()


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
        color: #111827;
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

    .skill-pill {
        display: inline-block;
        padding: 7px 12px;
        margin: 5px;
        border-radius: 30px;
        background-color: #ecfdf5;
        color: #047857;
        font-size: 14px;
        border: 1px solid #a7f3d0;
    }

    .missing-pill {
        display: inline-block;
        padding: 7px 12px;
        margin: 5px;
        border-radius: 30px;
        background-color: #fef2f2;
        color: #b91c1c;
        font-size: 14px;
        border: 1px solid #fecaca;
    }

    .info-pill {
        display: inline-block;
        padding: 7px 12px;
        margin: 5px;
        border-radius: 30px;
        background-color: #eff6ff;
        color: #1d4ed8;
        font-size: 14px;
        border: 1px solid #bfdbfe;
    }

    .score-box {
        padding: 18px;
        border-radius: 14px;
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# JOB DESCRIPTION TEMPLATES
# ============================================================
JOB_DESCRIPTIONS = {
    "INFORMATION-TECHNOLOGY": """
    We are looking for an information technology candidate with experience in programming,
    software development, web development, database management, SQL, Python, JavaScript,
    technical support, networking, cybersecurity, troubleshooting, cloud computing,
    system administration, API integration, and IT project support.
    """,

    "HR": """
    We are looking for a human resource candidate with experience in recruitment, payroll,
    employee relations, onboarding, staff records, HR policies, performance management,
    training, administration, communication, and conflict resolution.
    """,

    "ACCOUNTANT": """
    We are looking for an accountant with experience in accounting, bookkeeping,
    financial reporting, auditing, taxation, accounts payable, accounts receivable,
    budgeting, payroll, bank reconciliation, accounting software, and financial analysis.
    """,

    "FINANCE": """
    We are looking for a finance candidate with experience in financial analysis,
    budgeting, forecasting, investment analysis, risk management, reporting,
    financial planning, accounting, Microsoft Excel, and business decision support.
    """,

    "ENGINEERING": """
    We are looking for an engineering candidate with experience in project engineering,
    technical design, process improvement, maintenance, manufacturing, quality control,
    safety standards, documentation, engineering analysis, mechanical systems,
    and problem solving.
    """,

    "TEACHER": """
    We are looking for a teacher with experience in teaching, lesson planning,
    classroom management, student assessment, curriculum development, learning activities,
    educational technology, communication, and student support.
    """,

    "HEALTHCARE": """
    We are looking for a healthcare candidate with experience in patient care,
    medical records, clinical support, healthcare administration, treatment planning,
    safety procedures, health assessment, communication, and medical documentation.
    """,

    "SALES": """
    We are looking for a sales candidate with experience in sales strategy,
    customer relationship management, lead generation, negotiation, product presentation,
    communication, account management, target achievement, and business development.
    """,

    "DESIGNER": """
    We are looking for a designer with experience in graphic design, visual communication,
    Adobe Photoshop, Adobe Illustrator, branding, layout design, creative concepts,
    typography, digital media, and user interface design.
    """,

    "BUSINESS-DEVELOPMENT": """
    We are looking for a business development candidate with experience in market research,
    client relationship management, partnership development, proposal writing,
    negotiation, strategic planning, communication, sales planning, and revenue growth.
    """,

    "DIGITAL-MEDIA": """
    We are looking for a digital media candidate with experience in content creation,
    social media management, digital marketing, video editing, online campaigns,
    branding, analytics, communication, and creative media production.
    """,

    "FITNESS": """
    We are looking for a fitness candidate with experience in personal training,
    exercise planning, fitness assessment, health coaching, nutrition guidance,
    safety, motivation, and wellness programs.
    """,

    "CHEF": """
    We are looking for a chef with experience in food preparation, kitchen operations,
    menu planning, food safety, cooking techniques, inventory control, hygiene standards,
    and team coordination.
    """,

    "ADVOCATE": """
    We are looking for a legal candidate with experience in legal research,
    case preparation, drafting legal documents, client consultation, litigation support,
    compliance, negotiation, and legal communication.
    """,

    "CONSULTANT": """
    We are looking for a consultant with experience in business analysis, client advisory,
    research, strategy development, project management, process improvement,
    communication, reporting, and problem solving.
    """,

    "PUBLIC-RELATIONS": """
    We are looking for a public relations candidate with experience in media relations,
    corporate communication, press releases, event coordination, branding,
    stakeholder communication, crisis communication, and content writing.
    """,

    "BANKING": """
    We are looking for a banking candidate with experience in banking operations,
    customer service, financial products, loan processing, risk assessment,
    compliance, account management, and financial documentation.
    """,

    "ARTS": """
    We are looking for an arts candidate with experience in creative production,
    visual arts, artistic planning, exhibitions, media production, creative communication,
    and project coordination.
    """,

    "AVIATION": """
    We are looking for an aviation candidate with experience in airline operations,
    aircraft safety, customer service, flight operations, aviation regulations,
    scheduling, documentation, and safety compliance.
    """,

    "AGRICULTURE": """
    We are looking for an agriculture candidate with experience in crop production,
    farm management, agricultural operations, soil management, irrigation,
    livestock, sustainability, and agricultural technology.
    """,

    "APPAREL": """
    We are looking for an apparel candidate with experience in fashion design,
    garment production, textile knowledge, merchandising, quality control,
    retail, inventory management, and customer service.
    """,

    "CONSTRUCTION": """
    We are looking for a construction candidate with experience in site management,
    project coordination, safety procedures, building materials, construction planning,
    quality control, documentation, and team supervision.
    """,

    "AUTOMOBILE": """
    We are looking for an automobile candidate with experience in vehicle maintenance,
    automotive repair, diagnostics, mechanical systems, customer service,
    safety procedures, and technical documentation.
    """,

    "BPO": """
    We are looking for a BPO candidate with experience in customer support,
    call center operations, communication, data entry, client handling,
    service quality, problem solving, and process documentation.
    """
}


# ============================================================
# SKILL DICTIONARY
# ============================================================
SKILLS = [
    # IT / Software
    "python", "java", "sql", "mysql", "html", "css", "javascript",
    "react", "node", "nodejs", "php", "firebase", "api", "database",
    "web development", "software development", "programming",
    "system administration", "technical support", "networking",
    "cybersecurity", "troubleshooting", "cloud computing",
    "machine learning", "deep learning", "nlp", "data analysis",
    "data visualization", "power bi", "excel", "tableau",
    "tensorflow", "scikit learn", "streamlit", "flask", "django",

    # Soft skills
    "communication", "teamwork", "leadership", "problem solving",
    "project management", "time management", "critical thinking",
    "customer service", "training", "documentation", "reporting",
    "presentation", "planning", "coordination", "management",

    # HR
    "recruitment", "payroll", "employee relations", "onboarding",
    "performance management", "hr policies", "staff records",
    "human resource", "administration",

    # Accounting / Finance
    "accounting", "bookkeeping", "auditing", "taxation", "budgeting",
    "financial analysis", "bank reconciliation", "accounts payable",
    "accounts receivable", "financial reporting", "finance",

    # Business / Sales / Marketing
    "sales", "marketing", "negotiation", "lead generation",
    "business development", "market research", "client relationship",
    "social media", "digital marketing", "content creation",
    "branding", "customer relationship",

    # Design / Media
    "graphic design", "photoshop", "illustrator", "typography",
    "layout design", "ui design", "video editing", "creative",

    # Education
    "teaching", "lesson planning", "classroom management",
    "student assessment", "curriculum development", "education",

    # Healthcare
    "patient care", "medical records", "clinical support",
    "healthcare administration", "health assessment",

    # Engineering / Manufacturing
    "quality control", "maintenance", "manufacturing", "safety",
    "engineering analysis", "technical design", "process improvement",
    "production", "mechanical", "electrical", "construction",

    # Chef / Fitness / Others
    "food safety", "cooking", "menu planning", "kitchen operations",
    "personal training", "fitness assessment", "health coaching",
    "legal research", "compliance", "litigation", "banking operations",
    "loan processing", "risk assessment"
]


# ============================================================
# LOAD DATASET
# ============================================================
@st.cache_data
def load_resume_dataset():
    possible_files = ["Resume_sample.csv", "Resume_clean.csv", "Resume.csv"]

    for file in possible_files:
        if os.path.exists(file):
            df = pd.read_csv(file)

            if "Resume_str" not in df.columns or "Category" not in df.columns:
                st.error("Dataset must contain Resume_str and Category columns.")
                return pd.DataFrame(columns=["Resume_str", "Category"])

            df = df[["Resume_str", "Category"]].dropna()
            df["Resume_str"] = df["Resume_str"].astype(str)
            df["Category"] = df["Category"].astype(str).str.upper().str.strip()

            return df

    return pd.DataFrame(columns=["Resume_str", "Category"])


# ============================================================
# NLP PREPROCESSING
# ============================================================
def get_stop_words():
    try:
        return set(stopwords.words("english"))
    except:
        return {
            "the", "and", "is", "in", "to", "of", "for", "a", "an",
            "with", "on", "at", "by", "from", "this", "that", "are"
        }


STOP_WORDS = get_stop_words()
LEMMATIZER = WordNetLemmatizer()


def normalize_special_terms(text):
    text = str(text).lower()

    replacements = {
        "node.js": "nodejs",
        "node js": "nodejs",
        "scikit-learn": "scikit learn",
        "powerbi": "power bi",
        "ms excel": "excel",
        "microsoft excel": "excel",
        "js": "javascript",
        "hr": "human resource",
        "ui/ux": "ui ux",
        "front-end": "frontend",
        "back-end": "backend",
        "front end": "frontend",
        "back end": "backend",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def preprocess_text(text):
    """
    NLP preprocessing:
    1. Lowercasing
    2. Special term normalization
    3. Remove URLs/emails/numbers/punctuation
    4. Tokenization using regex
    5. Stopword removal
    6. Lemmatization using NLTK WordNetLemmatizer
    """

    text = normalize_special_terms(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text)

    processed_tokens = []
    for token in tokens:
        if token not in STOP_WORDS:
            lemma = LEMMATIZER.lemmatize(token)
            processed_tokens.append(lemma)

    return " ".join(processed_tokens)


def normalize_text_for_skills(text):
    text = normalize_special_terms(text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text):
    cleaned = normalize_text_for_skills(text)
    found = []

    for skill in SKILLS:
        skill_clean = normalize_text_for_skills(skill)
        pattern = r"\b" + re.escape(skill_clean) + r"\b"

        if re.search(pattern, cleaned):
            found.append(skill)

    return sorted(list(set(found)))


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


# ============================================================
# VECTOR MODEL
# ============================================================
@st.cache_resource
def build_vectorizers(corpus):
    processed_corpus = [preprocess_text(text) for text in corpus]

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=8000,
        min_df=1
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=5000,
        min_df=1
    )

    word_vectorizer.fit(processed_corpus)
    char_vectorizer.fit(processed_corpus)

    return word_vectorizer, char_vectorizer


def calculate_word_tfidf_score(resume_text, job_text, word_vectorizer):
    resume_processed = preprocess_text(resume_text)
    job_processed = preprocess_text(job_text)

    vectors = word_vectorizer.transform([resume_processed, job_processed])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    return round(score * 100, 2)


def calculate_char_tfidf_score(resume_text, job_text, char_vectorizer):
    resume_processed = preprocess_text(resume_text)
    job_processed = preprocess_text(job_text)

    vectors = char_vectorizer.transform([resume_processed, job_processed])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    return round(score * 100, 2)


def calculate_skill_match_score(resume_text, job_text):
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))

    if not job_skills:
        return 0.0

    matched_skills = resume_skills & job_skills
    score = len(matched_skills) / len(job_skills)

    return round(score * 100, 2)


def calculate_final_score(resume_text, job_text, word_vectorizer, char_vectorizer):
    word_score = calculate_word_tfidf_score(resume_text, job_text, word_vectorizer)
    char_score = calculate_char_tfidf_score(resume_text, job_text, char_vectorizer)
    skill_score = calculate_skill_match_score(resume_text, job_text)

    final_score = (0.50 * word_score) + (0.25 * char_score) + (0.25 * skill_score)

    return {
        "final_score": round(final_score, 2),
        "word_score": word_score,
        "char_score": char_score,
        "skill_score": skill_score
    }


def get_match_level(score):
    if score >= 45:
        return "High Match"
    elif score >= 25:
        return "Medium Match"
    else:
        return "Low Match"


def predict_match(score, threshold=25):
    if score >= threshold:
        return "Match"
    return "Not Match"


def get_recommendation(level, missing_skills):
    if level == "High Match":
        return "This resume is highly suitable for the selected job description."
    elif level == "Medium Match":
        if missing_skills:
            return "This resume is moderately suitable. The candidate should improve the missing skills listed below."
        return "This resume is moderately suitable, but the job may require more specific experience."
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
# INITIALIZE DATA AND MODELS
# ============================================================
df = load_resume_dataset()

if not df.empty:
    model_corpus = list(df["Resume_str"].head(1000)) + list(JOB_DESCRIPTIONS.values())
else:
    model_corpus = list(JOB_DESCRIPTIONS.values())

word_vectorizer, char_vectorizer = build_vectorizers(model_corpus)


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


# ============================================================
# HOME PAGE
# ============================================================
if page == "Home":
    st.markdown("<h1 class='main-title'>📄 ResumeFit</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>An NLP-Based Resume Screening and Job Matching System</p>",
        unsafe_allow_html=True
    )

    st.markdown("### Project Overview")
    st.write(
        """
        ResumeFit is a Natural Language Processing application that compares resume text 
        with a job description. The system calculates a matching score, identifies matched 
        and missing skills, and provides a suitability recommendation.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Rows", len(df))
    col2.metric("Resume Categories", df["Category"].nunique() if not df.empty else 0)
    col3.metric("NLP Libraries", "NLTK + Scikit-learn")

    st.markdown("### Main Features")
    st.write(
        """
        - Upload real resume PDF
        - Paste resume text manually
        - Use sample resume from dataset
        - Select or paste job description
        - Text preprocessing using NLTK
        - TF-IDF word similarity
        - Character n-gram similarity
        - Skill matching score
        - Evaluation using accuracy, precision, recall, and F1-score
        """
    )

    st.markdown("### NLP Flow")
    st.code(
        """
Resume Text + Job Description
        ↓
Text Cleaning
        ↓
Tokenization
        ↓
Stopword Removal
        ↓
Lemmatization using NLTK
        ↓
Word TF-IDF Similarity
        ↓
Character N-Gram TF-IDF Similarity
        ↓
Skill Extraction and Skill Match Score
        ↓
Final Matching Score
        ↓
Suitability Level + Recommendation
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
            st.error("Resume_sample.csv not found or empty.")
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
            scores = calculate_final_score(
                resume_text,
                job_text,
                word_vectorizer,
                char_vectorizer
            )

            final_score = scores["final_score"]
            word_score = scores["word_score"]
            char_score = scores["char_score"]
            skill_score = scores["skill_score"]

            level = get_match_level(final_score)

            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_text)

            matched_skills = sorted(list(set(resume_skills) & set(job_skills)))
            missing_skills = sorted(list(set(job_skills) - set(resume_skills)))
            extra_skills = sorted(list(set(resume_skills) - set(job_skills)))

            recommendation = get_recommendation(level, missing_skills)

            st.markdown("## Result")

            col1, col2, col3 = st.columns(3)
            col1.metric("Final Match Score", f"{final_score}%")
            col2.metric("Suitability Level", level)
            col3.metric("Missing Skills", len(missing_skills))

            st.progress(final_score / 100)

            st.markdown("### Score Breakdown")
            score_col1, score_col2, score_col3 = st.columns(3)
            score_col1.metric("Word TF-IDF Score", f"{word_score}%")
            score_col2.metric("Character TF-IDF Score", f"{char_score}%")
            score_col3.metric("Skill Match Score", f"{skill_score}%")

            st.markdown("### Matched Skills")
            display_pills(matched_skills, "skill-pill")

            st.markdown("### Missing Skills")
            display_pills(missing_skills, "missing-pill")

            st.markdown("### Extra Skills Found in Resume")
            display_pills(extra_skills[:25], "info-pill")

            st.markdown("### Recommendation")
            st.info(recommendation)

            result_df = pd.DataFrame({
                "Item": [
                    "Final Match Score",
                    "Word TF-IDF Score",
                    "Character TF-IDF Score",
                    "Skill Match Score",
                    "Suitability Level",
                    "Matched Skills",
                    "Missing Skills",
                    "Recommendation"
                ],
                "Result": [
                    f"{final_score}%",
                    f"{word_score}%",
                    f"{char_score}%",
                    f"{skill_score}%",
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
        st.error("Resume_sample.csv not found. Please place Resume_sample.csv in the same folder as app.py.")
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
        st.error("Resume_sample.csv not found. Please place Resume_sample.csv in the same folder as app.py.")
    else:
        st.write(
            """
            The evaluation compares resumes with selected job description templates.
            If the resume category is the same as the selected job category, the actual label is **Match**.
            If it is different, the actual label is **Not Match**.
            """
        )

        available_categories = sorted(list(set(df["Category"].unique()) & set(JOB_DESCRIPTIONS.keys())))

        selected_eval_category = st.selectbox(
            "Select Job Category for Evaluation",
            available_categories,
            index=available_categories.index("INFORMATION-TECHNOLOGY")
            if "INFORMATION-TECHNOLOGY" in available_categories else 0
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
            value=25,
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
            word_scores = []
            char_scores = []
            skill_scores = []

            for _, row in eval_df.iterrows():
                score_result = calculate_final_score(
                    row["Resume_str"],
                    job_text,
                    word_vectorizer,
                    char_vectorizer
                )

                final_score = score_result["final_score"]

                scores.append(final_score)
                word_scores.append(score_result["word_score"])
                char_scores.append(score_result["char_score"])
                skill_scores.append(score_result["skill_score"])

                actual = "Match" if row["Category"] == selected_eval_category else "Not Match"
                predicted = predict_match(final_score, threshold)

                actual_labels.append(actual)
                predicted_labels.append(predicted)

            eval_df["Final Score (%)"] = scores
            eval_df["Word TF-IDF Score (%)"] = word_scores
            eval_df["Character TF-IDF Score (%)"] = char_scores
            eval_df["Skill Match Score (%)"] = skill_scores
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
                    "Final Score (%)",
                    "Word TF-IDF Score (%)",
                    "Character TF-IDF Score (%)",
                    "Skill Match Score (%)",
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
        against job descriptions. Different candidates may use different wording to describe 
        similar skills and experience. Therefore, this project applies Natural Language Processing 
        techniques to support resume-job matching.
        """
    )

    st.markdown("### Objectives")
    st.write(
        """
        1. To develop an NLP-based system that compares resume text with job description text.
        2. To preprocess resume and job description text using NLP techniques.
        3. To calculate resume-job matching scores using TF-IDF and cosine similarity.
        4. To identify matched and missing skills from resume and job description text.
        5. To evaluate the system using accuracy, precision, recall, and F1-score.
        """
    )

    st.markdown("### Dataset")
    st.write(
        """
        The dataset used in this project is `Resume_sample.csv`, which contains resume text 
        and job category labels. The main text column used for NLP processing is `Resume_str`, 
        while the `Category` column is used for evaluation.
        """
    )

    st.markdown("### NLP Libraries and Techniques Used")
    st.write(
        """
        - **NLTK**: Stopword removal and lemmatization
        - **Scikit-learn**: TF-IDF vectorization and cosine similarity
        - **Word TF-IDF**: Measures similarity based on important words and phrases
        - **Character n-gram TF-IDF**: Handles small spelling or wording differences
        - **Keyword-based skill extraction**: Detects matched and missing job-related skills
        """
    )

    st.markdown("### Scoring Formula")
    st.code(
        """
Final Score =
(0.50 × Word TF-IDF Score)
+ (0.25 × Character TF-IDF Score)
+ (0.25 × Skill Match Score)
        """,
        language="text"
    )

    st.markdown("### Tools and Libraries")
    st.write(
        """
        - Python
        - Streamlit
        - Pandas
        - NLTK
        - Scikit-learn
        - Matplotlib
        - PyPDF
        """
    )

    st.markdown("### Limitations")
    st.write(
        """
        - The system depends on the quality of text extracted from resumes.
        - Scanned image-based PDF resumes may not be readable without OCR.
        - TF-IDF cannot fully understand deep semantic meaning like transformer-based models.
        - The skill extraction feature depends on a predefined skill dictionary.
        """
    )

    st.markdown("### Future Work")
    st.write(
        """
        - Add OCR support for scanned resumes.
        - Use BERT or Sentence Transformers for stronger semantic matching.
        - Add ranking for multiple candidates.
        - Improve the skill dictionary with more industry-specific skills.
        - Add automatic resume improvement suggestions.
        """
    )
