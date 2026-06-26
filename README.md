Based on the project details provided in the source material, here is a comprehensive **README.md** file for **ResumeFitTracker**. This document summarizes the system's purpose, features, methodology, and technical stack.

***

# ResumeFitTracker: NLP-Based Resume Screening and Job Matching System

**ResumeFitTracker** is a Natural Language Processing (NLP) application designed to automate the recruitment process by comparing resumes against job descriptions. The system calculates a similarity score, identifies skill gaps, and provides suitability recommendations to help recruiters and job seekers streamline the screening process.

**Live System:** [https://resume-fit-check.streamlit.app/](https://resume-fit-check.streamlit.app/)

## 🚀 Features
*   **Multiple Input Methods:** Upload resumes in PDF format, paste text manually, or select a sample resume from the built-in dataset.
*   **Comprehensive Scoring:** Generates a final matching score based on **Word TF-IDF**, **Character n-gram TF-IDF**, and **Skill Matching**.
*   **Skill Analysis:** Automatically extracts and categorizes skills into **Matched Skills**, **Missing Skills**, and **Extra Skills** found in the resume.
*   **Suitability Recommendation:** Provides a clear recommendation (e.g., "High Match") based on the final calculated score.
*   **Dataset Explorer:** Allows users to view and understand the distribution of the 2,483 resumes across 24 job categories used in the project.
*   **Evaluation Dashboard:** Displays performance metrics such as accuracy, precision, recall, and a confusion matrix.

## 🛠️ Technical Stack
*   **Language:** Python
*   **Web Framework:** Streamlit
*   **NLP Libraries:** NLTK (for tokenization, stopword removal, and lemmatization)
*   **Machine Learning:** Scikit-learn (for TF-IDF vectorization and Cosine Similarity)
*   **Data Handling:** Pandas
*   **Visualization:** Matplotlib
*   **PDF Processing:** PyPDF

## 📋 Methodology
The system follows a structured NLP pipeline to ensure accurate matching:
1.  **Text Preprocessing:** Converts text to lowercase and removes symbols, stopwords, and applies lemmatization to reduce noise.
2.  **TF-IDF Vectorization:** Converts cleaned text into numerical vectors using word-level and character n-gram analysis to handle spelling variations.
3.  **Cosine Similarity Calculation:** Measures the angle between the resume and job description vectors to determine text similarity.
4.  **Skill Extraction:** Uses a predefined dictionary to identify specific technical and soft skills.
5.  **Final Scoring:** Combines text similarity and skill matching into a single suitability percentage.

## 📊 Performance Summary
During evaluation (specifically within the Information Technology category), the system achieved the following results:
*   **Accuracy:** 73.00%
*   **Precision:** 100.00% (The system is highly reliable when it predicts a match)
*   **Recall:** 46.00%
*   **F1-Score:** 63.01%

## 🔮 Future Enhancements
*   Implementing **OCR support** for scanned PDF resumes.
*   Integrating semantic models like **BERT or Sentence Transformers** for deeper meaning-based matching.
*   Expanding the **skill dictionary** and adding support for **DOCX** files.
*   Automating **resume improvement suggestions** for candidates.

## 👥 Author
*   **Nur Hidayah Binti Ansari** (S23B0108)
*   **Faculty:** Faculty of Data Science & Computing, Universiti Malaysia Kelantan
*   **Lecturer:** Prof. Madya Dr. Nooraini Binti Yusoff
