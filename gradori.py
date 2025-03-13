import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import subprocess
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset only when needed
@st.cache_resource
def load_student_data():
    return pd.read_excel("student_performance_dataset_modified.xlsx")

df = load_student_data()

# Load MCQ dataset
mcq_df = pd.read_excel("mcq_machine_learning.xlsx")

# Compute total questions per topic
@st.cache_resource
def compute_questions_per_topic():
    return mcq_df.groupby('Topic')['Correct Answer'].count().to_dict()

total_questions_per_topic = compute_questions_per_topic()

# Load BERT tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer, model = load_model()

st.session_state.setdefault("register_number", None)

def grade_mcq(student_answers):
    correct_answers = mcq_df['Correct Answer'].dropna().values
    topics = mcq_df['Topic'].dropna().values
    topic_performance = {topic: 0 for topic in topics}

    if len(student_answers) != len(correct_answers):
        raise ValueError("Mismatch in number of responses.")

    correct_count = 0
    for student_ans, correct_ans, topic in zip(student_answers, correct_answers, topics):
        if student_ans.strip().lower() == correct_ans.strip().lower():
            correct_count += 1
            topic_performance[topic] += 1
    
    score = (correct_count / len(correct_answers)) * 100
    return round(score, 2), topic_performance

def analyze_topics(topic_performance):
    strong_topics, weak_topics = [], []
    for topic, correct_count in topic_performance.items():
        accuracy = (correct_count / total_questions_per_topic.get(topic, 1)) * 100
        if accuracy > 75:
            strong_topics.append(topic)
        elif accuracy < 50:
            weak_topics.append(topic)
    return strong_topics, weak_topics

def save_scores(mcq_score=None, subjective_score=None, coding_score=None):
    global df
    register_number = st.session_state.get("register_number")
    if register_number and register_number in df["Register Number"].values:
        student_index = df[df["Register Number"] == register_number].index[0]
        if mcq_score is not None:
            df.at[student_index, "MCQ Score"] = mcq_score
        if subjective_score is not None:
            df.at[student_index, "Subjective Score"] = subjective_score
        if coding_score is not None:
            df.at[student_index, "Coding Score"] = coding_score
        df.to_excel("student_performance_dataset_modified.xlsx", index=False)
        st.success("✅ Scores saved successfully!")

# Subjective answer grading with separate tokenization
def evaluate_subjective_answers(responses):
    scores = []
    for student_answer, reference_answer in responses.values():
        if student_answer.strip():
            inputs_student = tokenizer(student_answer, return_tensors="pt", padding=True, truncation=True)
            inputs_reference = tokenizer(reference_answer, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                output_student = model(**inputs_student).last_hidden_state.mean(dim=1)
                output_reference = model(**inputs_reference).last_hidden_state.mean(dim=1)
            
            similarity_score = F.cosine_similarity(output_student, output_reference).item()
            scores.append(round(similarity_score * 100, 2))
        else:
            scores.append(0)
    return scores

def evaluate_coding(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    try:
        result = subprocess.run(["python", temp_file_path], capture_output=True, text=True, timeout=5)
        coding_score = 100 if "120" in result.stdout else 50
    except Exception:
        coding_score = 0
    return coding_score

# Scorecard Section
if st.button("📊 View Scorecard"):
    if None in [st.session_state.get("mcq_score"), st.session_state.get("subjective_scores"), st.session_state.get("coding_score")]:
        st.warning("⚠️ Complete all sections to view the scorecard!")
    else:
        scores = {
            "MCQ Score": st.session_state["mcq_score"],
            "Subjective Score": sum(st.session_state["subjective_scores"]) / len(st.session_state["subjective_scores"]),
            "Coding Score": st.session_state["coding_score"]
        }
        
        df_scores = pd.DataFrame(scores.items(), columns=["Section", "Score"])
        st.table(df_scores)

        # Visualization
        fig, ax = plt.subplots()
        sns.barplot(x=df_scores["Section"], y=df_scores["Score"], palette="Blues", ax=ax)
        plt.ylim(0, 100)
        st.pyplot(fig)

        # Topic-wise performance visualization
        if "topic_performance" in st.session_state:
            topic_data = pd.DataFrame(list(st.session_state["topic_performance"].items()), columns=["Topic", "Correct Answers"])
            topic_data["Total Questions"] = topic_data["Topic"].map(total_questions_per_topic)
            topic_data["Accuracy"] = (topic_data["Correct Answers"] / topic_data["Total Questions"]) * 100
            
            st.subheader("📊 Topic-wise Performance")
            fig, ax = plt.subplots()
            sns.barplot(x=topic_data["Topic"], y=topic_data["Accuracy"], palette="coolwarm", ax=ax)
            plt.xticks(rotation=90)
            plt.ylim(0, 100)
            st.pyplot(fig)
