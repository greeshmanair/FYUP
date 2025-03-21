import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import subprocess
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

# Load BERT tokenizer and model only once
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer, model = load_model()

# Load MCQ dataset
mcq_df = pd.read_excel("mcq_machine_learning.xlsx")

def load_subjective_questions():
    return pd.read_excel("subjective_questions_ml.xlsx")
subjective_df = load_subjective_questions()

# Function to grade MCQs and analyze topic performance
def grade_mcq(student_answers):
    correct_answers = mcq_df['Correct Answer'].dropna().values
    topics = mcq_df['Topic'].dropna().values  # Extracting topics for analysis
    topic_performance = {}
    
    if len(student_answers) != len(correct_answers):
        raise ValueError("Mismatch in number of responses.")
    
    correct_count = 0
    for i, (student_ans, correct_ans, topic) in enumerate(zip(student_answers, correct_answers, topics)):
        if student_ans.strip().lower() == correct_ans.strip().lower():
            correct_count += 1
            topic_performance[topic] = topic_performance.get(topic, 0) + 1
        else:
            topic_performance[topic] = topic_performance.get(topic, 0)
    
    score = (correct_count / len(correct_answers)) * 100
    return round(score, 2), topic_performance

# Identify strong and weak topics
def analyze_topics(topic_performance, total_questions_per_topic):
    strong_topics = []
    weak_topics = []
    
    for topic, correct_count in topic_performance.items():
        accuracy = (correct_count / total_questions_per_topic[topic]) * 100
        if accuracy > 75:
            strong_topics.append(topic)
        elif accuracy < 50:
            weak_topics.append(topic)
    
    return strong_topics, weak_topics

# Generate personalized feedback
def gene_feedback(strong_topics, weak_topics):
    feedback = "\n### Personalized Feedback\n"
    if strong_topics:
        feedback += "✅ Great job! You have a strong understanding of: " + ", ".join(strong_topics) + "\n"
    if weak_topics:
        feedback += "⚠️ Areas to Improve: " + ", ".join(weak_topics) + "\n"
        feedback += "\n📚 Recommended Tutorials:\n"
        for topic in weak_topics:
            feedback += f"🔗 [Learn {topic}](https://www.google.com/search?q={topic}+tutorial)\n"
    return feedback

# Compute similarity score
def compute_similarity(answer, reference_answer):
    if not answer.strip():
        return 0.0  # If empty response, return lowest score

    inputs = tokenizer([answer, reference_answer], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    answer_embedding = outputs.last_hidden_state.mean(dim=1)[0]
    reference_embedding = outputs.last_hidden_state.mean(dim=1)[1]

    similarity_score = F.cosine_similarity(answer_embedding.unsqueeze(0), reference_embedding.unsqueeze(0)).item()
    return round(similarity_score, 2)

# Generate feedback
def generate_feedback(similarity_score):
    if similarity_score > 0.85:
        return "✅ Excellent answer! Well-structured and relevant."
    elif similarity_score > 0.65:
        return "🟡 Good attempt, but needs more clarity and depth."
    else:
        return "❌ Needs improvement. Focus on key points and provide a structured response."

# Store section-wise results
st.session_state.setdefault("subjective_scores", [])
st.session_state.setdefault("coding_score", None)

# Count total questions per topic
total_questions_per_topic = mcq_df['Topic'].value_counts().to_dict()

# Streamlit UI
st.title("📚 Smart Grading & Feedback System")

section = st.sidebar.radio("Select Section", ["MCQ","Subjective", "Coding", "Scorecard"])


if section == "MCQ":
    st.write("### Answer the MCQs Below:")
    student_mcq_answers = []

    for index, row in mcq_df.iterrows():
        question = row["Question"]
        options = [row["Option A"], row["Option B"], row["Option C"], row["Option D"]]
        answer = st.radio(row['Question'], options, key=f"mcq_{index}")
        student_mcq_answers.append(chr(97 + options.index(answer)))
    
    if st.button("🔍 Evaluate MCQs"):
        try:
            st.session_state['mcq_score'], topic_performance = grade_mcq(student_mcq_answers)
            st.session_state['topic_performance'] = topic_performance
            st.success(f"MCQ Score: {st.session_state['mcq_score']:.2f}%")
        except ValueError as e:
            st.error(str(e))

if section == "Subjective":
    st.write("### Answer the Questions Below:")
    responses = {}

    for index, row in subjective_df.iterrows():
        question = row["Question"]
        reference_answer = row["Sample Answer"]
        student_response = st.text_area(f"{index+1}. {question}", "")
        responses[question] = (student_response, reference_answer)

    if st.button("🔍 Evaluate Answers"):
        subjective_scores = [compute_similarity(s, r) * 100 for s, r in responses.values()]
        st.session_state["subjective_scores"] = subjective_scores
        st.success("Subjective answers evaluated!")

if section == "Coding":
    st.write("### Write a code to find the factorial of 5")
    uploaded_file = st.file_uploader("Upload Python Code", type=["py", "ipynb"])
    if uploaded_file and st.button("Submit Code"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        try:
            result = subprocess.run(["python", temp_file_path], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip() == "120":
                coding_score = 100
                feedback = "✅ Correct output! Well done."
            else:
                coding_score = 50
                feedback = f"❌ Incorrect output. Expected 120, but got {result.stdout.strip()}"
        except Exception as e:
            coding_score = 0
            feedback = f"❌ Failed to execute. Error: {str(e)}"

        st.session_state["coding_score"] = coding_score
        st.success(feedback)

if section == "Scorecard":
    st.header("Scorecard")
    
    if "mcq_score" not in st.session_state or "topic_performance" not in st.session_state:
        st.warning("Complete MCQ section to view the scorecard!")
    else:
        st.write(f"MCQ Score: {st.session_state['mcq_score']}%")
        strong_topics, weak_topics = analyze_topics(st.session_state['topic_performance'], total_questions_per_topic)
        
        st.subheader("📊 Performance Analysis")
        st.write(f"✅ Strong Topics: {', '.join(strong_topics) if strong_topics else 'None'}")
        st.write(f"⚠️ Weak Topics: {', '.join(weak_topics) if weak_topics else 'None'}")
        
        # Visualization
        topic_data = pd.DataFrame(list(st.session_state['topic_performance'].items()), columns=["Topic", "Correct Answers"])
        topic_data["Total Questions"] = topic_data["Topic"].map(total_questions_per_topic)
        topic_data["Accuracy"] = (topic_data["Correct Answers"] / topic_data["Total Questions"]) * 100
        
        fig, ax = plt.subplots()
        sns.barplot(x=topic_data["Topic"], y=topic_data["Accuracy"], palette="coolwarm", ax=ax)
        plt.xticks(rotation=90)
        plt.ylim(0, 100)
        st.pyplot(fig)
        
        # Display personalized feedback
        feedback = gene_feedback(strong_topics, weak_topics)
        st.markdown(feedback, unsafe_allow_html=True)

st.sidebar.write("Complete all sections to unlock Scorecard!")
