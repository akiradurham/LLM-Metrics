import os
from dotenv import load_dotenv
# import openai
import pandas as pd
import time
import numpy as np
from datasets import load_dataset
from sklearn.metrics import mean_squared_error
import streamlit as st
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, SafetySettingDict, HarmCategory, HarmBlockThreshold
from rouge_score import rouge_scorer

# Helper RMSE calculator method
def calculate_rmse(true_scores, predicted_scores):
    mse = mean_squared_error(true_scores, predicted_scores)
    return np.sqrt(mse)

# Load API KEY
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

# Load dataset for semantic similarity, top 10 rows
dataset = load_dataset("mteb/stsbenchmark-sts", split="train[:10]")
sentence_pairs = [(row['sentence1'], row['sentence2']) for row in dataset]
true_scores = [row['score'] for row in dataset]

# Load models
model1 = genai.GenerativeModel("gemini-1.5-flash")
model2 = genai.GenerativeModel("gemini-1.5-flash-8b")
embedding_model = "models/text-embedding-004"

# Get model similarity between two sentences, calls model to provide score
def get_gemini_similarity1(sentence1, sentence2):
    prompt = (
        f"Your job is to provide a semantic similarity score between the following sentences and only respond with a numeric, float similarity score between 0 and 5. Do not provide any other response than the x.x formatted float number.\n"
        f"Sentence 1: {sentence1}\nSentence 2: {sentence2}"# \nScore: "
    )
    response = model1.generate_content(prompt)
    return float(response.text)

def get_gemini_similarity2(sentence1, sentence2):
    prompt = (
        f"Your job is to provide a semantic similarity score between the following sentences and only respond with a numeric, float similarity score between 0 and 5. Do not provide any other response than the x.x formatted float number.\n"
        f"Sentence 1: {sentence1}\nSentence 2: {sentence2}" #\nScore: "
    )
    response = model2.generate_content(prompt)
    return float(response.text)

# Calculate the cosine similarity, and scale to semantic similarity score from dataset (0-5)
def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1['embedding'])
    vector2 = np.array(vector2['embedding'])
    dot = np.dot(vector1, vector2)
    magnitude = (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    cos = dot / magnitude
    return ((cos + 1) / 2) * 5

# Get embedding similarity with embedding model, return cosine similarity score
def get_embedding_similarity(sentence1, sentence2):
    embedding1 = genai.embed_content(model=embedding_model, content=sentence1)
    embedding2 = genai.embed_content(model=embedding_model, content=sentence2)

    return cosine_similarity(embedding1, embedding2)

# Iterate over dataset and call above methods to get model and embedding scores
model1_scores = []
model2_scores = []
embedding_scores = []
i = 1
for sentence1, sentence2 in sentence_pairs:
    if i == 5:
        print("Waiting...")
        print()
        time.sleep(60)
    model1_score = get_gemini_similarity1(sentence1, sentence2)
    model1_scores.append(model1_score)
    
    model2_score = get_gemini_similarity2(sentence1, sentence2)
    model2_scores.append(model2_score)

    embedding_score = get_embedding_similarity(sentence1, sentence2)
    embedding_score = round(embedding_score, 3)
    embedding_scores.append(embedding_score)
    i += 1

print("\nGenerated Scores:")
print("True Scores:", true_scores)
print("Model1 Scores:", model1_scores)
print("Model2 Scores:", model2_scores)
print("Embd Scores:", embedding_scores)
print()

# Calculate the RMSE values of model and embeddings
mod1_rmse = round(calculate_rmse(true_scores, model1_scores),2)
mod1 = sum(model1_scores) / len(model1_scores)

mod2_rmse = round(calculate_rmse(true_scores, model2_scores),2)
mod2 = sum(model2_scores) / len(model2_scores)

embedding_rmse = round(calculate_rmse(true_scores, embedding_scores),2)
emb = sum(embedding_scores) / len(embedding_scores)

mod1_per_err = round(mod1/sum(true_scores),2) 
mod2_per_err = round(mod2/sum(true_scores),2) 
em_per_err = round(emb/sum(true_scores),2)

print(f"Model 1 Similarity Score RMSE: {mod1_rmse}")
print(f"Model 1 Percent Error Score: {mod1_per_err*100}%")

print(f"Model 2 Similarity Score RMSE: {mod2_rmse}")
print(f"Model 2 Percent Error Score: {mod2_per_err*100}%")

print(f"Embedding Similarity Score RMSE: {embedding_rmse}")
print(f"Embedding Percent Error Score: {round(em_per_err*100,2)}%")

print()

print("Waiting 60s")
time.sleep(60)
print()

# Load ROUGE dataset, cnndailymail with accompanying summarization
dataset = load_dataset("abisee/cnn_dailymail", "1.0.0", split="train[:12]")
dataset = dataset.select([i for i in range(len(dataset)) if i not in (7, 8)])
article_pairs = [(row['article'], row['highlights']) for row in dataset]

gen_config = GenerationConfig(
    max_output_tokens=250
)

# Get model's summary of article
def get_model_rogue1(article): # 250 char limit
    prompt = (
        f"Your job is to provide a 50 word, 250 character maximum, no more than 250 characters, summary of the following article.\n"
        f"Article: {article}\n Summary: "
    )
    response = model1.generate_content(prompt, generation_config=gen_config)
    # print(response)
    return response.text

def get_model_rogue2(article): # 250 char limit
    prompt = (
        f"Your job is to provide a 50 word, 250 character maximum, no more than 250 characters, summary of the following article.\n"
        f"Article: {article}\n Summary: "
    )
    response = model2.generate_content(prompt, generation_config=gen_config)
    # print(response)
    return response.text

# Iterate over dataset and call above methods to get summary scores
model_rogue_scores1 = []
model_rogue_scores2 = []
summary_rogue_scores = []
i = 1
for article, highlight in article_pairs:
    if i == 5:
        print("Waiting...")
        print()
        time.sleep(60)
    generated_summary1 = get_model_rogue1(article)
    generated_summary2 = get_model_rogue2(article)
    
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores1 = rouge.score(article, generated_summary1)
    scores2 = rouge.score(article, highlight)
    scores3 = rouge.score(article, generated_summary2)
    
    model_measure1 = (scores1['rouge1'].fmeasure + scores1['rouge2'].fmeasure + scores1['rougeL'].fmeasure) / 3
    given_measure = (scores2['rouge1'].fmeasure + scores2['rouge2'].fmeasure + scores2['rougeL'].fmeasure) / 3
    model_measure2 = (scores3['rouge1'].fmeasure + scores3['rouge2'].fmeasure + scores3['rougeL'].fmeasure) / 3
    
    model_rogue_scores1.append(model_measure1)
    model_rogue_scores2.append(model_measure2)
    summary_rogue_scores.append(given_measure)
    i += 1

# Calculate the RMSE values of model summary
model1_rmse = round(calculate_rmse(summary_rogue_scores, model_rogue_scores1),2)
mod1 = sum(model_rogue_scores1) / len(model_rogue_scores1)
mo_per_err1 = round(mod1/sum(summary_rogue_scores),5) 

model2_rmse = round(calculate_rmse(summary_rogue_scores, model_rogue_scores2),2)
mod2 = sum(model_rogue_scores2) / len(model_rogue_scores2)
mo_per_err2 = round(mod2/sum(summary_rogue_scores),5) 

print(f"Model 1 ROGUE Score RMSE: {model1_rmse}")
print(f"Model 1 Percent Error Score: {mo_per_err1*100}%")
print(f"Model 2 ROGUE Score RMSE: {model2_rmse}")
print(f"Model 2 Percent Error Score: {mo_per_err2*100}%")
print()


# Begin Streamlit dashboard
st.title('Model Evaluation Dashboard')

st.header("Models")
st.write("Model 1 : gpt-4o-mini")
st.write("Model 2 : gpt-4o-mini-2024-07-18")
st.write("Embeddings Model : text-embedding-3-small")

st.header("Semantic Similarity Score")
st.write(f"Model 1 Similarity Score RMSE: {mod1_rmse}")
st.write(f"Model 1 Percent Error Score: {round(mod1_per_err*100,2)}%")
st.write(f"Model 2 Similarity Score RMSE: {mod2_rmse}")
st.write(f"Model 2 Percent Error Score: {round(mod2_per_err*100,2)}%")
st.write(f"Embedding Similarity Score RMSE: {embedding_rmse}")
st.write(f"Embedding Percent Error Score: {round(em_per_err*100,2)}%")

st.subheader("Scores Table")
score_df = pd.DataFrame({
    "True Scores": true_scores,
    "Model 1 Scores": model1_scores,
    "Model 2 Scores": model2_scores,
    "Embedding Scores": embedding_scores,
})
st.table(score_df)

st.header("LLM Metrics")
st.write("ROUGE Metric")
st.write(f"Model 1 ROGUE Score RMSE: {model1_rmse}")
st.write(f"Model 1 Percent Error Score: {round(mo_per_err1*100,2)}%")
st.write(f"Model 2 ROGUE Score RMSE: {model2_rmse}")
st.write(f"Model 2 Percent Error Score: {round(mo_per_err2*100,2)}%")

st.subheader("Scores Table")
rouge_df = pd.DataFrame({
    "Dataset Scores": summary_rogue_scores,
    "Model 1 Scores": model_rogue_scores1,
    "Model 2 Scores": model_rogue_scores2
})
st.table(rouge_df)
