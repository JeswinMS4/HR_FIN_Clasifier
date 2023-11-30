import streamlit as st
import random
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

# Load dataset
dataset = load_dataset("JeswinMS4/HR_FIN_CLASSIFIER_DATA")
final_data = pd.DataFrame(dataset['train'])

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit app
def main():
    st.title("Inference Using Sentence Transformers for Query Classifier")
    st.header("Multi Level Gen AI")

    # Accept query input
    query = st.text_input("Enter your query:")

    # Slider for selecting number of samples
    num_samples = st.slider("Select number of samples (20-400)", 20, 400, 20, step=20)

    # Slider for selecting similarity threshold
    similarity_threshold = st.slider("Select similarity threshold (0 to 1)", 0.0, 1.0, 0.5, step=0.01)

    if st.button("Run Inference"):
        st.text("Running inference...")

        # Extract random samples based on labels
        random_hr_samples = random.sample(final_data[final_data['label'] == 'HR']['text'].tolist(), num_samples // 2)
        random_finance_samples = random.sample(final_data[final_data['label'] == 'Finance']['text'].tolist(), num_samples // 2)

        # Encode query and samples
        query_embedding = model.encode(query, convert_to_tensor=True)
        hr_embeddings = model.encode(random_hr_samples, convert_to_tensor=True)
        finance_embeddings = model.encode(random_finance_samples, convert_to_tensor=True)

        # Calculate cosine similarities
        hr_scores = util.cos_sim(query_embedding, hr_embeddings)
        finance_scores = util.cos_sim(query_embedding, finance_embeddings)

        # Calculate average scores
        avg_hr_score = torch.mean(hr_scores).item()
        avg_finance_score = torch.mean(finance_scores).item()

        # Display results
        st.text(f"Average Score for HR: {avg_hr_score:.4f}")
        st.text(f"Average Score for Finance: {avg_finance_score:.4f}")
        # Determine the higher scoring label
        if avg_hr_score > similarity_threshold and avg_finance_score > similarity_threshold:
            if avg_hr_score > avg_finance_score:
                st.text("Agent Called: HR")
            else:
                st.text("Agent Called: Finance")
        elif avg_hr_score > similarity_threshold:
            st.text("Agent Called: HR (Finance score below threshold)")
        elif avg_finance_score > similarity_threshold:
            st.text("Agent Called: Finance (HR score below threshold)")
        else:
            st.text("Query Out of Scope")

        # Display DataFrame
        st.subheader("Sampled Data:")
        sampled_data = pd.DataFrame({'Text': random_hr_samples + random_finance_samples,
                                     'Label': ['HR'] * len(random_hr_samples) + ['Finance'] * len(random_finance_samples)})
        st.dataframe(sampled_data)

    # Button to display complete dataset
    if st.button("Show Complete Dataset"):
        st.subheader("Complete Dataset:")
        st.dataframe(final_data)
        st.button("Close Dataset")


if __name__ == "__main__":
    main()
