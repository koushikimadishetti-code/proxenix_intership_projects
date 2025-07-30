# proxenix_intership_projects
# 📰 Text Summarization Project: Bridging Readability and Information Extraction

This project delves into the fascinating world of text summarization, offering a hands-on exploration of both traditional and cutting-edge techniques.  
My work focuses on building robust systems that can condense lengthy documents into concise, coherent summaries — a critical task in today's information-rich environment.

---

## 📦 What's Inside?

### 📝 Extractive Summarization

I've implemented methods that extract the most important sentences directly from the original text to form a summary.

- **Custom TF-IDF & Cosine Similarity**  
  A foundational approach where sentences are ranked based on the significance of their words (TF-IDF) and their similarity to other sentences, allowing the system to identify key information.

- **sumy Library Integration**  
  For broader exploration, I've incorporated the `sumy` Python library, demonstrating its capabilities with algorithms like:
  - TextRank  
  - Latent Semantic Analysis (LSA)  
  for efficient extractive summarization.

---

### ✨ Abstractive Summarization

Moving beyond simple extraction, I've ventured into generative models that create new, human-like summaries by rephrasing and synthesizing information from the source text.

- **Hugging Face Transformers Power**  
  This core component leverages state-of-the-art pre-trained Transformer models from the Hugging Face Transformers library.

- **Diverse Model Exploration**  
  My implementation showcases the performance of various abstractive models, including:

  - `sshleifer/distilbart-cnn-12-6` – a faster, distilled BART model  
  - `google/pegasus-cnn_dailymail` – known for generating highly abstractive summaries  
  - `facebook/bart-large-cnn` – a larger, powerful BART model for nuanced summarization

---

## 📊 Comprehensive Evaluation

Understanding performance is key. I've integrated robust evaluation metrics to objectively assess the quality of the generated summaries.

- **ROUGE Scores**  
  I've used ROUGE (Recall-Oriented Understudy for Gisting Evaluation) to quantify the overlap of n-grams (individual words, pairs of words, and longest common sequences) between the generated summaries and human-written reference summaries:
  - ROUGE-1  
  - ROUGE-2  
  - ROUGE-L  

- **BERTScore for Semantic Richness**  
  To go beyond simple word matching, I've included **BERTScore**, which utilizes contextual embeddings to measure the semantic similarity between the generated and reference summaries, providing a deeper insight into meaning preservation.

---

## 🚀 Getting Started

This project is best explored through its **Google Colab notebook**, which guides you through each step from setup to evaluation.

---

