#install the requirements
!pip install nltk scikit-learn rouge-score bert-score

#import the libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Download necessary NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


#extractive summarization
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def extractive_summary(text, num_sentences=2):
    # Step 1: Split into sentences
    sentences = sent_tokenize(text)

    # Handle cases with very few sentences
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    if not sentences:
        return ""

    # Step 2: Clean the sentences and remove stopwords for TF-IDF calculation
    stop_words = set(stopwords.words('english'))
    cleaned_sentences_for_tfidf = []
    for s in sentences:
        # Remove non-alphabetic characters, convert to lowercase, and tokenize
        words = word_tokenize(re.sub(r'[^a-zA-Z]', ' ', s).lower())
        # Remove stopwords and join back
        cleaned_sentence = ' '.join([word for word in words if word not in stop_words and word.strip() != ''])
        cleaned_sentences_for_tfidf.append(cleaned_sentence)

    # Step 3: Vectorize with TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences_for_tfidf)

    # Step 4: Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Step 5: Score sentences
    # Sum the similarity scores for each sentence (excluding self-similarity for better ranking)
    sentence_scores = similarity_matrix.sum(axis=1)

    # Step 6: Pick top N sentences, keeping original order
    # Get indices of top N sentences
    top_indices_unsorted = sentence_scores.argsort()[-num_sentences:][::-1]

    # Sort these indices to preserve original sentence order
    top_indices_sorted = sorted(top_indices_unsorted)

    summary = " ".join([sentences[i] for i in top_indices_sorted])
    return summary

# Example usage
text_example = """
Artificial Intelligence is transforming industries across the globe.
From healthcare to finance, AI technologies are revolutionizing how businesses operate.
In healthcare, AI helps diagnose diseases, personalize treatments, and improve patient care.
In finance, it assists in fraud detection, algorithmic trading, and customer service.
As the field evolves, ethical considerations and transparency remain critical for responsible AI adoption.
"""

print("--- Extractive Summary (Top 2 Sentences) ---")
summary_output = extractive_summary(text_example, num_sentences=2)
print(summary_output)

#evaluation for extractive summarization
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def extractive_summary(text, num_sentences=2):
    # Step 1: Split into sentences
    sentences = sent_tokenize(text)

    # Handle cases with very few sentences
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    if not sentences:
        return ""

    # Step 2: Clean the sentences and remove stopwords for TF-IDF calculation
    stop_words = set(stopwords.words('english'))
    cleaned_sentences_for_tfidf = []
    for s in sentences:
        # Remove non-alphabetic characters, convert to lowercase, and tokenize
        words = word_tokenize(re.sub(r'[^a-zA-Z]', ' ', s).lower())
        # Remove stopwords and join back
        cleaned_sentence = ' '.join([word for word in words if word not in stop_words and word.strip() != ''])
        cleaned_sentences_for_tfidf.append(cleaned_sentence)

    # Step 3: Vectorize with TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences_for_tfidf)

    # Step 4: Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Step 5: Score sentences
    # Sum the similarity scores for each sentence (excluding self-similarity for better ranking)
    sentence_scores = similarity_matrix.sum(axis=1)

    # Step 6: Pick top N sentences, keeping original order
    # Get indices of top N sentences
    top_indices_unsorted = sentence_scores.argsort()[-num_sentences:][::-1]

    # Sort these indices to preserve original sentence order
    top_indices_sorted = sorted(top_indices_unsorted)

    summary = " ".join([sentences[i] for i in top_indices_sorted])
    return summary

# Example usage
text_example = """
Artificial Intelligence is transforming industries across the globe.
From healthcare to finance, AI technologies are revolutionizing how businesses operate.
In healthcare, AI helps diagnose diseases, personalize treatments, and improve patient care.
In finance, it assists in fraud detection, algorithmic trading, and customer service.
As the field evolves, ethical considerations and transparency remain critical for responsible AI adoption.
"""

print("--- Extractive Summary (Top 2 Sentences) ---")
summary_output = extractive_summary(text_example, num_sentences=2)
print(summary_output)


# Dummy Dataset: Text documents and their human-written reference summaries


sample_data = [
    {
        'document': """
        Artificial Intelligence (AI) is rapidly transforming various sectors, including healthcare, finance, and education.
        In healthcare, AI is being used for disease diagnosis, drug discovery, and personalized treatment plans, leading to more efficient and effective patient care.
        Financial institutions leverage AI for fraud detection, algorithmic trading, and customer support via chatbots, enhancing security and operational efficiency.
        The education sector uses AI for adaptive learning platforms, automated grading, and personalized tutoring, aiming to improve learning outcomes.
        However, the widespread adoption of AI also raises significant ethical concerns, such as data privacy, algorithmic bias, and job displacement.
        Ensuring transparency, fairness, and accountability in AI systems is crucial for responsible development and deployment.
        Researchers are actively working on developing explainable AI (XAI) to address the 'black box' problem and build trust in AI decisions.
        """,
        'reference_summary': "Artificial Intelligence is transforming sectors like healthcare, finance, and education. It helps diagnose diseases, detect fraud, and personalize learning. Ethical concerns such as privacy, bias, and job displacement need careful consideration for responsible AI adoption."
    },
    {
        'document': """
        Climate change is one of the most pressing issues facing humanity today.
        Rising global temperatures lead to a multitude of adverse effects, including more frequent and intense heatwaves, melting glaciers, and rising sea levels.
        These changes threaten ecosystems, biodiversity, and human settlements, particularly coastal communities.
        The primary driver of climate change is the emission of greenhouse gases, predominantly from the burning of fossil fuels for energy, transportation, and industry.
        Mitigation efforts include transitioning to renewable energy sources, improving energy efficiency, and implementing carbon capture technologies.
        Adaptation strategies involve building resilient infrastructure, developing drought-resistant crops, and improving early warning systems for extreme weather events.
        International cooperation, such as agreements made at COP conferences, is essential to address this global challenge effectively.
        Individual actions, like reducing carbon footprints and promoting sustainable consumption, also play a vital role.
        """,
        'reference_summary': "Climate change, driven by greenhouse gas emissions, causes rising temperatures, heatwaves, and sea level rise, threatening ecosystems and communities. Solutions involve transitioning to renewable energy, improving energy efficiency, and international cooperation, alongside individual efforts to reduce carbon footprints."
    }
]


generated_summaries = []
reference_summaries = []
evaluation_results = []

# Choose the number of sentences for your extractive summary
NUM_SUMMARY_SENTENCES = 3

print(f"--- Running Extractive Summarization and Evaluation (Top {NUM_SUMMARY_SENTENCES} sentences) ---\n")

for i, item in enumerate(sample_data):
    document = item['document']
    reference = item['reference_summary']

    # Generate summary using this function
    gen_summary = extractive_summary(document, num_sentences=NUM_SUMMARY_SENTENCES)

    generated_summaries.append(gen_summary)
    reference_summaries.append(reference)

    print(f"--- Example {i+1} ---")
    print("Original Document:\n", document[:200], "...\n") # Print first 200 chars
    print("Reference Summary:\n", reference, "\n")
    print("Generated Summary:\n", gen_summary, "\n")

    # Calculate ROUGE scores for this example
    rouge_scores = calculate_rouge(gen_summary, reference)
    print("ROUGE Scores (F1):", rouge_scores)
    evaluation_results.append(rouge_scores)
    print("-" * 30)

print("\n--- Aggregated Evaluation Results ---")


bert_scores = calculate_bertscore(generated_summaries, reference_summaries)
print("Average BERTScore:", bert_scores)



#abstractive summarization
#install dependies
!pip install transformers torch


from transformers import pipeline

def abstractive_summarize(text, max_length=150, min_length=30, model_name="sshleifer/distilbart-cnn-12-6"):
    """
    Generates an abstractive summary of the given text using a pre-trained
    Hugging Face Transformer model.

    Args:
        text (str): The input text to summarize.
        max_length (int): The maximum length of the generated summary in tokens.
        min_length (int): The minimum length of the generated summary in tokens.
        model_name (str): The name of the pre-trained model to use.
                          Common choices:
                          - "sshleifer/distilbart-cnn-12-6" (good balance of speed/quality)
                          - "facebook/bart-large-cnn" (higher quality, larger)
                          - "google/pegasus-cnn_dailymail" (another strong option)
                          - "t5-small" / "t5-base" (T5 models, often good for summarization)

    Returns:
        str: The abstractive summary.
    """
    if not text or text.strip() == "":
        return "No text provided for summarization."

    # Initialize the summarization pipeline with the specified pre-trained model
    try:
        # The pipeline automatically handles tokenization, model inference, and text generation.
        summarizer = pipeline("summarization", model=model_name)
    except Exception as e:
        return f"Error loading summarization model '{model_name}': {e}\n" \
               "Please ensure you have an active internet connection to download the model " \
               "and sufficient memory/disk space. Also, check if the model name is correct."

    # Generate the summary
    # 'truncation=True' handles input texts that are longer than the model's maximum input size.
    # 'do_sample=False' ensures deterministic output (no randomness in generation)
    # 'num_beams' can be adjusted for better quality (higher means more exploration, but slower)
    summary = summarizer(text, max_length=max_length, min_length=min_length,
                         do_sample=False, truncation=True, num_beams=4) # num_beams=4 is a common default

    # The pipeline returns a list of dictionaries, extract the summary text
    return summary[0]['summary_text']

if __name__ == "__main__":
    long_article = """
    The rapid advancements in Artificial Intelligence (AI) are fundamentally reshaping various aspects of human society, from daily routines to complex industrial operations. In the healthcare sector, AI is proving to be revolutionary, aiding in the early diagnosis of diseases like cancer and Alzheimer's by analyzing medical images with high precision. It also accelerates drug discovery processes by simulating molecular interactions and predicting potential drug candidates, significantly reducing the time and cost associated with traditional research methods. Furthermore, AI-powered systems are enabling personalized treatment plans based on a patient's genetic makeup and medical history, leading to more effective interventions and improved patient outcomes.

    Beyond healthcare, AI is transforming the financial industry. It enhances fraud detection capabilities by identifying unusual transaction patterns in real-time, far surpassing human ability. Algorithmic trading, driven by AI, executes trades at lightning speed based on complex market analysis, optimizing investment strategies. Customer service is also being revolutionized through AI-driven chatbots and virtual assistants that provide instant support, answer queries, and even handle complex banking operations, improving customer satisfaction and reducing operational costs.

    The education sector is not immune to AI's influence. Adaptive learning platforms powered by AI can tailor educational content and pace to individual student needs, making learning more engaging and effective. Automated grading systems free up educators' time, allowing them to focus more on teaching and less on administrative tasks. Personalized tutoring systems can identify a student's weaknesses and provide targeted exercises and explanations.

    However, the proliferation of AI also brings forth significant ethical considerations. Concerns around data privacy, especially with large datasets used for training, are paramount. Algorithmic bias, where AI systems inadvertently perpetuate or amplify societal biases present in their training data, can lead to unfair or discriminatory outcomes. The potential for job displacement across various industries due to automation is another critical societal challenge. Ensuring transparency, fairness, and accountability in AI systems is crucial for responsible development and deployment. Researchers are actively working on developing explainable AI (XAI) to address the 'black box' problem, making AI decisions understandable to humans and fostering trust in these powerful technologies.
    """

    print("--- Abstractive Summary (Default Settings) ---")
    summary_default = abstractive_summarize(long_article)
    print(summary_default)
    print("\n" + "="*70 + "\n")

    print("--- Abstractive Summary (Shorter, PEGAsus Model) ---")
    # PEGASUS models are often good for abstractive summarization
    summary_short_pegasus = abstractive_summarize(long_article, max_length=60, min_length=20,
                                                  model_name="google/pegasus-cnn_dailymail")
    print(summary_short_pegasus)
    print("\n" + "="*70 + "\n")

    print("--- Abstractive Summary (Longer, BART Model) ---")
    summary_long_bart = abstractive_summarize(long_article, max_length=200, min_length=80,
                                               model_name="facebook/bart-large-cnn")
    print(summary_long_bart)


#evaluation for abstractive summarization
from transformers import pipeline
from rouge_score import rouge_scorer
from bert_score import score
import pandas as pd

# Define your abstractive_summarize function
def abstractive_summarize(text, max_length=150, min_length=30, model_name="sshleifer/distilbart-cnn-12-6"):
    """
    Generates an abstractive summary of the given text using a pre-trained
    Hugging Face Transformer model.

    Args:
        text (str): The input text to summarize.
        max_length (int): The maximum length of the generated summary in tokens.
        min_length (int): The minimum length of the generated summary in tokens.
        model_name (str): The name of the pre-trained model to use.

    Returns:
        str: The abstractive summary.
    """
    if not text or text.strip() == "":
        return "No text provided for summarization."

    try:
        # The pipeline automatically handles tokenization, model inference, and text generation.
        # Adding trust_remote_code=True for potential custom model components if needed
        # trust_remote_code=True is only needed for models with custom code, removing for standard models
        summarizer = pipeline("summarization", model=model_name) # Removed trust_remote_code=True
    except Exception as e:
        return f"Error loading summarization model '{model_name}': {e}\n" \
               "Please ensure you have an active internet connection to download the model " \
               "and sufficient memory/disk space. Also, check if the model name is correct."

    # Generate the summary
    # 'truncation=True' handles input texts that are longer than the model's maximum input size.
    # 'do_sample=False' ensures deterministic output (no randomness in generation)
    # 'num_beams' can be adjusted for better quality (higher means more exploration, but slower)
    summary = summarizer(text, max_length=max_length, min_length=min_length,
                         do_sample=False, truncation=True, num_beams=4) # num_beams=4 is a common default

    # The pipeline returns a list of dictionaries, extract the summary text
    return summary[0]['summary_text']

# --- Evaluation Metrics Functions ---

def calculate_rouge(generated_summary, reference_summary):
    """
    Calculates ROUGE scores (1, 2, L) for a single generated summary against a single reference.
    Args:
        generated_summary (str): The summary generated by your model.
        reference_summary (str): The human-written reference summary.
    Returns:
        dict: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)

    rouge_1_f = scores['rouge1'].fmeasure
    rouge_2_f = scores['rouge2'].fmeasure
    rouge_l_f = scores['rougeL'].fmeasure

    return {
        'ROUGE-1 F1': f"{rouge_1_f:.4f}",
        'ROUGE-2 F1': f"{rouge_2_f:.4f}",
        'ROUGE-L F1': f"{rouge_l_f:.4f}"
    }

def calculate_bertscore_batch(generated_summaries_list, reference_summaries_list):
    """
    Calculates BERTScore for a batch of generated summaries against references.
    Args:
        generated_summaries_list (list of str): List of generated summaries.
        reference_summaries_list (list of str): List of human-written reference summaries.
    Returns:
        dict: A dictionary containing average BERTScore Precision, Recall, and F1.
    """
    if not generated_summaries_list or not reference_summaries_list:
        return {'BERTScore P': 'N/A', 'BERTScore R': 'N/A', 'BERTScore F1': 'N/A'}

    # Use 'roberta-large' for better performance, or 'bert-base-uncased' for faster/smaller.
    # Set verbose=False to suppress detailed score printing for each sentence if you have many.
    # Adding trust_remote_code=True for potential custom model components if needed
    # Removed trust_remote_code=True as it's not supported by bert_score.score in this version
    P, R, F1 = score(generated_summaries_list, reference_summaries_list, lang="en", verbose=False)

    return {
        'BERTScore P': f"{P.mean().item():.4f}",
        'BERTScore R': f"{R.mean().item():.4f}",
        'BERTScore F1': f"{F1.mean().item():.4f}"
    }

if __name__ == "__main__":
    # --- DUMMY DATASET FOR DEMONSTRATION ---
    # IMPORTANT: For a meaningful evaluation, you need a diverse and sufficiently large
    # dataset with human-written reference summaries.
    sample_data_for_eval = [
        {
            'document': """
            The rapid advancements in Artificial Intelligence (AI) are fundamentally reshaping various aspects of human society, from daily routines to complex industrial operations. In the healthcare sector, AI is proving to be revolutionary, aiding in the early diagnosis of diseases like cancer and Alzheimer's by analyzing medical images with high precision. It also accelerates drug discovery processes by simulating molecular interactions and predicting potential drug candidates, significantly reducing the time and cost associated with traditional research methods. Furthermore, AI-powered systems are enabling personalized treatment plans based on a patient's genetic makeup and medical history, leading to more effective interventions and improved patient outcomes.

            Beyond healthcare, AI is transforming the financial industry. It enhances fraud detection capabilities by identifying unusual transaction patterns in real-time, far surpassing human ability. Algorithmic trading, driven by AI, executes trades at lightning speed based on complex market analysis, optimizing investment strategies. Customer service is also being revolutionized through AI-driven chatbots and virtual assistants that provide instant support, answer queries, and even handle complex banking operations, improving customer satisfaction and reducing operational costs.

            The education sector is not immune to AI's influence. Adaptive learning platforms powered by AI can tailor educational content and pace to individual student needs, making learning more engaging and effective. Automated grading systems free up educators' time, allowing them to focus more on teaching and less on administrative tasks. Personalized tutoring systems can identify a student's weaknesses and provide targeted exercises and explanations.

            However, the proliferation of AI also brings forth significant ethical considerations. Concerns around data privacy, especially with large datasets used for training, are paramount. Algorithmic bias, where AI systems inadvertently perpetuate or amplify societal biases present in their training data, can lead to unfair or discriminatory outcomes. The potential for job displacement across various industries due to automation is another critical societal challenge. Ensuring transparency, fairness, and accountability in AI systems is crucial for responsible development and deployment. Researchers are actively working on developing explainable AI (XAI) to address the 'black box' problem, making AI decisions understandable to humans and fostering trust in these powerful technologies.
            """,
            'reference_summary': "Artificial Intelligence is rapidly transforming healthcare, finance, and education. It helps diagnose diseases, accelerate drug discovery, improve customer service, and personalize learning. Ethical concerns like data privacy, bias, and job displacement need careful consideration for responsible AI adoption and development of explainable AI."
        },
        {
            'document': """
            Renewable energy sources are becoming increasingly vital in the global effort to combat climate change and reduce reliance on fossil fuels. Solar power, harnessing energy from sunlight through photovoltaic panels, is one of the most popular and rapidly growing forms of clean energy. Its versatility allows for large-scale solar farms and smaller rooftop installations. Wind power, generated by wind turbines, is another significant contributor, particularly in coastal and high-altitude regions. Advances in turbine technology have made it more efficient and cost-effective.

            Hydropower, which uses the force of moving water to generate electricity, is a well-established renewable source, often from large dams. Geothermal energy taps into the Earth's internal heat, using steam or hot water from underground reservoirs. Bioenergy, derived from organic matter like agricultural waste and dedicated energy crops, offers another pathway to renewable power, although its sustainability can depend on sourcing methods.

            The transition to these clean energy sources offers numerous benefits, including reduced greenhouse gas emissions, improved air quality, and energy independence. However, challenges remain, such as intermittency (solar and wind depend on weather), land use requirements, and initial infrastructure costs. Research and development continue to focus on improving energy storage solutions, grid integration, and overall efficiency to accelerate the global shift towards a sustainable energy future.
            """,
            'reference_summary': "Renewable energy, including solar, wind, hydro, geothermal, and bioenergy, is crucial for combating climate change and reducing fossil fuel reliance. While offering benefits like reduced emissions and energy independence, challenges such as intermittency and high costs persist, driving ongoing research into storage and grid integration."
        }
    ]

    # --- Define model configurations to evaluate ---
    # This list allows you to easily loop through and evaluate multiple models/parameter sets.
    model_configs = [
        {"name": "sshleifer/distilbart-cnn-12-6", "max_length": 120, "min_length": 30},
        {"name": "google/pegasus-cnn_dailymail", "max_length": 60, "min_length": 20},
        {"name": "facebook/bart-large-cnn", "max_length": 200, "min_length": 80},
        # Add more models or different parameter sets here if you want to compare
        # {"name": "t5-small", "max_length": 80, "min_length": 20}
    ]

    all_evaluation_results = []

    print("\n--- Starting Comprehensive Abstractive Summarization Evaluation ---")

    for config in model_configs:
        model_name = config["name"]
        max_len = config["max_length"]
        min_len = config["min_length"]

        print(f"\n======== Evaluating Model: {model_name} (max_len={max_len}, min_len={min_len}) ========")

        generated_summaries_for_batch_eval = []
        reference_summaries_for_batch_eval = []
        current_model_rouge_results = []

        for i, item in enumerate(sample_data_for_eval):
            document = item['document']
            reference = item['reference_summary']

            # Generate summary using the abstractive function
            gen_summary = abstractive_summarize(document,
                                                max_length=max_len,
                                                min_length=min_len,
                                                model_name=model_name)

            # Handle PEGASUS specific newline token for cleaner output
            if model_name == "google/pegasus-cnn_dailymail":
                gen_summary = gen_summary.replace('<n>', ' ').strip()

            generated_summaries_for_batch_eval.append(gen_summary)
            reference_summaries_for_batch_eval.append(reference)

            # Calculate and store ROUGE scores for each individual example
            rouge_scores = calculate_rouge(gen_summary, reference)
            current_model_rouge_results.append(rouge_scores)

            print(f"\n--- Example {i+1} for {model_name} ---")
            print("Reference Summary:\n", reference, "\n")
            print("Generated Summary:\n", gen_summary, "\n")
            print("ROUGE Scores (F1):", rouge_scores)
            print("-" * 30)

        # Calculate aggregated BERTScore for the current model across all examples
        print(f"\nCalculating BERTScore for {model_name}...")
        # Removed trust_remote_code=True from the bert_score.score call
        bert_scores = calculate_bertscore_batch(generated_summaries_for_batch_eval, reference_summaries_for_batch_eval)
        print("Average BERTScore:", bert_scores)

        # Aggregate ROUGE results for the current model
        avg_rouge_1 = sum([float(res['ROUGE-1 F1']) for res in current_model_rouge_results]) / len(current_model_rouge_results)
        avg_rouge_2 = sum([float(res['ROUGE-2 F1']) for res in current_model_rouge_results]) / len(current_model_rouge_results)
        avg_rouge_l = sum([float(res['ROUGE-L F1']) for res in current_model_rouge_results]) / len(current_model_rouge_results)

        print(f"Average ROUGE-1 F1 for {model_name}: {avg_rouge_1:.4f}")
        print(f"Average ROUGE-2 F1 for {model_name}: {avg_rouge_2:.4f}")
        print(f"Average ROUGE-L F1 for {model_name}: {avg_rouge_l:.4f}")

        # Store overall results for comparison in a list
        all_evaluation_results.append({
            'Model': model_name,
            'Max Length': max_len,
            'Min Length': min_len,
            'Avg ROUGE-1 F1': avg_rouge_1,
            'Avg ROUGE-2 F1': avg_rouge_2,
            'Avg ROUGE-L F1': avg_rouge_l,
            'Avg BERTScore P': float(bert_scores['BERTScore P']),
            'Avg BERTScore R': float(bert_scores['BERTScore R']),
            'Avg BERTScore F1': float(bert_scores['BERTScore F1'])
        })
        print("=" * 80)

    # Display a final summary table of all models' performance
    print("\n--- Overall Evaluation Summary (All Models) ---")
    results_df = pd.DataFrame(all_evaluation_results)
    print(results_df.round(4))
