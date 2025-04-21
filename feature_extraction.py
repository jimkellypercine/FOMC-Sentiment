import os
import json
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import lda
import pandas as pd
from PyPDF2 import PdfReader
import re
from nltk import ngrams

most_frequent_words = []
most_rare_words = []

GOOD_PHRASES = [
    "grow", "growth", "expand", "expansion",
    "improve", "improvement", "recover", "recovery",
    "increase", "hire", "job", "hiring",
    "confidence", "confident", "optimism", "optimistic",
    "strong", "robust", "resilient", "healthy",
    "invest", "investment", "gain", "profit", "productive", "productivity",
    "bull", "bullish", "rally", "boom",
    "positive", "upturn", "surge", "outperform"
]

BAD_PHRASES = [
    "contract", "recede", "recession", "decline", "shrink",
    "unemploy", "layoff", "jobless",
    "fall", "drop", "plunge", "dip", "slide", "tumble",
    "reduce", "cut", "weak", "weaken", "slump",
    "bear", "bearish", "selloff", "panic",
    "negative", "worse", "worsen", "instability", "uncertainty", "volatile", "risk", "fragile", "slow",
    "unstable", "bankrupt", "default", "debt", "crisis", "collapse", "volatility"
]

SEED_TOPICS = {
    'monetary_policy': ['inflation', 'rate', 'stability'],
    'economic_growth': ['growth', 'activity', 'recovery'],
    'labor_market': ['unemployment', 'labor', 'employment'],
    'financial_markets': ['markets', 'credit', 'liquidity'],
    'asset_purchases': ['treasury', 'purchases', 'securities'],
    'global_risk': ['global', 'geopolitical', 'foreign'],
    'governance_communication': ['committee', 'statement', 'guidance'],
    'mandate_objectives': ['mandate', 'objective', 'stability'],
    'market_operations': ['market', 'reserve', 'repo'],
    'fiscal_external': ['fiscal', 'spending', 'shock']
}

TOPIC_KEYWORDS = {
    'macroeconomic_commentary': {
        'words': ['policy', 'rate', 'inflation', 'labor', 'employment', 'unemployment', 'activity', 
                'condition', 'spending', 'support', 'recovery'],
        'description': 'General macroeconomic discussion including policy, labor markets, and economic conditions'
    },
    'asset_purchases_balance_sheet': {
        'words': ['purchases', 'treasury', 'economic', 'committee', 'stability', 'mandate', 'growth', 'objective'],
        'description': 'Clearly reflects QE operations and balance sheet management'
    },
    'financial_conditions_market_functioning': {
        'words': ['credit', 'reserve', 'market', 'open market', 'stability', 'decision', 'liquidity', 'spending'],
        'description': 'Describes tools and goals for ensuring smooth market function'
    },
    'timing_operations': {
        'words': ['month', 'release', 'timing', 'pace', 'schedule', 'frequency', 'period'],
        'description': 'Focuses on procedural mechanics: when things happen, not what they are'
    },
    'federal_reserve_agencies': {
        'words': ['federal reserve', 'agency', 'security', 'operation', 'balance', 'instruments', 'implementation', 'mandate'],
        'description': 'Emphasizes the institutional and operational backbone of policy execution'
    }
}


def extract_word_frequencies(group):
    """
    Extract word frequencies from JSON files in specified group directories.

    Args:
        group (int): Group number (1 or 2) representing the directories to process.
    """
    group_dirs = {
        1: ['output/Format 1 JSON Lemmatized', 'output/Format 2 JSON Lemmatized'],
        2: ['output/Format 3 JSON Lemmatized', 'output/Format 4 JSON Lemmatized']
    }

    if group not in group_dirs:
        print("Invalid group number. Please choose 1 or 2.")
        return

    word_counter = Counter()

    for directory in group_dirs[group]:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} not found")
            continue

        # Process each JSON file in the directory
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        for json_file in json_files:
            try:
                with open(os.path.join(directory, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Count words in each sentence of each paragraph
                for page in data['pages']:
                    for paragraph in page['paragraphs']:
                        for sentence in paragraph['sentences']:
                            words = sentence.split()
                            word_counter.update(words)

            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")

    word_freq_dict = dict(word_counter)

    print("Word Frequency Dictionary:")
    print(word_freq_dict)

    # Calculate percentiles
    frequencies = np.array(list(word_counter.values()))
    p97_5 = np.percentile(frequencies, 97.5)
    p2_5 = np.percentile(frequencies, 2.5)

    global most_frequent_words, most_rare_words
    most_frequent_words = [word for word, freq in word_counter.items() if freq >= p97_5]
    most_rare_words = [word for word, freq in word_counter.items() if freq <= p2_5]

    print("\nMost Frequent Words (97.5th percentile):")
    for word in most_frequent_words:
        print(f"{word}: {word_counter[word]}")

    print("\nMost Rare Words (2.5th percentile):")
    for word in most_rare_words:
        print(f"{word}: {word_counter[word]}")


def extract_lda_topics(group):
    """
    Extract LDA topics from JSON files in specified group directories.
    Runs multiple times with different numbers of words per topic (4-7).

    Args:
        group (int): Group number (1 or 2) representing the directories to process.
    """
    group_dirs = {
        1: ['output/Format 1 JSON Lemmatized', 'output/Format 2 JSON Lemmatized'],
        2: ['output/Format 3 JSON Lemmatized', 'output/Format 4 JSON Lemmatized']
    }

    if group not in group_dirs:
        print("Invalid group number. Please choose 1 or 2.")
        return

    all_speeches = []

    for directory in group_dirs[group]:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} not found")
            continue

        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        for json_file in json_files:
            try:
                with open(os.path.join(directory, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                speech_text = []
                for page in data['pages']:
                    for paragraph in page['paragraphs']:
                        speech_text.extend(paragraph['sentences'])

                # Join sentences with spaces
                all_speeches.append(' '.join(speech_text))

            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")

    # Create document-term matrix with n-grams
    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words='english', 
        ngram_range=(1, 3),  # unigrams, bigrams, and trigrams
        min_df=2,  
        token_pattern=r'(?u)[a-zA-Z]+(?:-[a-zA-Z]+)*'  
    )
    doc_term_matrix = vectorizer.fit_transform(all_speeches)

    # Apply LDA
    lda_model = LatentDirichletAllocation(
        n_components=10, 
        random_state=42,
        learning_method='batch',
        max_iter=20
    )
    lda_output = lda_model.fit_transform(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()

    # Create output string for topics
    output_text = "Extracted Topics with Different Numbers of Words per Topic:\n\n"

    for num_words in range(4, 8):  # 4 to 7 words
        output_text += f"=== Results with {num_words} words per topic ===\n\n"
        
        for topic_idx, topic in enumerate(lda_model.components_):
            # Get top N words for this topic
            top_phrases_idx = topic.argsort()[:-num_words-1:-1]
            top_phrases = [feature_names[i] for i in top_phrases_idx]

            output_text += f"Topic {topic_idx + 1}:\n"
            output_text += ", ".join(top_phrases) + "\n"
        
        output_text += "\n"  # Add spacing between different num_words results

    # Save topics to a file
    with open('topics.txt', 'w') as f:
        f.write(output_text)

    print("Topics saved to topics.txt")


def extract_seeded_topics(group):
    """
    Extract topics using seeded LDA approach from JSON files in specified group directories.
    Uses seed words to guide the topic discovery process through document augmentation.

    Args:
        group (int): Group number (1 or 2) representing the directories to process.
    """
    # Define directories for each group
    group_dirs = {
        1: ['output/Format 1 JSON Lemmatized', 'output/Format 2 JSON Lemmatized'],
        2: ['output/Format 3 JSON Lemmatized', 'output/Format 4 JSON Lemmatized']
    }

    if group not in group_dirs:
        print("Invalid group number. Please choose 1 or 2.")
        return

    # Collect all speeches
    all_speeches = []

    # Process each directory in the group
    for directory in group_dirs[group]:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} not found")
            continue

        # Process each JSON file in the directory
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        for json_file in json_files:
            try:
                with open(os.path.join(directory, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Combine all sentences from all paragraphs
                speech_text = []
                for page in data['pages']:
                    for paragraph in page['paragraphs']:
                        speech_text.extend(paragraph['sentences'])

                # Join sentences with spaces
                speech = ' '.join(speech_text)
                
                # Augment document with seed words based on presence of topic words
                augmented_speech = speech
                for topic_name, seed_words in SEED_TOPICS.items():
                    # Check if any seed word is in the speech
                    if any(word in speech.lower() for word in seed_words):
                        # Add seed words to reinforce the topic
                        augmented_speech += ' ' + ' '.join(seed_words * 3)
                
                all_speeches.append(augmented_speech)

            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")

    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
        min_df=2,
        token_pattern=r'(?u)[a-zA-Z]+(?:-[a-zA-Z]+)*'
    )
    doc_term_matrix = vectorizer.fit_transform(all_speeches)
    
    # Get vocabulary mapping
    vocab = vectorizer.get_feature_names_out()
    
    # Initialize and fit LDA model
    model = lda.LDA(
        n_topics=len(SEED_TOPICS),
        n_iter=500,
        random_state=42
    )
    
    # Fit the model
    model.fit(doc_term_matrix.toarray())

    # Create output string for topics
    output_text = "Extracted Topics (Seeded LDA) with Different Numbers of Words per Topic:\n\n"
    output_text += "Seed topics used:\n"
    for topic_name, seed_words in SEED_TOPICS.items():
        output_text += f"{topic_name}: {', '.join(seed_words)}\n"
    output_text += "\n"

    # Run for different numbers of words per topic
    for num_words in range(4, 8):  # 4 to 7 words
        output_text += f"=== Results with {num_words} words per topic ===\n\n"
        
        for topic_idx, topic in enumerate(model.components_):
            # Get top N words for this topic
            top_words_idx = np.argsort(topic)[:-num_words-1:-1]
            top_words = [vocab[i] for i in top_words_idx]
            
            topic_name = list(SEED_TOPICS.keys())[topic_idx]
            output_text += f"Topic {topic_idx + 1} ({topic_name}):\n"
            output_text += ", ".join(top_words) + "\n"
        
        output_text += "\n"  # Add spacing between different num_words results

    # Save topics to a file
    with open('topics_seeded.txt', 'w') as f:
        f.write(output_text)

    print("Seeded Topics saved to topics_seeded.txt")


def train_lda_model(group):
    """
    Train LDA model on full speeches and return the trained model and vectorizer.
    
    Args:
        group (int): Group number (1 or 2) representing the directories to process.
    
    Returns:
        tuple: (vectorizer, lda_model, all_speeches, filenames)
    """
    group_dirs = {
        1: ['output/Format 1 JSON Lemmatized', 'output/Format 2 JSON Lemmatized'],
        2: ['output/Format 3 JSON Lemmatized', 'output/Format 4 JSON Lemmatized']
    }

    if group not in group_dirs:
        print("Invalid group number. Please choose 1 or 2.")
        return None, None, None, None

    # Collect speeches and filenames
    all_speeches = []
    filenames = []
    
    # Process each directory
    for directory in group_dirs[group]:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} not found")
            continue

        for json_file in [f for f in os.listdir(directory) if f.endswith('_lemmatized.json')]:
            try:
                with open(os.path.join(directory, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Combine sentences
                speech_text = []
                for page in data['pages']:
                    for paragraph in page['paragraphs']:
                        speech_text.extend(paragraph['sentences'])

                # Augment document with topic keywords
                speech = ' '.join(speech_text)
                augmented_speech = speech
                for topic, info in TOPIC_KEYWORDS.items():
                    if any(word in speech.lower() for word in info['words']):
                        # Add keywords to reinforce topics
                        augmented_speech += ' ' + ' '.join(info['words'] * 3)

                all_speeches.append(augmented_speech)
                filenames.append(json_file)

            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")

    # Create and fit vectorizer
    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=2,
        token_pattern=r'(?u)[a-zA-Z]+(?:-[a-zA-Z]+)*'
    )
    doc_term_matrix = vectorizer.fit_transform(all_speeches)

    # Train LDA model
    lda_model = LatentDirichletAllocation(
        n_components=len(TOPIC_KEYWORDS),
        random_state=42,
        learning_method='batch',
        max_iter=20
    )
    lda_model.fit(doc_term_matrix)

    return vectorizer, lda_model, all_speeches, filenames


def create_topic_probabilities_csv(group):
    """
    Create a CSV file containing topic probabilities for each speech using guided LDA.
    Uses predefined topic keywords to guide the topic modeling process.
    
    Args:
        group (int): Group number (1 or 2) representing the directories to process.
    """
    # Train model once
    vectorizer, lda_model, all_speeches, filenames = train_lda_model(group)
    if vectorizer is None:
        return

    # Transform documents to get topic probabilities
    doc_term_matrix = vectorizer.transform(all_speeches)
    topic_probabilities = lda_model.transform(doc_term_matrix)

    # Create DataFrame with topic names
    df = pd.DataFrame(topic_probabilities, columns=TOPIC_KEYWORDS.keys())
    df.insert(0, 'filename', filenames)
    
    # Save to CSV
    output_file = f'topic_probabilities_group_{group}.csv'
    df.to_csv(output_file, index=False)
    print(f"Topic probabilities saved to {output_file}")
    
    return vectorizer, lda_model  # Return for use in extract_topic_sentences


def extract_topic_sentences(group, vectorizer=None, lda_model=None, output_folder='output/extracted_topics_jsons'):
    """
    Extract sentences from speeches and assign them to topics using guided LDA.
    For each speech, output a JSON file in the specified folder with topic-sorted sentences.
    Args:
        group (int): Group number (1 or 2) representing the directories to process.
        vectorizer: Optional pre-trained CountVectorizer
        lda_model: Optional pre-trained LDA model
        output_folder (str): Folder to save the JSON files
    """
    import os
    import json
    # If no pre-trained models provided, train them
    if vectorizer is None or lda_model is None:
        vectorizer, lda_model, _, _ = train_lda_model(group)
        if vectorizer is None:
            return

    group_dirs = {
        1: ['output/Format 1 JSON Lemmatized', 'output/Format 2 JSON Lemmatized'],
        2: ['output/Format 3 JSON Lemmatized', 'output/Format 4 JSON Lemmatized']
    }

    if group not in group_dirs:
        print("Invalid group number. Please choose 1 or 2.")
        return

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each directory
    for directory in group_dirs[group]:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} not found")
            continue

        for json_file in [f for f in os.listdir(directory) if f.endswith('_lemmatized.json')]:
            try:
                with open(os.path.join(directory, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Collect sentences
                sentences = []
                for page in data['pages']:
                    for paragraph in page['paragraphs']:
                        sentences.extend(paragraph['sentences'])

                # Augment sentences with topic keywords
                augmented_sentences = []
                for sentence in sentences:
                    augmented = sentence
                    for topic, info in TOPIC_KEYWORDS.items():
                        if any(word in sentence.lower() for word in info['words']):
                            augmented += ' ' + ' '.join(info['words'] * 2)
                    augmented_sentences.append(augmented)

                # Use the pre-trained vectorizer and model
                sentence_matrix = vectorizer.transform(augmented_sentences)
                sentence_topics = lda_model.transform(sentence_matrix)

                # Store sentences by topic
                topic_sentences = {topic: [] for topic in TOPIC_KEYWORDS.keys()}
                
                # Assign each sentence to its most probable topic
                for sentence, topic_dist in zip(sentences, sentence_topics):
                    best_topic_idx = topic_dist.argmax()
                    topic_name = list(TOPIC_KEYWORDS.keys())[best_topic_idx]
                    topic_sentences[topic_name].append(sentence)

                # Prepare JSON output
                output_json = {
                    "filename": json_file,
                    "topics": topic_sentences
                }
                out_path = os.path.join(output_folder, f"{json_file.replace('_lemmatized.json', '')}_extracted_topics.json")
                with open(out_path, 'w', encoding='utf-8') as out_f:
                    json.dump(output_json, out_f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")


def create_sentiment_weights():
    """
    Create a dictionary of sentiment word weights using 1/count (raw frequency) across all lemmatized JSON files.
    The more rare a word, the higher its weight (1/count). If a word does not appear, its weight is 0.
    Returns:
        dict: Dictionary where keys are sentiment words and values are their 1/count weights
    """
    import os
    import json
    from collections import Counter

    sentiment_words = GOOD_PHRASES + BAD_PHRASES

    documents = []
    format_dirs = [
        'output/Format 1 JSON Lemmatized',
        'output/Format 2 JSON Lemmatized',
        'output/Format 3 JSON Lemmatized',
        'output/Format 4 JSON Lemmatized'
    ]
    for format_dir in format_dirs:
        if not os.path.exists(format_dir):
            print(f"Warning: Directory {format_dir} not found")
            continue
        json_files = [f for f in os.listdir(format_dir) if f.endswith('.json')]
        for json_file in json_files:
            try:
                with open(os.path.join(format_dir, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                speech_text = []
                for page in data['pages']:
                    for paragraph in page['paragraphs']:
                        speech_text.extend(paragraph['sentences'])
                documents.append(' '.join(speech_text))
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")

    # Count all words in the corpus
    all_words = ' '.join(documents).lower().split()
    word_counts = Counter(all_words)

    # Assign 1/count as weight (0 if not present)
    word_weights = {}
    for word in sentiment_words:
        count = word_counts.get(word, 0)
        word_weights[word] = 1.0 / count if count > 0 else 0.0

    return word_weights


def compute_topic_sentiments(jsons_folder, sentiment_weights, output_csv='topic_sentiment_scores.csv'):
    """
    Compute sentiment scores for each topic in each speech using TF-IDF weights from per-speech topic JSONs.
    Args:
        jsons_folder (str): Path to the folder containing per-speech topic JSONs
        sentiment_weights (dict): Dictionary of sentiment word weights from create_sentiment_weights()
        output_csv (str): Path to save the output CSV
    Returns:
        pandas.DataFrame: DataFrame with sentiment scores for each topic in each speech
    """
    import os
    import json
    import pandas as pd

    all_topics = [
        'macroeconomic_commentary',
        'asset_purchases_balance_sheet',
        'financial_conditions_market_functioning',
        'timing_operations',
        'federal_reserve_agencies'
    ]

    results = []
    for fname in os.listdir(jsons_folder):
        if not fname.endswith('_extracted_topics.json'):
            continue
        fpath = os.path.join(jsons_folder, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        row = {'filename': data['filename']}
        topics = data['topics']
        for topic in all_topics:
            score = 0.0
            for sentence in topics.get(topic, []):
                words = sentence.lower().split()
                for word in words:
                    if word in GOOD_PHRASES and word in sentiment_weights:
                        score += sentiment_weights[word]
                    elif word in BAD_PHRASES and word in sentiment_weights:
                        score -= sentiment_weights[word]
            row[f'topic_sentiment_score_{topic}'] = score
        results.append(row)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return df


def print_sentiment_word_stats():
    """
    Print each sentiment word, its 1/count weight, and its count in the corpus, sorted by descending count.
    """
    from collections import Counter
    sentiment_weights = create_sentiment_weights()
    sentiment_words = GOOD_PHRASES + BAD_PHRASES

    documents = []
    format_dirs = [
        'output/Format 1 JSON Lemmatized',
        'output/Format 2 JSON Lemmatized',
        'output/Format 3 JSON Lemmatized',
        'output/Format 4 JSON Lemmatized'
    ]
    for format_dir in format_dirs:
        if not os.path.exists(format_dir):
            continue
        json_files = [f for f in os.listdir(format_dir) if f.endswith('.json')]
        for json_file in json_files:
            try:
                with open(os.path.join(format_dir, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                speech_text = []
                for page in data['pages']:
                    for paragraph in page['paragraphs']:
                        speech_text.extend(paragraph['sentences'])
                documents.append(' '.join(speech_text))
            except Exception:
                continue
    all_words = ' '.join(documents).lower().split()
    word_counts = Counter(all_words)
    stats = []
    for word in sentiment_words:
        weight = sentiment_weights.get(word, 0.0)
        count = word_counts.get(word, 0)
        stats.append((word, weight, count))
    stats.sort(key=lambda x: x[2], reverse=True)
    print(f"{'Word':<20} {'1/Count Weight':<20} {'Count in Corpus':<15}")
    print('-'*60)
    for word, weight, count in stats:
        print(f"{word:<20} {weight:<20.6f} {count:<15}")


def merge_sentiment_csvs(csv1_path, csv2_path, output_csv='merged_sentiment_scores.csv'):
    """
    Merge two CSV files containing sentiment scores, using filename as the key.
    Both CSVs must have a 'filename' column. All columns from both CSVs will be preserved.
    If there are duplicate column names (other than 'filename'), they will be suffixed with _x and _y.
    
    Args:
        csv1_path (str): Path to first CSV file
        csv2_path (str): Path to second CSV file
        output_csv (str): Path for the output merged CSV file
    
    Returns:
        pandas.DataFrame: The merged DataFrame that was saved to output_csv
    """
    import pandas as pd
    
    # Read both CSVs
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    
    # Ensure both have filename column
    if 'filename' not in df1.columns or 'filename' not in df2.columns:
        raise ValueError("Both CSV files must have a 'filename' column")
    
    # Merge the dataframes on filename
    merged_df = pd.merge(df1, df2, on='filename', how='outer')
    
    # Save to CSV
    merged_df.to_csv(output_csv, index=False)
    print(f"Merged CSV saved to {output_csv}")
    
    return merged_df


def add_word_counts_to_csv(input_csv, group, output_csv=None):
    """
    Add word count column to a CSV file by counting words from speeches.
    For Group 1: Counts words from JSON files in /output/Format 1 JSON and /output/Format 2 JSON
    For Group 2: Counts words from PDF files in the specified directories
    
    Args:
        input_csv (str): Path to the input CSV file containing speech filenames
        group (int): Group number (1 or 2) to determine which directories to process
        output_csv (str, optional): Path for the output CSV. If None, appends '_with_wordcounts' to input filename
    
    Returns:
        pandas.DataFrame: The DataFrame with added word count column
    """
    import pandas as pd
    import os
    import json
    import re
    
    # Read input CSV
    df = pd.read_csv(input_csv)
    print(f"Read input CSV with {len(df)} rows")
    print("Sample filenames from CSV:", df['filename'].head())
    
    # Create word count dictionary
    word_counts = {}
    
    if group == 1:
        # Define JSON directories for Group 1
        json_dirs = [
            './output/Format 1 JSON',
            './output/Format 2 JSON'
        ]
        
        # Process each directory
        for directory in json_dirs:
            print(f"\nProcessing directory: {directory}")
            if not os.path.exists(directory):
                print(f"Warning: Directory {directory} not found")
                continue
                
            # Process each JSON file
            json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
            print(f"Found {len(json_files)} JSON files in {directory}")
            print("Sample JSON filenames:", json_files[:5])
            
            for json_file in json_files:
                try:
                    json_path = os.path.join(directory, json_file)
                    print(f"Processing {json_file}")
                    
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Count words only from sentences arrays
                    word_count = 0
                    for page in data.get('pages', []):
                        for paragraph in page.get('paragraphs', []):
                            for sentence in paragraph.get('sentences', []):
                                # Clean and count words in each sentence
                                words = re.sub(r'[^\w\s]', ' ', sentence).split()
                                word_count += len(words)
                    
                    print(f"Word count for {json_file}: {word_count}")
                    # Store with the exact JSON filename
                    word_counts[json_file] = word_count
                    
                except Exception as e:
                    print(f"Error processing {json_file}: {str(e)}")
        
        print("\nWord counts dictionary sample:", dict(list(word_counts.items())[:5]))
        
        # Create a function to convert CSV filenames to JSON filenames
        def csv_to_json_filename(csv_filename):
            # Remove _lemmatized.json if present
            base_name = csv_filename.replace('_lemmatized.json', '')
            # Remove .json if present
            base_name = base_name.replace('.json', '')
            # Add .json back
            return base_name + '.json'
        
        # Add word count column to DataFrame
        # Map using the converted filenames
        df['word_count'] = df['filename'].apply(lambda x: word_counts.get(csv_to_json_filename(x), 0))
        
        print("\nSample of filename mapping results:")
        sample_df = df[['filename', 'word_count']].head()
        print(sample_df)
        
    else:  # Group 2 - keep existing PDF-based logic
        from PyPDF2 import PdfReader
        
        # Define directories for Group 2
        group_dirs = {
            2: [
                './data/FOMC_PDFs/Format 3',
                './data/FOMC_PDFs/Format 4'
            ]
        }
        
        # Process each directory
        for directory in group_dirs[group]:
            print(f"\nProcessing directory: {directory}")
            if not os.path.exists(directory):
                print(f"Warning: Directory {directory} not found")
                continue
                
            # Process each PDF file
            pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
            print(f"Found {len(pdf_files)} PDF files in {directory}")
            
            for pdf_file in pdf_files:
                try:
                    pdf_path = os.path.join(directory, pdf_file)
                    print(f"Processing {pdf_file}")
                    
                    reader = PdfReader(pdf_path)
                    
                    # Extract text from all pages
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    
                    # Clean text and count words
                    text = re.sub(r'[^\w\s]', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    word_count = len(text.split())
                    print(f"Word count for {pdf_file}: {word_count}")
                    
                    word_counts[pdf_file] = word_count
                    
                except Exception as e:
                    print(f"Error processing {pdf_file}: {str(e)}")
        
        # Create mapping from JSON to PDF filenames for Group 2
        def json_to_pdf_filename(json_filename):
            base_name = json_filename.replace('_lemmatized.json', '')
            base_name = base_name.replace('.json', '')
            return base_name + '.pdf'
        
        # Add word count column to DataFrame
        df['word_count'] = df['filename'].apply(lambda x: word_counts.get(json_to_pdf_filename(x), 0))
    
    print("\nWord count statistics:")
    print(f"Number of files with word counts: {df['word_count'].gt(0).sum()}")
    print(f"Number of files without word counts: {df['word_count'].eq(0).sum()}")
    print("\nSample of rows with word counts:")
    print(df[df['word_count'].gt(0)][['filename', 'word_count']].head())
    
    # Set output filename if not provided
    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_with_wordcounts{ext}"
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nCSV with word counts saved to {output_csv}")
    
    return df


def add_overall_sentiment(input_csv, group, output_csv=None):
    """
    Add overall sentiment score to a CSV by counting occurrences of good and bad phrases in lemmatized speeches.
    Each good phrase adds +1 to the score, each bad phrase subtracts -1.
    
    Args:
        input_csv (str): Path to the input CSV file containing speech filenames
        group (int): Group number (1 or 2) to determine which directories to process
        output_csv (str, optional): Path for the output CSV. If None, appends '_with_sentiment' to input filename
    
    Returns:
        pandas.DataFrame: The DataFrame with added overall_sentiment column
    """
    import pandas as pd
    import os
    import json
    
    # Define directories for each group
    group_dirs = {
        1: ['output/Format 1 JSON Lemmatized', 'output/Format 2 JSON Lemmatized'],
        2: ['output/Format 3 JSON Lemmatized', 'output/Format 4 JSON Lemmatized']
    }
    
    if group not in group_dirs:
        raise ValueError("Group must be 1 or 2")
    
    # Read input CSV
    df = pd.read_csv(input_csv)
    print(f"Read input CSV with {len(df)} rows")
    
    # Create sentiment scores dictionary
    sentiment_scores = {}
    
    # Process each directory in the group
    for directory in group_dirs[group]:
        print(f"\nProcessing directory: {directory}")
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} not found")
            continue
            
        # Process each JSON file
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files in {directory}")
        
        for json_file in json_files:
            try:
                with open(os.path.join(directory, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Combine all sentences
                all_text = []
                for page in data['pages']:
                    for paragraph in page['paragraphs']:
                        all_text.extend(paragraph['sentences'])
                text = ' '.join(all_text).lower()
                
                # Count good and bad phrases
                good_count = sum(text.count(phrase) for phrase in GOOD_PHRASES)
                bad_count = sum(text.count(phrase) for phrase in BAD_PHRASES)
                
                # Calculate overall sentiment score
                score = good_count - bad_count
                print(f"File: {json_file}, Good phrases: {good_count}, Bad phrases: {bad_count}, Score: {score}")
                
                # Store score
                sentiment_scores[json_file] = score
                
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
    
    # Add sentiment score column to DataFrame
    df['overall_sentiment'] = df['filename'].map(sentiment_scores)
    
    print("\nSentiment score statistics:")
    print(f"Number of files with sentiment scores: {df['overall_sentiment'].notna().sum()}")
    print(f"Number of files without sentiment scores: {df['overall_sentiment'].isna().sum()}")
    print("\nSample of rows with sentiment scores:")
    print(df[df['overall_sentiment'].notna()][['filename', 'overall_sentiment']].head())
    
    # Set output filename if not provided
    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_with_sentiment{ext}"
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nCSV with sentiment scores saved to {output_csv}")
    
    return df


def convert_sentiment_to_categories(input_csv, output_csv=None):
    """
    Convert numerical sentiment scores to categorical values (very poor, poor, neutral, good, very good).
    Uses quantiles to determine thresholds for each category. Handles cases where thresholds might be identical.
    
    Args:
        input_csv (str): Path to the input CSV file containing topic sentiment scores
        output_csv (str, optional): Path for the output CSV. If None, appends '_categorical' to input filename
    
    Returns:
        pandas.DataFrame: The DataFrame with categorical sentiment columns
    """
    import pandas as pd
    import numpy as np
    
    # Read input CSV
    df = pd.read_csv(input_csv)
    
    # Get all columns that start with 'topic_sentiment_score'
    sentiment_cols = [col for col in df.columns if col.startswith('topic_sentiment_score_')]
    
    if not sentiment_cols:
        raise ValueError("No columns starting with 'topic_sentiment_score_' found in the CSV")
    
    print(f"Found {len(sentiment_cols)} sentiment score columns")
    
    # Create a copy of the DataFrame for categorical values
    df_cat = df.copy()
    
    # Process each sentiment column
    for col in sentiment_cols:
        try:
            # Calculate thresholds using quantiles
            thresholds = {
                'very_poor': df[col].quantile(0.2),
                'poor': df[col].quantile(0.4),
                'neutral': df[col].quantile(0.6),
                'good': df[col].quantile(0.8)
            }
            
            # Create bins, ensuring they're unique
            bins = [-np.inf]
            labels = []
            
            # Add thresholds in order, skipping duplicates
            current_threshold = -np.inf
            for threshold_name, threshold_value in thresholds.items():
                if threshold_value > current_threshold:
                    bins.append(threshold_value)
                    current_threshold = threshold_value
                    labels.append(threshold_name.replace('_', ' '))
            
            # Add final bin
            bins.append(np.inf)
            
            # Adjust labels based on number of unique thresholds
            if len(labels) == 1:
                final_labels = ['poor', 'good']
            elif len(labels) == 2:
                final_labels = ['poor', 'neutral', 'good']
            elif len(labels) == 3:
                final_labels = ['poor', 'neutral', 'good', 'very good']
            else:
                final_labels = ['very poor', 'poor', 'neutral', 'good', 'very good']
            
            print(f"\nThresholds for {col}:")
            for i in range(len(bins)-1):
                if i == 0:
                    print(f"{final_labels[i]}: < {bins[i+1]:.4f}")
                elif i == len(bins)-2:
                    print(f"{final_labels[i]}: > {bins[i]:.4f}")
                else:
                    print(f"{final_labels[i]}: {bins[i]:.4f} to {bins[i+1]:.4f}")
            
            # Convert to categorical
            df_cat[col + '_category'] = pd.cut(
                df[col],
                bins=bins,
                labels=final_labels,
                duplicates='drop'
            )
            
            print(f"\nDistribution of categories for {col}:")
            print(df_cat[col + '_category'].value_counts().sort_index())
            
        except Exception as e:
            print(f"Error processing column {col}: {str(e)}")
            print("Skipping this column...")
            continue
    
    # Set output filename if not provided
    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_categorical{ext}"
    
    # Save to CSV
    df_cat.to_csv(output_csv, index=False)
    print(f"\nCSV with categorical sentiment scores saved to {output_csv}")
    
    return df_cat


def add_keyword_columns_to_dataframe(input_csv, group, output_csv=None):
    """
    Add columns tracking the frequency of key phrases in each speech.
    Each column represents the count of a specific phrase in the speech.
    
    Args:
        input_csv (str): Path to the input CSV file containing speech filenames
        group (int): Group number (1 or 2) to determine which directories to process
        output_csv (str, optional): Path for the output CSV. If None, appends '_with_keywords' to input filename
    
    Returns:
        pandas.DataFrame: The DataFrame with added keyword frequency columns
    """
    import pandas as pd
    import os
    import json
    
    # Define key phrases to track
    KEY_PHRASES = [
        'monetary policy',
        'federal reserve',
        'inflation',
        'federal fund rate',
        'labor market',
        'target range',
        'stance monetary policy',
        'maximum employment price',
        'open market committee',
        'holding treasury security'
    ]
    
    # Define directories for each group
    group_dirs = {
        1: ['output/Format 1 JSON Lemmatized', 'output/Format 2 JSON Lemmatized'],
        2: ['output/Format 3 JSON Lemmatized', 'output/Format 4 JSON Lemmatized']
    }
    
    if group not in group_dirs:
        raise ValueError("Group must be 1 or 2")
    
    # Read input CSV
    df = pd.read_csv(input_csv)
    print(f"Read input CSV with {len(df)} rows")
    
    # Initialize keyword frequency dictionary
    keyword_freqs = {phrase: {} for phrase in KEY_PHRASES}
    
    # Process each directory in the group
    for directory in group_dirs[group]:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} not found")
            continue
            
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        for json_file in json_files:
            try:
                with open(os.path.join(directory, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Combine all sentences
                all_text = []
                for page in data['pages']:
                    for paragraph in page['paragraphs']:
                        all_text.extend(paragraph['sentences'])
                text = ' '.join(all_text).lower()
                
                # Count occurrences of each key phrase
                for phrase in KEY_PHRASES:
                    # Replace spaces with underscores for column name
                    col_name = phrase.replace(' ', '_')
                    # Count occurrences
                    count = text.count(phrase.lower())
                    keyword_freqs[phrase][json_file] = count

            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")

    # Add columns to DataFrame
    for phrase in KEY_PHRASES:
        col_name = phrase.replace(' ', '_')
        df[col_name] = df['filename'].map(keyword_freqs[phrase])
    
    print("\nKeyword frequency statistics:")
    for phrase in KEY_PHRASES:
        col_name = phrase.replace(' ', '_')
        print(f"\n{phrase}:")
        print(f"Total occurrences: {df[col_name].sum()}")
        print(f"Average per speech: {df[col_name].mean():.2f}")
        print(f"Maximum in a speech: {df[col_name].max()}")
        print(f"Speeches containing phrase: {df[col_name].gt(0).sum()}")
    
    # Set output filename if not provided
    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_with_keywords{ext}"
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nCSV with keyword frequencies saved to {output_csv}")
    
    return df


def join_csvs_on_filename(csv_paths, output_csv='joined_data.csv'):
    """
    Join multiple CSVs on the 'filename' column, normalizing filenames by removing '_lemmatized' suffix.
    
    Args:
        csv_paths (list): List of paths to CSV files to join
        output_csv (str): Path for the output joined CSV file
    
    Returns:
        pandas.DataFrame: The joined DataFrame
    """
    import pandas as pd
    import os
    
    if not csv_paths:
        raise ValueError("No CSV paths provided")
    
    # Read the first CSV
    print(f"Reading {csv_paths[0]}")
    df = pd.read_csv(csv_paths[0])
    
    # Normalize filenames in the first DataFrame
    df['filename'] = df['filename'].str.replace('_lemmatized.json', '.json')
    
    # Read and join each subsequent CSV
    for i, csv_path in enumerate(csv_paths[1:], 1):
        print(f"Reading and joining {csv_path}")
        try:
            # Read the next CSV
            next_df = pd.read_csv(csv_path)
            
            # Normalize filenames if needed
            if 'filename' in next_df.columns:
                next_df['filename'] = next_df['filename'].str.replace('_lemmatized.json', '.json')
            
            # Join with the main DataFrame
            df = pd.merge(df, next_df, on='filename', how='outer')
            
            print(f"Successfully joined {csv_path}")
            print(f"Current shape: {df.shape}")
            
        except Exception as e:
            print(f"Error joining {csv_path}: {str(e)}")
            print("Continuing with next CSV...")
    
    print("\nFinal DataFrame Statistics:")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nJoined CSV saved to {output_csv}")
    
    return df


def adjust_sentiment_categories(input_csv, output_csv=None):
    """
    Adjust categorical sentiment columns to change 'poor' to 'neutral' when the corresponding
    numerical sentiment score is 0. Preserves all other categorical values.
    
    Args:
        input_csv (str): Path to the input CSV file containing both numerical and categorical sentiment columns
        output_csv (str, optional): Path for the output CSV. If None, appends '_adjusted' to input filename
    
    Returns:
        pandas.DataFrame: The DataFrame with adjusted categorical sentiment columns
    """
    import pandas as pd
    import os
    
    # Read input CSV
    df = pd.read_csv(input_csv)
    print(f"Read input CSV with {len(df)} rows")
    
    # Get numerical sentiment score columns
    num_cols = [col for col in df.columns if col.startswith('topic_sentiment_score_') and not col.endswith('_category')]
    
    # Get corresponding categorical columns
    cat_cols = [col for col in df.columns if col.endswith('_category')]
    
    print(f"Found {len(num_cols)} numerical sentiment columns")
    print(f"Found {len(cat_cols)} categorical sentiment columns")
    
    # Process each pair of numerical and categorical columns
    for num_col in num_cols:
        cat_col = num_col + '_category'
        if cat_col not in df.columns:
            print(f"Warning: No categorical column found for {num_col}")
            continue
            
        # Count initial distribution
        print(f"\nInitial distribution for {cat_col}:")
        print(df[cat_col].value_counts())
        
        # Create mask for rows where numerical score is 0 and categorical is 'poor'
        mask = (df[num_col] == 0) & (df[cat_col] == 'poor')
        
        # Count how many values will be changed
        num_changes = mask.sum()
        print(f"Changing {num_changes} 'poor' values to 'neutral' for {cat_col}")
        
        # Make the changes
        df.loc[mask, cat_col] = 'neutral'
        
        print(f"Final distribution for {cat_col}:")
        print(df[cat_col].value_counts())
    
    # Set output filename if not provided
    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_adjusted{ext}"
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nAdjusted CSV saved to {output_csv}")
    
    return df


# Example usage for Group 2 speeches pipeline
if __name__ == "__main__":
    sp500_df = pd.read_csv('Group 2 with SP500.csv')
    bonds_df = pd.read_csv('group2_bonds_data.csv')
    
    #filename as index for both dataframes
    sp500_df.set_index('filename', inplace=True)
    bonds_df.set_index('filename', inplace=True)
    
    # Get overlapping columns (excluding 'filename' since it's now the index)
    overlapping_cols = bonds_df.columns.intersection(sp500_df.columns)
    
    # Update SP500 dataframe with new values from bonds dataframe
    for col in overlapping_cols:
        sp500_df[col] = bonds_df[col]
    
    # Reset index to get filename back as a column
    sp500_df.reset_index(inplace=True)
    
    sp500_df.to_csv('Group 2 with SP500 and Bonds.csv', index=False)
    print("Merged CSV saved as 'Group 2 with SP500 and Bonds.csv'")
    
    print(f"\nNumber of rows in original SP500 data: {len(sp500_df)}")
    print(f"Number of columns updated: {len(overlapping_cols)}")
    print("\nUpdated columns:")
    for col in overlapping_cols:
        print(f"- {col}")

