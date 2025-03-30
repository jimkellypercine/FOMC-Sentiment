import json
import re
import string
import nltk
import spacy
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Ensure necessary NLP models are downloaded
nltk.download("stopwords")
nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")

# Load stopwords
stop_words = set(stopwords.words("english"))

def clean_text(text, remove_stopwords=True, remove_numbers=True, remove_special_chars=True, lowercase=True, lemmatization=True):
    """Cleans a given text based on selected options."""
    if not isinstance(text, str) or text.strip() == "":
        return text  # Preserve structure for paragraph markers

    # Convert to lowercase
    if lowercase:
        text = text.lower()

    # Remove numbers
    if remove_numbers:
        text = re.sub(r"\d+", "", text)

    # Remove special characters except periods (for structure)
    if remove_special_chars:
        text = re.sub(r"[^\w\s.]", "", text)

    # Tokenize and process text
    doc = nlp(text)
    processed_tokens = []

    for token in doc:
        word = token.text

        # Apply stopword removal
        if remove_stopwords and word in stop_words:
            continue
        
        # Apply lemmatization
        if lemmatization:
            word = token.lemma_

        processed_tokens.append(word)

    # Reconstruct cleaned sentence
    return " ".join(processed_tokens)

def process_json_file(input_file, output_file, remove_stopwords=True, remove_numbers=True, remove_special_chars=True, lowercase=True, lemmatization=True):
    """Reads a JSON file, processes text with selected cleaning steps, and writes cleaned output to a new JSON file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize cleaned data with metadata
    cleaned_data = {
        "format": data["format"],
        "date": data["date"],
        "pages": []
    }

    # Process each page
    for page in data["pages"]:
        cleaned_page = {
            "page": page["page"],
            "paragraphs": []
        }
        
        # Process each paragraph in the page
        for paragraph in page["paragraphs"]:
            cleaned_paragraph = {
                "paragraph": paragraph["paragraph"],
                "sentences": [
                    clean_text(
                        sentence,
                        remove_stopwords=remove_stopwords,
                        remove_numbers=remove_numbers,
                        remove_special_chars=remove_special_chars,
                        lowercase=lowercase,
                        lemmatization=lemmatization
                    ) for sentence in paragraph["sentences"]
                ]
            }
            cleaned_page["paragraphs"].append(cleaned_paragraph)
        
        cleaned_data["pages"].append(cleaned_page)

    # Save the cleaned data to a new JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4)

    print(f"Processed file saved as: {output_file}")

def process_all_files():
    """Process all JSON files in the Speech Paragraphs folder and save to Tokenized Speeches folder."""
    # Define input and output directories
    input_dir = "Speech Paragraphs Format 4"
    output_dir = "Tokenized Speeches Format 4"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all JSON files from input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # Process each file
    for json_file in json_files:
        # Construct input and output file paths
        input_path = os.path.join(input_dir, json_file)
        output_filename = os.path.splitext(json_file)[0] + '_tokenized.json'
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing {json_file}...")
        try:
            process_json_file(
                input_path,
                output_path,
                remove_stopwords=True,
                remove_numbers=False,
                remove_special_chars=True,
                lowercase=True,
                lemmatization=True
            )
            print(f"Successfully processed {json_file}")
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

def count_words():
    """Count words in each speech JSON file and save results to a CSV."""
    # Define input directory
    input_dir = "Speech Paragraphs Format 4"
    
    # Initialize lists to store results
    filenames = []
    word_counts = []
    
    # Get all JSON files from input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # Process each file
    for json_file in json_files:
        input_path = os.path.join(input_dir, json_file)
        
        print(f"Counting words in {json_file}...")
        try:
            # Read JSON file
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Initialize word count for this file
            total_words = 0
            
            # Count words in each sentence of each paragraph
            for page in data["pages"]:
                for paragraph in page["paragraphs"]:
                    for sentence in paragraph["sentences"]:
                        # Split sentence into words and count non-empty words
                        words = [word for word in sentence.split() if word.strip()]
                        total_words += len(words)
            
            # Store results
            filenames.append(json_file)
            word_counts.append(total_words)
            print(f"Successfully counted words in {json_file}: {total_words} words")
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'filename': filenames,
        'word_count': word_counts
    })
    
    # Save to CSV
    output_file = 'speech_word_counts.csv'
    df.to_csv(output_file, index=False)
    print(f"\nWord counts saved to {output_file}")

def plot_word_distribution(csv_file='speech_word_counts.csv'):
    """Plot the distribution of word counts across speeches with mean and median lines."""
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Set the style for better visualization
    plt.style.use('seaborn-v0_8')  # Using a valid matplotlib style
    
    # Create figure with larger size
    plt.figure(figsize=(12, 6))
    
    # Create histogram with KDE
    sns.histplot(data=df['word_count'], bins=30, kde=True, color='skyblue', alpha=0.6)
    
    # Add mean and median lines
    mean_count = df['word_count'].mean()
    median_count = df['word_count'].median()
    
    plt.axvline(x=mean_count, color='red', linestyle='--', label=f'Mean: {mean_count:.0f}')
    plt.axvline(x=median_count, color='green', linestyle='--', label=f'Median: {median_count:.0f}')
    
    # Customize the plot
    plt.title('Distribution of Word Counts Across Speeches', fontsize=14, pad=20)
    plt.xlabel('Word Count', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    
    # Add text box with statistics
    stats_text = f'Total Speeches: {len(df)}\n'
    stats_text += f'Mean: {mean_count:.0f} words\n'
    stats_text += f'Median: {median_count:.0f} words\n'
    stats_text += f'Std Dev: {df["word_count"].std():.0f} words'
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    plt.savefig('word_count_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution plot saved as 'word_count_distribution.png'")

def find_shortest_sentence():
    """Find the shortest sentence in each speech and track which paragraph it belongs to."""
    # Define input directory
    input_dir = "Speech Paragraphs Format 3"
    
    # Initialize lists to store results
    filenames = []
    paragraph_numbers = []
    shortest_sentences = []
    word_counts = []
    
    # Get all JSON files from input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # Process each file
    for json_file in json_files:
        input_path = os.path.join(input_dir, json_file)
        
        print(f"Finding shortest sentence in {json_file}...")
        try:
            # Read JSON file
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Initialize variables to track shortest sentence
            min_sentence_length = float('inf')
            shortest_sentence = ""
            paragraph_with_shortest = 0
            
            # Process each page
            for page in data["pages"]:
                # Process each paragraph
                for paragraph in page["paragraphs"]:
                    paragraph_num = paragraph["paragraph"]
                    
                    # Process each sentence in the paragraph
                    for sentence in paragraph["sentences"]:
                        # Count words in sentence (excluding empty strings)
                        words = [word for word in sentence.split() if word.strip()]
                        sentence_length = len(words)
                        
                        # Only consider sentences with at least 4 words
                        if sentence_length >= 4 and sentence_length < min_sentence_length:
                            min_sentence_length = sentence_length
                            shortest_sentence = sentence
                            paragraph_with_shortest = paragraph_num
            
            # Store results
            filenames.append(json_file)
            paragraph_numbers.append(paragraph_with_shortest)
            shortest_sentences.append(shortest_sentence)
            word_counts.append(min_sentence_length)
            
            print(f"Successfully processed {json_file}")
            print(f"Shortest sentence (min 4 words): {min_sentence_length} words in paragraph {paragraph_with_shortest}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'filename': filenames,
        'paragraph_number': paragraph_numbers,
        'shortest_sentence': shortest_sentences,
        'word_count': word_counts
    })
    
    # Save to CSV
    output_file = 'shortest_sentences.csv'
    df.to_csv(output_file, index=False)
    print(f"\nShortest sentences saved to {output_file}")

def count_sentiment_phrases():
    """Count good and bad sentiment phrases in each speech JSON file."""
    # Define input directory
    input_dir = "Speech Paragraphs Format 4"
    
    # Define sentiment phrases
    good_phrases = [
        # Economic Growth
        "economy", "grow", "economic", "growing", "grown", "growth",
        # Market Improvement
        "market", "improve", "marketing", "markets", "improved", "improving", "improvement",
        # Increase in Employment
        "increase", "employ", "increased", "increasing", "employs", "employed", "employment",
        # Rising Consumer Confidence
        "rise", "consumer", "confidence", "rising", "rose", "risen", "consumer's", "confident", "confidently",
        # Strong Economic Indicators
        "strong", "economy", "indicate", "stronger", "strongest", "economic", "indicator", "indicating", "indicated",
        # Growth in Investments
        "grow", "invest", "growth", "growing", "grown", "investment", "investing", "invested", "investors",
        # Expansion of the Labor Market
        "expand", "labor", "market", "expansion", "expanding", "expanded", "labor's", "marketing", "markets",
        # Bullish Market Trends
        "bull", "market", "trend", "bullish", "marketable", "marketing", "markets", "trending", "trends",
        # Positive Economic Outlook
        "positive", "economy", "outlook", "positively", "positivity", "economic", "outlooks",
        # Increased Productivity
        "increase", "productive", "increased", "increasing", "productivity", "productively"
    ]
    
    bad_phrases = [
        # Economic Downturn
        "economy", "downturn", "economic", "downturns",
        # Decline in Employment
        "decline", "employ", "declined", "declining", "employs", "employed", "employment",
        # Falling Stock Prices
        "fall", "stock", "price", "fallen", "falling", "stocks", "pricing", "prices",
        # Reduced Consumer Spending
        "reduce", "consumer", "spend", "reduced", "reducing", "reduction", "consumer's", "spends", "spending", "spent",
        # Bearish Market Conditions
        "bear", "market", "condition", "bearish", "marketable", "marketing", "markets", "conditioned", "conditioning", "conditions",
        # Negative Growth Forecasts
        "negative", "grow", "forecast", "negativity", "growing", "grown", "growth", "forecasts", "forecasting", "forecasted",
        # Increase in Unemployment
        "increase", "unemployment", "increased", "increasing", "unemployment",
        # Slowdown in Manufacturing
        "slow", "manufacture", "slowed", "slowing", "slowdown", "manufactures", "manufactured", "manufacturing",
        # Financial Instability
        "finance", "unstable", "financial", "instability", "instabilities",
        # Worsening Economic Conditions
        "worsen", "economy", "condition", "worsening", "worsened", "economic", "conditions", "conditioning"
    ]
    
    # Initialize lists to store results
    filenames = []
    good_counts = []
    bad_counts = []
    
    # Get all JSON files from input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # Process each file
    for json_file in json_files:
        input_path = os.path.join(input_dir, json_file)
        
        print(f"Counting sentiment phrases in {json_file}...")
        try:
            # Read JSON file
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Initialize counters for this file
            good_count = 0
            bad_count = 0
            
            # Process each page
            for page in data["pages"]:
                # Process each paragraph
                for paragraph in page["paragraphs"]:
                    # Process each sentence in the paragraph
                    for sentence in paragraph["sentences"]:
                        # Convert sentence to lowercase for case-insensitive matching
                        sentence_lower = sentence.lower()
                        
                        # Count good phrases
                        for phrase in good_phrases:
                            if phrase in sentence_lower:
                                good_count += 1
                        
                        # Count bad phrases
                        for phrase in bad_phrases:
                            if phrase in sentence_lower:
                                bad_count += 1
            
            # Store results
            filenames.append(json_file)
            good_counts.append(good_count)
            bad_counts.append(bad_count)
            
            print(f"Successfully processed {json_file}")
            print(f"Good sentiment phrases: {good_count}, Bad sentiment phrases: {bad_count}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'filename': filenames,
        'good_sentiment_count': good_counts,
        'bad_sentiment_count': bad_counts
    })
    
    # Save to CSV
    output_file = 'sentiment_phrase_counts.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSentiment phrase counts saved to {output_file}")

def calculate_certainty_scores():
    """Calculate certainty scores for each speech based on modal verb usage."""
    # Define input directory
    input_dir = "Format 4 JSON Lemmatized"
    
    # Define modal verbs with their certainty weights (1 = most certain, 0 = least certain)
    modal_verbs = {
        # High certainty (weight = 1.0)
        "will": 1.0,
        "shall": 1.0,
        "must": 1.0,
        "always": 1.0,
        "never": 1.0,
        "definitely": 1.0,
        "certainly": 1.0,
        "absolutely": 1.0,
        
        # Medium-high certainty (weight = 0.8)
        "going to": 0.8,
        "plan to": 0.8,
        "intend to": 0.8,
        "expect to": 0.8,
        
        # Medium certainty (weight = 0.6)
        "should": 0.6,
        "ought to": 0.6,
        "likely": 0.6,
        "probably": 0.6,
        
        # Medium-low certainty (weight = 0.4)
        "might": 0.4,
        "may": 0.4,
        "could": 0.4,
        "possibly": 0.4,
        
        # Low certainty (weight = 0.2)
        "maybe": 0.2,
        "perhaps": 0.2,
        "perchance": 0.2,
        "potentially": 0.2
    }
    
    # Initialize lists to store results
    filenames = []
    certainty_scores = []
    total_modal_verbs = []
    
    # Get all JSON files from input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # Process each file
    for json_file in json_files:
        input_path = os.path.join(input_dir, json_file)
        
        print(f"Calculating certainty score for {json_file}...")
        try:
            # Read JSON file
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Initialize counters for this file
            weighted_sum = 0
            total_count = 0
            
            # Process each page
            for page in data["pages"]:
                # Process each paragraph
                for paragraph in page["paragraphs"]:
                    # Process each sentence in the paragraph
                    for sentence in paragraph["sentences"]:
                        # Convert sentence to lowercase for case-insensitive matching
                        sentence_lower = sentence.lower()
                        
                        # Count modal verbs and their weights
                        for modal, weight in modal_verbs.items():
                            count = sentence_lower.count(modal)
                            weighted_sum += count * weight
                            total_count += count
            
            # Calculate normalized certainty score (0 to 1)
            certainty_score = weighted_sum / (total_count + 1) if total_count > 0 else 0.5
            
            # Store results
            filenames.append(json_file)
            certainty_scores.append(certainty_score)
            total_modal_verbs.append(total_count)
            
            print(f"Successfully processed {json_file}")
            print(f"Certainty score: {certainty_score:.3f}, Total modal verbs: {total_count}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'filename': filenames,
        'certainty_score': certainty_scores,
        'total_modal_verbs': total_modal_verbs
    })
    
    # Save to CSV
    output_file = 'certainty_scores.csv'
    df.to_csv(output_file, index=False)
    print(f"\nCertainty scores saved to {output_file}")

def analyze_sp500_returns_around_speeches(days_before_after=60):  # Increased default window
    """
    Analyze SP500 returns around speech dates.
    
    Args:
        days_before_after (int): Number of days to look before and after each speech date
    """
    # Define input directory
    input_dir = "Format 4 JSON Lemmatized"
    
    # Initialize lists to store results
    speech_dates = []
    filenames = []
    next_market_days = []
    next_day_open_return = []  # Return from previous close to next day's open
    next_day_close_return = []  # Return from open to close on next trading day
    returns_1d = []  # Returns over 1 trading day
    returns_3d = []  # Returns over 3 trading days
    returns_5d = []  # Returns over 5 trading days
    returns_10d = []  # Returns over 10 trading days
    momentum_before = []
    volatility_before = []
    momentum_after = []
    volatility_after = []
    
    # Get SP500 data
    sp500 = yf.Ticker("^GSPC")
    
    # Get all JSON files from input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # Process each file
    for json_file in json_files:
        input_path = os.path.join(input_dir, json_file)
        
        print(f"\nAnalyzing SP500 returns for {json_file}...")
        try:
            # Read JSON file
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract date from JSON and parse it (MM/DD/YYYY format)
            # Convert to timezone-aware datetime
            speech_date = datetime.strptime(data["date"], "%m/%d/%Y")
            speech_date = pytz.timezone('America/New_York').localize(speech_date)
            
            # Calculate date range for analysis
            start_date = speech_date - timedelta(days=days_before_after)
            end_date = speech_date + timedelta(days=days_before_after)
            
            # Get historical data
            hist_data = sp500.history(start=start_date, end=end_date)
            
            if not hist_data.empty:
                # Debug prints
                print(f"\nDebug info for {json_file}:")
                print(f"Start date: {start_date}")
                print(f"End date: {end_date}")
                print(f"Speech date: {speech_date}")
                print(f"Number of rows in hist_data: {len(hist_data)}")
                print(f"Data range: {hist_data.index[0]} to {hist_data.index[-1]}")
                
                # If we don't have enough data, try extending the window
                if len(hist_data) < 20:  # Arbitrary threshold
                    print("Not enough data, trying to extend the window...")
                    extended_start = speech_date - timedelta(days=days_before_after * 2)
                    extended_end = speech_date + timedelta(days=days_before_after * 2)
                    extended_data = sp500.history(start=extended_start, end=extended_end)
                    if not extended_data.empty and len(extended_data) > len(hist_data):
                        print(f"Got more data with extended window: {len(extended_data)} rows")
                        hist_data = extended_data
                
                # Calculate daily returns (both open and close)
                hist_data['Open_Return'] = hist_data['Open'].pct_change()
                hist_data['Close_Return'] = hist_data['Close'].pct_change()
                
                # Split data into before and after speech
                before_data = hist_data[hist_data.index < speech_date]
                after_data = hist_data[hist_data.index > speech_date]
                
                print(f"Number of rows in before_data: {len(before_data)}")
                print(f"Number of rows in after_data: {len(after_data)}")
                
                # Check if we have enough data
                if len(after_data) < 10:
                    print(f"WARNING: Not enough trading days after speech date for {json_file}")
                    print(f"Need at least 10 trading days, but only have {len(after_data)}")
                    print(f"Speech date: {speech_date.strftime('%Y-%m-%d')}")
                    print(f"Available dates after speech: {after_data.index.tolist()}")
                
                # Find next market day after speech
                next_market_day = after_data.index[0] if not after_data.empty else None
                
                # Calculate momentum (cumulative return)
                momentum_before_val = (1 + before_data['Close_Return']).prod() - 1 if not before_data.empty else None
                momentum_after_val = (1 + after_data['Close_Return']).prod() - 1 if not after_data.empty else None
                
                # Calculate volatility (standard deviation of returns)
                volatility_before_val = before_data['Close_Return'].std() * np.sqrt(252) if not before_data.empty else None
                volatility_after_val = after_data['Close_Return'].std() * np.sqrt(252) if not after_data.empty else None
                
                # Get returns for next market day after speech
                next_day_open_return_val = after_data['Open_Return'].iloc[0] if not after_data.empty else None
                next_day_close_return_val = after_data['Close_Return'].iloc[0] if not after_data.empty else None
                
                # Calculate returns over different periods
                # Get the closing price on the speech day (or last available price before speech)
                speech_day_close = before_data['Close'].iloc[-1] if not before_data.empty else None
                
                print(f"Speech day close price: {speech_day_close}")
                
                # Calculate returns over different periods with proper None handling
                # First check if we have enough data points
                has_1d_data = not after_data.empty and len(after_data) > 0
                has_3d_data = not after_data.empty and len(after_data) > 2
                has_5d_data = not after_data.empty and len(after_data) > 4
                has_10d_data = not after_data.empty and len(after_data) > 9
                
                print(f"Data availability - 1d: {has_1d_data}, 3d: {has_3d_data}, 5d: {has_5d_data}, 10d: {has_10d_data}")
                
                # Then calculate returns only if we have enough data
                try:
                    returns_1d_val = (after_data['Close'].iloc[0] / speech_day_close - 1) if (speech_day_close is not None and has_1d_data) else None
                    returns_3d_val = (after_data['Close'].iloc[2] / speech_day_close - 1) if (speech_day_close is not None and has_3d_data) else None
                    returns_5d_val = (after_data['Close'].iloc[4] / speech_day_close - 1) if (speech_day_close is not None and has_5d_data) else None
                    returns_10d_val = (after_data['Close'].iloc[9] / speech_day_close - 1) if (speech_day_close is not None and has_10d_data) else None
                except Exception as e:
                    print(f"Error calculating returns: {str(e)}")
                    returns_1d_val = None
                    returns_3d_val = None
                    returns_5d_val = None
                    returns_10d_val = None
                
                print(f"Calculated returns - 1d: {returns_1d_val}, 3d: {returns_3d_val}, 5d: {returns_5d_val}, 10d: {returns_10d_val}")
                
                # Store results
                speech_dates.append(speech_date)
                filenames.append(json_file)
                next_market_days.append(next_market_day)
                next_day_open_return.append(next_day_open_return_val)
                next_day_close_return.append(next_day_close_return_val)
                returns_1d.append(returns_1d_val)
                returns_3d.append(returns_3d_val)
                returns_5d.append(returns_5d_val)
                returns_10d.append(returns_10d_val)
                momentum_before.append(momentum_before_val)
                volatility_before.append(volatility_before_val)
                momentum_after.append(momentum_after_val)
                volatility_after.append(volatility_after_val)
                
                print(f"Successfully processed {json_file}")
                print(f"Speech date: {speech_date.strftime('%m/%d/%Y')}")
                print(f"Next market day: {next_market_day.strftime('%Y-%m-%d') if next_market_day else 'N/A'}")
                print(f"Next day returns - Open: {next_day_open_return_val:.4f if next_day_open_return_val is not None else 'N/A'}, Close: {next_day_close_return_val:.4f if next_day_close_return_val is not None else 'N/A'}")
                print(f"Returns - 1d: {returns_1d_val:.4f if returns_1d_val is not None else 'N/A'}, 3d: {returns_3d_val:.4f if returns_3d_val is not None else 'N/A'}, 5d: {returns_5d_val:.4f if returns_5d_val is not None else 'N/A'}, 10d: {returns_10d_val:.4f if returns_10d_val is not None else 'N/A'}")
                print(f"Momentum - Before: {momentum_before_val:.4f if momentum_before_val is not None else 'N/A'}, After: {momentum_after_val:.4f if momentum_after_val is not None else 'N/A'}")
                print(f"Volatility - Before: {volatility_before_val:.4f if volatility_before_val is not None else 'N/A'}, After: {volatility_after_val:.4f if volatility_after_val is not None else 'N/A'}")
            else:
                print(f"No data available for {json_file}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'ticker': ['^GSPC'] * len(filenames),
        'filename': filenames,
        'date': speech_dates,
        'next_market_day': next_market_days,
        'next_day_open_return': next_day_open_return,  # Return from previous close to next day's open
        'next_day_close_return': next_day_close_return,  # Return from open to close on next trading day
        'returns_1d': returns_1d,  # Returns over 1 trading day
        'returns_3d': returns_3d,  # Returns over 3 trading days
        'returns_5d': returns_5d,  # Returns over 5 trading days
        'returns_10d': returns_10d,  # Returns over 10 trading days
        'momentum_before': momentum_before,
        'volatility_before': volatility_before,
        'momentum_after': momentum_after,
        'volatility_after': volatility_after
    })
    
    # Sort by speech date
    df = df.sort_values('date')
    
    # Save to CSV
    output_file = 'sp500_returns_around_speeches.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSP500 returns analysis saved to {output_file}")
    
    return df

def combine_speech_analysis_data():
    """
    Combine word counts, sentiment analysis, and SP500 returns data into a single DataFrame.
    """
    try:
        # Read all three CSV files
        word_counts_df = pd.read_csv('speech_word_counts.csv')
        sentiment_df = pd.read_csv('sentiment_phrase_counts.csv')
        sp500_df = pd.read_csv('sp500_returns_around_speeches.csv')
        
        # Remove "_tokenized" from filenames in SP500 dataframe
        sp500_df['filename'] = sp500_df['filename'].str.replace('_tokenized', '')
        
        # Merge the dataframes
        # First merge word counts with sentiment
        combined_df = pd.merge(word_counts_df, sentiment_df, on='filename', how='outer')
        
        # Then merge with SP500 data
        combined_df = pd.merge(combined_df, sp500_df, on='filename', how='outer')
        
        # Sort by date
        combined_df = combined_df.sort_values('date')
        
        # Save to CSV
        output_file = 'combined_speech_analysis.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"\nCombined analysis saved to {output_file}")
        
        return combined_df
        
    except Exception as e:
        print(f"Error combining data: {str(e)}")
        return None

def extract_topics_from_speeches_by_group(formats, group_name, num_topics=10, num_words_per_topic=3):
    """
    Extract topics from speeches in specified formats using LDA with n-grams.
    
    Args:
        formats (list): List of format folders to process
        group_name (str): Name of the format group (for output files)
        num_topics (int): Number of topics to extract
        num_words_per_topic (int): Number of phrases to show per topic
    """
    # Collect all speeches
    all_speeches = []
    speech_filenames = []
    
    for folder in formats:
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} not found")
            continue
            
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        for json_file in json_files:
            try:
                with open(os.path.join(folder, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Combine all sentences from all paragraphs
                speech_text = []
                for page in data['pages']:
                    for paragraph in page['paragraphs']:
                        speech_text.extend(paragraph['sentences'])
                
                # Join sentences with spaces
                speech_text = ' '.join(speech_text)
                all_speeches.append(speech_text)
                speech_filenames.append(json_file)
                
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
    
    print(f"\nProcessing {group_name} - Found {len(all_speeches)} speeches")
    
    # Create document-term matrix with n-grams
    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1,1),  # Use both bigrams and trigrams
        min_df=2,  # Minimum document frequency to include a phrase
        max_df=0.95  # Maximum document frequency to exclude too common phrases
    )
    doc_term_matrix = vectorizer.fit_transform(all_speeches)
    
    # Apply LDA
    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        learning_method='batch',
        max_iter=20
    )
    lda_output = lda_model.fit_transform(doc_term_matrix)
    
    # Get feature names (phrases)
    feature_names = vectorizer.get_feature_names_out()
    
    # Print topics
    print(f"\nExtracted Topics for {group_name}:")
    for topic_idx, topic in enumerate(lda_model.components_):
        top_phrases_idx = topic.argsort()[:-num_words_per_topic-1:-1]
        top_phrases = [feature_names[i] for i in top_phrases_idx]
        print(f"\nTopic {topic_idx + 1}:")
        print(", ".join(top_phrases))
    
    # Save topic distribution for later use
    topic_distribution = pd.DataFrame(lda_output, columns=[f'Topic_{i+1}' for i in range(num_topics)])
    topic_distribution['filename'] = speech_filenames
    topic_distribution.to_csv(f'topic_distribution_{group_name}.csv', index=False)
    
    return lda_model, vectorizer, topic_distribution

def plot_topic_distribution_by_group(topic_distribution, group_name):
    """
    Plot histogram distribution of topics across speeches for a specific group.
    
    Args:
        topic_distribution (pd.DataFrame): Topic distribution dataframe
        group_name (str): Name of the format group (for output files)
    """
    # Calculate mean topic distribution
    topic_means = topic_distribution.drop('filename', axis=1).mean()
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    topic_means.plot(kind='bar')
    plt.title(f'Distribution of Topics Across {group_name} Speeches')
    plt.xlabel('Topic Number')
    plt.ylabel('Average Topic Strength')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'topic_distribution_{group_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTopic distribution plot saved as 'topic_distribution_{group_name}.png'")

def calculate_topic_strengths_for_group(lda_model, vectorizer, speech_file):
    """
    Calculate topic strength scores for a single speech.
    
    Args:
        lda_model: Fitted LDA model
        vectorizer: Fitted CountVectorizer
        speech_file: Path to the speech JSON file
    
    Returns:
        Dictionary of topic strength scores
    """
    try:
        # Read and process the speech
        with open(speech_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Combine all sentences
        speech_text = []
        for page in data['pages']:
            for paragraph in page['paragraphs']:
                speech_text.extend(paragraph['sentences'])
        
        # Join sentences with spaces
        speech_text = ' '.join(speech_text)
        
        # Transform the speech text
        speech_vector = vectorizer.transform([speech_text])
        
        # Get topic distribution
        topic_distribution = lda_model.transform(speech_vector)[0]
        
        # Create dictionary of topic strengths
        topic_strengths = {f'Topic_{i+1}': float(strength) for i, strength in enumerate(topic_distribution)}
        
        return topic_strengths
        
    except Exception as e:
        print(f"Error processing {speech_file}: {str(e)}")
        return None

def analyze_manual_topic_distribution(input_dir="Format 4 JSON Lemmatized"):
    """
    Analyze speeches for distribution of manually defined topics based on keyword frequency.

    Args:
        input_dir (str): Directory containing lemmatized JSON speeches
    """
    # Your manually defined topic keywords
    topic_keywords = {
        'Inflation and Prices': ['inflation', 'price', 'stability', 'pressure', 'elevated', 'expectation', 'development', 'moderate'],
        'Monetary Policy and Stance': ['policy', 'monetary', 'stance', 'decision', 'committee', 'assess', 'adjust'],
        'Labor Market and Employment': ['employment', 'labor', 'job', 'indicator', 'gain', 'unemployment', 'slack'],
        'Economic Activity and Growth': ['economic', 'growth', 'activity', 'recovery', 'moderate', 'strong'],
        'Federal Reserve and Mandate': ['reserve', 'federal', 'dual', 'mandate', 'consistent', 'attainment'],
        'Financial and Credit Markets': ['market', 'financial', 'condition', 'credit', 'security', 'mortgage'],
        'Asset Purchases and Balance Sheet': ['purchase', 'asset', 'treasury', 'holding', 'security', 'agency', 'mortgage'],
        'Certainty and Outlook': ['expect', 'risk', 'uncertainty', 'outlook', 'judge', 'remain', 'development'],
        'Time Horizon and Guidance': ['long', 'term', 'shorter', 'run', 'pace', 'coming', 'future'],
        'Liquidity and Resource Utilization': ['utilization', 'resource', 'condition', 'level', 'liquidity', 'consistent']
    }

    all_speeches = []
    speech_filenames = []

    print(f"Reading speeches from {input_dir}...")
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    for json_file in json_files:
        try:
            with open(os.path.join(input_dir, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)

            speech_text = []
            for page in data['pages']:
                for paragraph in page['paragraphs']:
                    speech_text.extend(paragraph['sentences'])

            speech_text = ' '.join(speech_text)
            all_speeches.append(speech_text)
            speech_filenames.append(json_file)

        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

    print(f"Found {len(all_speeches)} speeches")

    # Vectorize words (unigrams only since the data is lemmatized)
    vectorizer = CountVectorizer(lowercase=True, stop_words='english')
    term_matrix = vectorizer.fit_transform(all_speeches)
    feature_names = vectorizer.get_feature_names_out()

    # Map feature names to indices
    word_index = {word: i for i, word in enumerate(feature_names)}

    topic_distributions = []

    for doc_idx, doc_vector in enumerate(term_matrix):
        doc_array = doc_vector.toarray()[0]
        topic_strengths = {}
        topic_strengths['Filename'] = speech_filenames[doc_idx]

        total_strength = 0
        for topic, keywords in topic_keywords.items():
            score = 0
            for word in keywords:
                if word in word_index:
                    score += doc_array[word_index[word]]
            topic_strengths[topic] = score
            total_strength += score

        # Normalize to distribution
        if total_strength > 0:
            for topic in topic_keywords:
                topic_strengths[topic] = topic_strengths[topic] / total_strength
        else:
            for topic in topic_keywords:
                topic_strengths[topic] = 0.0

        topic_strengths['Sum'] = sum(topic_strengths[topic] for topic in topic_keywords)
        topic_distributions.append(topic_strengths)

    # Create DataFrame
    df = pd.DataFrame(topic_distributions)
    ordered_cols = ['Filename'] + list(topic_keywords.keys()) + ['Sum']
    df = df[ordered_cols]

    # Save
    output_file = 'manual_topic_distribution.csv'
    df.to_csv(output_file, index=False)
    print(f"\nManual topic distribution saved to {output_file}")

    # Summary
    print("\nAverage Topic Strengths:")
    mean_strengths = df.drop(['Filename', 'Sum'], axis=1).mean()
    for topic, val in mean_strengths.items():
        print(f"{topic}: {val:.4f}")

    return df

def merge_topic_and_returns_data():
    """
    Merge manual topic distribution data with returns data and reorder columns.
    """
    # Read both CSVs
    manual_topics_df = pd.read_csv('manual_topic_distribution.csv')
    returns_df = pd.read_csv('combined_speech_analysis.csv')
    
    # Convert filename columns to lowercase for matching
    manual_topics_df['Filename'] = manual_topics_df['Filename'].str.lower()
    returns_df['filename'] = returns_df['filename'].str.lower()
    
    # Merge the dataframes on filename (using outer join to keep all rows from both tables)
    merged_df = pd.merge(manual_topics_df, 
                        returns_df,
                        left_on='Filename', 
                        right_on='filename',
                        how='outer')  # Changed to outer join
    
    # Drop the duplicate filename column
    merged_df = merged_df.drop('filename', axis=1)
    
    # Move the specified columns to the end
    cols_to_move = ['returns_1d', 'returns_3d', 'returns_5d', 'momentum_after', 'volatility_after']
    other_cols = [col for col in merged_df.columns if col not in cols_to_move]
    final_cols = other_cols + cols_to_move
    
    merged_df = merged_df[final_cols]
    
    # Save to CSV with specific formatting parameters
    merged_df.to_csv('full_data_format_3.csv', 
                     index=False,
                     lineterminator='\n',  # Fixed parameter name
                     float_format='%.6f',
                     encoding='utf-8',
                     na_rep='NA')
    
    print("Merged data saved to full_data_format_3.csv")
    print(f"\nShape of merged dataframe: {merged_df.shape}")
    print(f"Number of rows in manual_topics_df: {len(manual_topics_df)}")
    print(f"Number of rows in returns_df: {len(returns_df)}")
    print(f"Number of rows in merged_df: {len(merged_df)}")
    
    return merged_df

if __name__ == "__main__":
    #process_all_files()
    count_words()
    #plot_word_distribution()
    #find_shortest_sentence()
    count_sentiment_phrases()
    calculate_certainty_scores()
    analyze_sp500_returns_around_speeches(days_before_after=5)
    combine_speech_analysis_data()
    
    # Define the two groups of formats
    group1_formats = [
        "Format 1 JSON",
        "Format 2 JSON"
    ]
    
    group2_formats = [
        "Format 3 JSON",
        "Format 4 JSON"
    ]
    
    # Process Group 1 (Formats 1 & 2)
    print("\nProcessing Group 1 (Formats 1 & 2)...")
    lda_model1, vectorizer1, topic_distribution1 = extract_topics_from_speeches_by_group(
        group1_formats, 
        "group1", 
        num_topics=10, 
        num_words_per_topic=3
    )
    plot_topic_distribution_by_group(topic_distribution1, "group1")
    
    # Process Group 2 (Formats 3 & 4)
    print("\nProcessing Group 2 (Formats 3 & 4)...")
    lda_model2, vectorizer2, topic_distribution2 = extract_topics_from_speeches_by_group(
        group2_formats, 
        "group2", 
        num_topics=10, 
        num_words_per_topic=3
    )
    plot_topic_distribution_by_group(topic_distribution2, "group2")
    
    # Analyze manual topic strengths
    analyze_manual_topic_distribution()
    
    # Merge topic and returns data
    merge_topic_and_returns_data()