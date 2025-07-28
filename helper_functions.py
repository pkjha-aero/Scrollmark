import os
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import datetime
from dateutil import parser

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage


def visualize_sentiments (df: pd.DataFrame, text_sentiment_map: dict, sentiment_colors: dict) -> None:
    sorted_sentiment_colors = dict(sorted(sentiment_colors.items()))
    num_text_types = len(text_sentiment_map.keys())
    fig, axes = plt.subplots(1, num_text_types, figsize=(12, 6))
    for sent_type_count, text_key in enumerate(text_sentiment_map.keys()):
        #print(text_key, text_sentiment_map[text_key])
        counts_sent = df[text_sentiment_map[text_key]].value_counts()
        counts_sent = counts_sent.reindex(sorted_sentiment_colors.keys())
        counts_colors = pd.DataFrame(counts_sent).fillna(0)
        counts_colors['colors'] = sorted_sentiment_colors.values()
        axes[sent_type_count].pie(counts_colors['count'], colors=counts_colors['colors'], labels=counts_colors.index, autopct="%1.1f%%", startangle=90)
        axes[sent_type_count].set_title(text_key)
        axes[sent_type_count].legend()
        
    
    plt.suptitle("Sentiment Distributions Across Text Columns", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def reduce_df_to_a_sentiment (df_orig: pd.DataFrame, text_sentiment_map: dict, desired_sentiment: str) -> pd.DataFrame:
    df = df_orig.copy()
    for text_key in text_sentiment_map.keys():
        #print(text_key, text_sentiment_map[text_key])
        df = df [df[text_sentiment_map[text_key]] == desired_sentiment]

    return df

# Function to extract keywords and counts using OpenAI
def extract_keywords_with_counts(llm, prompt_template, text: str):
    prompt = prompt_template.format(text=text[:4000])  # truncate if needed
    response = llm([HumanMessage(content=prompt)])
    
    # Parse response into dict
    result = {}
    for line in response.content.strip().split("\n"):
        if ":" in line:
            try:
                key, val = line.split(":")
                result[key.strip()] = int(val.strip())
            except ValueError:
                continue
    return result


def extract_frequent_words_for_sentiment (df_orig, text_sentiment_map, desired_sentiment, llm, prompt_template, freq_words_collection):
    # Reduce the data frame so that each text column has the same sentiment type
    #df = reduce_df_to_a_sentiment (df_orig, text_sentiment_map, desired_sentiment)
    
    for sent_type_count, text_key in enumerate(text_sentiment_map.keys()):
        #print(text_key, text_sentiment_map[text_key])
        df = df_orig [df_orig[text_sentiment_map[text_key]] == desired_sentiment]
        col = text_key
        text = " ".join(df[col].dropna().astype(str).tolist())
        result = extract_keywords_with_counts(llm, prompt_template, text)
        
        freq_words_collection[text_key][desired_sentiment] = result
    
        print(f"\nTop 10 keywords from column '{col}' that represent '{desired_sentiment}' sentiments:")
        for k, v in result.items():
            print(f"{k}: {v}")

    return freq_words_collection


def visualize_frequent_words_for_sentiment (freq_words_collection, sentiment_colors):

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), sharey=False)
    fig.suptitle("Top Keywords by Sentiment for Each Text Source", fontsize=16, fontweight='bold')
    
    for ax, (source, sentiments) in zip(axes, freq_words_collection.items()):
        ax.set_title(f"{source.replace('_', ' ').title()}")
        all_words = []
        all_counts = []
        all_colors = []
    
        for sentiment, words in sentiments.items():
            for word, count in words.items():
                all_words.append(word)
                all_counts.append(count)
                all_colors.append(sentiment_colors.get(sentiment, 'black'))
    
        # Plot histogram
        ax.barh(all_words, all_counts, color=all_colors)
        ax.invert_yaxis()  # Highest count at top
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Words")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def visualize_sentiment_trend (grouped, time_type, text_key, sentiment_colors):
    fig, axes = plt.subplots(len(grouped.columns), 1, figsize=(10, 6), sharex=True)
    
    # Ensure axes is iterable
    if len(grouped.columns) == 1:
        axes = [axes]
    
    for i, sentiment in enumerate(grouped.columns):
        axes[i].plot(grouped.index, grouped[sentiment], marker='o', color = sentiment_colors.get(sentiment, 'black'))
        axes[i].set_title(f"Sentiment: {sentiment}")
        axes[i].set_ylabel("Count")
        axes[i].grid(True)
    
    plt.xlabel(time_type)
    plt.tight_layout()
    plt.suptitle(f"Sentiment Counts by {time_type} for {text_key}", fontsize=16, fontweight='bold', y=1.02)
    plt.show()


