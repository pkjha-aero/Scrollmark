# Scrollmark

## Tasks Performed:
- Sentiments (Positive, Negative, Neutral) computed using `OpenAI` and `LangChain`.
- Sentiments visualized as pie charts with a chosen color code.
- Analyzed and visualized requent words of each of the sentiments for `media_caption` and `comment_text`.
- Using the `timestamp`, computed `Date`, `Hour`, and `DayName` for analysis of trends, i.e.,
  - How do the sentiments vary by `Date` (beginning of month or end of month)?
  - How do the sentiments vary by `Hour` (Morning versus Evening hours)?
  - How do the sentiments vary by `DayName` (Mon, Tue versus Sat and Sun)?

## Key Findings:
- There is preponderance of positive or neutral sentiments for `media_caption` as well as `comment_text`.
- Some words associated with the sentiments may be misleading and need further assessment.
- There are more positive commenst in the evenings and on Mondays.
 
## Code Organization:
- `Scrollmark.ipynb`: Driver jupyter notebook.
- `helper_functions.py`: Contained helper functions for computation and visualization.

## Code Dependencies:
- pandas                    2.3.1                    pypi_0    pypi
- openpyxl                  3.1.5           py312h5eee18b_1  
- matplotlib                3.10.0          py312h06a4308_0   
- python-dateutil           2.9.0post0      py312h06a4308_2  
- langchain                 0.3.27                   pypi_0    pypi
- Access to **OpenAI API** via key, i.e., `OPENAI_API_KEY` in environment variable

## Running the Code:
- Obtain `OPENAI_API_KEY` from Open AI.
- In a conda environment with the above-mentioned packages, simply run the notebook `Scrollmark.ipynb`after assigning the right string value to `OPENAI_API_KEY`

## Extension Proposal
- Parallelize wherever possible to scale to larger data size.
- Upgrade Open AI limits to deal with larger data set.
- Implement topic or theme modeling to reviews.
- Extend the sentiment analysis to include specific aspects or features mentioned in the reviews.
- Better analyze the most frequent words associated with the sentiments so that only the most relevant are considered.
- Identify and extract named entities like product names, features.
- Extend trend analysis from sentiments to topics and entity recognition.

## AI & Tool Usage:
- Chat GPT (web version)
- Google Search (Regula and AI mode)
