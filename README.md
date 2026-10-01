\# IMDb Movie Review Sentiment Analysis



\## Assignment: Word Embeddings for Sentiment Analysis

\- \*\*Problem:\*\* Classify IMDb movie reviews as positive or negative.

\- \*\*Embeddings used:\*\* Pre-trained GloVe (glove-wiki-gigaword-100), compared against TF-IDF.

\- \*\*How to run:\*\* Open the notebook in Jupyter or Kaggle and run all cells in order.

\- \*\*Results:\*\*



| Model | Accuracy | Precision | Recall | F1 |

|---|---|---|---|---|

| TF-IDF + Logistic Regression | 0.884 | 0.876 | 0.895 | 0.885 |

| GloVe + Logistic Regression | 0.798 | 0.800 | 0.795 | 0.797 |



\- \*\*Findings:\*\* TF-IDF outperformed GloVe on this task, likely because simple averaging of word vectors loses word sequence and emphasis, whereas TF-IDF captured key sentiment terms directly.

