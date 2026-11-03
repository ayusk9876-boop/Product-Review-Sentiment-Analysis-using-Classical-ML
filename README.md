# Product Review Sentiment Analysis Using Classical Machine Learning

Classifies product review text as positive or negative using TF-IDF and three classical ML classifiers.

## Group members
| Name | Roll number |
|---|---|
| AYUS KUMAR | 1ETBTA202598 |
| SMRUTI RANJAN DAS | 1ETBTA202599 |
| SOUMYARANJAN JENA | 1ETBTA2025100 |
| CHARCHIT KUMAR BEURA | 1ETBTA2025101 |

Project guide: Yugansh Varshney (please confirm with faculty).

## Dataset
Download [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). Put `Reviews.csv` in `data/`. Required columns: `Text`, `Score`. Ratings 1–2 = negative, 4–5 = positive, 3 = excluded. The dataset is not included in this repository.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```
Optional app after training: `streamlit run app.py`

## Workflow
1. Load and clean reviews.
2. Map ratings to binary labels.
3. Split 80:20 with stratification.
4. Extract TF-IDF unigrams and bigrams.
5. Train Multinomial Naive Bayes, Logistic Regression and Linear SVM.
6. Compare accuracy, precision, recall and F1-score.
7. Save the highest-F1 pipeline and confusion matrix.

The script uses up to 50,000 eligible reviews by default (`MAX_ROWS` in `main.py`); set it to `None` to use all eligible rows.

## Outputs
`outputs/model_comparison.csv`, `outputs/classification_report.csv`, `outputs/confusion_matrix.png`; saved pipeline: `models/sentiment_model.joblib`.

**Important:** Performance depends on the dataset and run configuration. Use actual generated metrics; do not invent results.

## Limitations
Neutral reviews are excluded. Sarcasm, mixed sentiment, slang and domain-specific language may be misclassified. Food reviews may not generalize to every product category.

## References
- Amazon Fine Food Reviews dataset: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
- Pedregosa et al. (2011), Scikit-learn: Machine Learning in Python, JMLR 12, 2825–2830.
- Manning, Raghavan & Schütze (2008), Introduction to Information Retrieval.
- https://scikit-learn.org/stable/
- https://docs.streamlit.io/
