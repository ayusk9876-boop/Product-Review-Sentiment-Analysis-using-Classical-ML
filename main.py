from pathlib import Path
import re, warnings, joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"Reviews.csv"
OUT=ROOT/"outputs"; MODELS=ROOT/"models"
OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
RANDOM_STATE=42
MAX_ROWS=50000  # Change to None to use all eligible rows.

def clean_text(text):
    text=str(text).lower()
    text=re.sub(r"<[^>]+>|https?://\\S+|www\\.\\S+"," ",text)
    text=re.sub(r"[^a-z\\s']"," ",text)
    return re.sub(r"\\s+"," ",text).strip()

def load_data():
    if not DATA.exists():
        raise FileNotFoundError("Download Reviews.csv from the Kaggle dataset linked in README.md and place it in data/.")
    df=pd.read_csv(DATA)
    if not {"Text","Score"}.issubset(df.columns):
        raise ValueError("Reviews.csv must contain Text and Score columns.")
    df=df[["Text","Score"]].dropna().copy()
    df["Score"]=pd.to_numeric(df["Score"],errors="coerce")
    df=df[df["Score"].isin([1,2,3,4,5])]
    df=df[df["Score"]!=3].copy()
    df["sentiment"]=(df["Score"]>=4).astype(int)
    df["clean_review"]=df["Text"].map(clean_text)
    df=df[df["clean_review"].str.len()>0]
    if MAX_ROWS and len(df)>MAX_ROWS:
        n=min(MAX_ROWS//2,df["sentiment"].value_counts().min())
        df=df.groupby("sentiment",group_keys=False).sample(n=n,random_state=RANDOM_STATE)
    return df.reset_index(drop=True)

def main():
    warnings.filterwarnings("ignore")
    df=load_data()
    print("Usable reviews:",len(df))
    print(df["sentiment"].map({0:"Negative",1:"Positive"}).value_counts())
    Xtr,Xte,ytr,yte=train_test_split(df.clean_review,df.sentiment,test_size=.2,random_state=RANDOM_STATE,stratify=df.sentiment)
    classifiers={
        "Multinomial Naive Bayes":MultinomialNB(),
        "Logistic Regression":LogisticRegression(max_iter=1000,class_weight="balanced"),
        "Linear SVM":LinearSVC(class_weight="balanced",random_state=RANDOM_STATE)
    }
    rows=[]; fitted={}
    for name,clf in classifiers.items():
        pipe=Pipeline([("tfidf",TfidfVectorizer(stop_words="english",ngram_range=(1,2),max_features=50000,min_df=2,max_df=.95,sublinear_tf=True)),("classifier",clf)])
        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); fitted[name]=(pipe,pred)
        rows.append({"Model":name,"Accuracy":accuracy_score(yte,pred),"Precision":precision_score(yte,pred,zero_division=0),"Recall":recall_score(yte,pred,zero_division=0),"F1-score":f1_score(yte,pred,zero_division=0)})
        print("\\n",name,"\\n",classification_report(yte,pred,target_names=["Negative","Positive"]))
    results=pd.DataFrame(rows).sort_values("F1-score",ascending=False)
    results.to_csv(OUT/"model_comparison.csv",index=False)
    best=results.iloc[0]["Model"]; pipe,pred=fitted[best]
    joblib.dump(pipe,MODELS/"sentiment_model.joblib")
    pd.DataFrame(classification_report(yte,pred,target_names=["Negative","Positive"],output_dict=True)).transpose().to_csv(OUT/"classification_report.csv")
    cm=confusion_matrix(yte,pred)
    plt.figure(figsize=(6,5)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Negative","Positive"],yticklabels=["Negative","Positive"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix - "+best); plt.tight_layout()
    plt.savefig(OUT/"confusion_matrix.png",dpi=200); plt.close()
    print("\\nMODEL COMPARISON\\n",results.to_string(index=False))
    print("\\nBest model by F1-score:",best)

if __name__=="__main__": main()
