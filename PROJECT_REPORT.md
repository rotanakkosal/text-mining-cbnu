# Sentiment Analysis of Amazon Mobile Phone Reviews
## Machine Learning Project Report

**Author:** [Your Name]  
**Date:** [Current Date]  
**Course:** Text Mining / Machine Learning

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Dataset Description](#dataset-description)
4. [Methodology](#methodology)
5. [Data Preprocessing](#data-preprocessing)
6. [Feature Engineering](#feature-engineering)
7. [Model Training & Evaluation](#model-training--evaluation)
8. [Results & Analysis](#results--analysis)
9. [Model Limitations & Improvements](#model-limitations--improvements)
10. [Conclusion](#conclusion)
11. [References](#references)

---

## 1. Executive Summary

This project implements a complete machine learning pipeline to analyze customer reviews from Amazon's unlocked mobile phone dataset. The goal is to automatically classify reviews as either **Defects/Complaints** or **Praise/Noise**, and generate actionable business intelligence in the form of a prioritized scorecard.

**Key Achievements:**
- Processed **382,075 customer reviews** after data cleaning
- Achieved **93.37% accuracy** and **86.66% F1-score** using XGBoost
- Generated prioritized scorecard identifying top complaint categories
- Identified and addressed model limitations (sarcasm, negation)

**Main Findings:**
- **Battery/Power** issues are the most frequent specific complaint category (20,061 issues)
- **General/Other** category has the highest average anger score (0.159), indicating high customer frustration
- Model successfully generalizes with minimal overfitting (1.44% gap between train and test performance)

---

## 2. Project Overview

### 2.1 Objective

To build an automated sentiment analysis system that:
1. Classifies customer reviews as Defect/Complaint (Class 1) or Praise/Noise (Class 0)
2. Identifies the most critical product issues for business prioritization
3. Provides actionable insights through a prioritized scorecard

### 2.2 Problem Statement

With hundreds of thousands of customer reviews, manual analysis is impractical. This project automates the process of:
- Understanding customer sentiment at scale
- Identifying common product defects
- Prioritizing issues based on frequency and customer frustration levels

---

## 3. Dataset Description

### 3.1 Data Source

**Dataset:** Amazon Unlocked Mobile Phone Reviews  
**File:** `Amazon_Unlocked_Mobile.csv`  
**Original Size:** 413,850 reviews

### 3.2 Data Characteristics

| Metric | Value |
|--------|-------|
| **Total Reviews (Original)** | 413,850 |
| **Reviews After Filtering** | 382,075 |
| **3-Star Reviews Removed** | 31,775 (7.7%) |
| **Defects (Class 1: 1-2 stars)** | 97,078 (25.4%) |
| **Praise (Class 0: 4-5 stars)** | 284,997 (74.6%) |

### 3.3 Data Structure

**Columns:**
- `Reviews`: Raw review text
- `Rating`: Star rating (1-5)
- `label`: Binary classification (1 = Defect, 0 = Praise)

### 3.4 Data Preprocessing Decisions

**3-Star Reviews Filtering:**
- **Decision:** Removed all 3-star reviews
- **Reason:** Ambiguous sentiment - neither clearly positive nor negative
- **Impact:** Improved model clarity by focusing on clear examples

**Class Imbalance:**
- Dataset is imbalanced (74.6% positive, 25.4% negative)
- F1-score used as primary metric to account for imbalance

---

## 4. Methodology

### 4.1 Pipeline Architecture

```
Raw CSV → Data Loading → Text Cleaning → Feature Engineering → 
Vectorization → Model Training → Evaluation → Scorecard Generation
```

### 4.2 Technology Stack

| Library | Purpose | Version |
|---------|---------|---------|
| **pandas** | Data manipulation | Latest |
| **numpy** | Numerical operations | Latest |
| **scikit-learn** | Machine learning algorithms | Latest |
| **vaderSentiment** | Sentiment analysis | Latest |
| **xgboost** | Gradient boosting model | Latest |
| **matplotlib** | Visualization | Latest |

---

## 5. Data Preprocessing

### 5.1 Text Cleaning Process

**Code Implementation:**
```python
def clean_text(text):
    text = text.lower()                    # Standardize case
    text = re.sub(r'[^\w\s]', '', text)   # Remove punctuation
    return text
```

**Steps:**
1. **Lowercase Conversion:** "TERRIBLE" → "terrible"
2. **Punctuation Removal:** "Great!" → "great"
3. **Missing Value Handling:** Empty reviews filled with empty strings

**Example Transformation:**
- **Before:** `"The battery is TERRIBLE! It dies after 2 hours."`
- **After:** `"the battery is terrible it dies after 2 hours"`

### 5.2 Label Creation

**Binary Classification:**
- **Class 1 (Defect/Complaint):** Reviews with 1-2 star ratings
- **Class 0 (Praise/Noise):** Reviews with 4-5 star ratings
- **Excluded:** 3-star reviews (ambiguous)

---

## 6. Feature Engineering

### 6.1 TF-IDF Vectorization

**Implementation:**
```python
vectorizer = TfidfVectorizer(
    max_features=1000,        # Top 1000 words
    stop_words='english',     # Remove common words
    ngram_range=(1, 2)         # Words + 2-word phrases
)
```

**Process:**
1. **Vocabulary Building:** Analyzed all 382,075 reviews to identify top 1000 most important words/phrases
2. **TF-IDF Calculation:** 
   - **TF (Term Frequency):** How often word appears in specific review
   - **IDF (Inverse Document Frequency):** How rare word is across all reviews
   - **TF-IDF = TF × IDF:** Balances local importance with global rarity

**Key Features:**
- **1000 features** representing word/phrase importance
- **N-grams (1-2):** Captures phrases like "battery life", "doesn't work"
- **Stop words removed:** Common words like "the", "is", "and" filtered out

**Example Vocabulary:**
- Product features: "battery", "screen", "camera", "charger"
- Sentiment words: "terrible", "amazing", "disappointed", "love"
- Action phrases: "doesn't work", "highly recommend", "don't buy"

### 6.2 VADER Sentiment Analysis

**Implementation:**
```python
analyzer = SentimentIntensityAnalyzer()
df['compound_score'] = df['clean_text'].apply(
    lambda text: analyzer.polarity_scores(text)['compound']
)
```

**Purpose:**
- Provides context-aware sentiment score (-1 to +1)
- Better handles negation and sarcasm than simple word counting
- Becomes **Feature #1001** in final dataset

**Why VADER:**
- Rule-based system with built-in grammar understanding
- Handles negation: "not great" → negative score
- Provides additional signal beyond word frequencies

### 6.3 Feature Combination

**Final Feature Set:**
- **Features 1-1000:** TF-IDF scores for vocabulary words
- **Feature 1001:** VADER compound sentiment score
- **Total:** 1,001 features per review

**Data Structure:**
- **Training Set:** 305,660 reviews × 1,001 features
- **Test Set:** 76,415 reviews × 1,001 features
- **Sparsity:** ~98% zeros (typical for text data)

---

## 7. Model Training & Evaluation

### 7.1 Models Compared

| Model | Description | Hyperparameters |
|-------|-------------|-----------------|
| **Logistic Regression** | Linear classifier | max_iter=1000 |
| **XGBoost** | Gradient boosting | use_label_encoder=False, eval_metric='logloss' |

### 7.2 Training Process

**Train-Test Split:**
- **Training:** 80% (305,660 reviews)
- **Testing:** 20% (76,415 reviews)
- **Random State:** 42 (reproducibility)

**Evaluation Metrics:**
- **Accuracy:** Overall correctness
- **F1-Score:** Harmonic mean of precision and recall (better for imbalanced data)

### 7.3 Results

**Model Performance Comparison:**

| Model | Train Accuracy | Test Accuracy | Train F1 | Test F1 |
|-------|---------------|---------------|-----------|----------|
| **Logistic Regression** | 92.81% | 92.60% | 0.8545 | 0.8499 |
| **XGBoost** | **94.08%** | **93.37%** | **0.8810** | **0.8666** |

**Best Model Selected:** XGBoost

**Key Observations:**
1. **XGBoost outperformed** Logistic Regression by ~1.7% accuracy
2. **Minimal overfitting:** Only 1.44% gap between train and test F1-scores
3. **Strong generalization:** Model performs well on unseen data

### 7.4 Overfitting Analysis

**XGBoost Performance:**
- **Training F1:** 88.10%
- **Test F1:** 86.66%
- **Gap:** 1.44%

**Conclusion:** Model is **not overfitting**. The small gap indicates good generalization.

**Reasons for Good Generalization:**
1. **Large dataset:** 305,660 training examples prevent memorization
2. **Feature engineering:** VADER score provides robust signal
3. **XGBoost regularization:** Built-in mechanisms prevent overfitting

---

## 8. Results & Analysis

### 8.1 Prioritized Scorecard

**Top Complaint Categories:**

| Category | Issue Count | Avg Anger Score | Priority |
|----------|-------------|-----------------|----------|
| **General/Other** | 49,230 | 0.159 | 🔴 High |
| **Battery/Power** | 20,061 | 0.113 | 🟠 Medium-High |
| **Screen/Display** | 10,386 | 0.106 | 🟡 Medium |
| **Connectivity** | 7,055 | 0.095 | 🟢 Low-Medium |
| **Performance/UI** | 3,152 | 0.118 | 🟡 Medium |
| **Audio/Mic** | 2,979 | 0.128 | 🟡 Medium |

### 8.2 Key Insights

1. **General/Other Category:**
   - Largest category (49,230 issues)
   - Highest anger score (0.159)
   - **Action:** Needs detailed sub-categorization

2. **Battery/Power Issues:**
   - Second most frequent (20,061 issues)
   - Clear, actionable category
   - **Action:** Top priority for engineering team

3. **Screen/Display:**
   - Third most frequent (10,386 issues)
   - Moderate anger score
   - **Action:** Quality control focus area

4. **Audio/Mic:**
   - Smallest category but high anger score (0.128)
   - **Action:** High impact despite lower frequency

### 8.3 Business Recommendations

1. **Immediate Actions:**
   - Investigate "General/Other" category for sub-categorization
   - Prioritize battery/power improvements
   - Review screen/display quality control

2. **Long-term Strategy:**
   - Implement automated review monitoring
   - Set up alerts for emerging issues
   - Track improvement metrics over time

---

## 9. Model Limitations & Improvements

### 9.1 Identified Limitations

**1. Sarcasm Detection:**
- **Example:** "The phone does a great job reminding me how often I need to reboot it..."
- **Model Prediction:** Positive (incorrect)
- **VADER Score:** Also failed (0.86 - positive)
- **Reason:** Complex irony requires deeper context understanding

**2. Negation Handling:**
- **Example:** "The phone does not do a great job..."
- **Initial Model:** Predicted Positive (incorrect)
- **After VADER Integration:** Improved but not perfect
- **Reason:** Simple models struggle with grammatical negation

### 9.2 Improvements Implemented

**1. VADER Feature Engineering:**
- Added VADER sentiment score as Feature #1001
- Improved model's ability to understand context
- Result: XGBoost performance increased

**2. N-gram Features:**
- Captured 2-word phrases like "doesn't work", "battery life"
- Better context understanding than single words

### 9.3 Future Improvements

**1. Advanced Models:**
- **Transformer Models (BERT, RoBERTa):** Better context understanding
- **Deep Learning:** LSTM/GRU for sequence modeling
- **Ensemble Methods:** Combine multiple models

**2. Feature Engineering:**
- **Topic Modeling:** LDA for automatic categorization
- **Named Entity Recognition:** Extract product names, brands
- **Emotion Detection:** Beyond sentiment to specific emotions

**3. Data Enhancement:**
- **Data Augmentation:** Generate synthetic examples
- **Active Learning:** Focus on difficult examples
- **Multi-class Classification:** Include 3-star reviews as neutral class

---

## 10. Conclusion

### 10.1 Project Success

This project successfully demonstrates:
- ✅ Complete ML pipeline from raw data to actionable insights
- ✅ High-performance model (93.37% accuracy, 86.66% F1-score)
- ✅ Practical business intelligence output
- ✅ Critical thinking in identifying and addressing limitations

### 10.2 Key Learnings

1. **Feature Engineering is Critical:** Adding VADER score significantly improved performance
2. **Model Selection Matters:** XGBoost outperformed simpler models
3. **Data Quality > Quantity:** Filtering ambiguous data improved results
4. **Real-world Challenges:** Sarcasm and negation remain difficult problems

### 10.3 Final Thoughts

The project successfully transforms unstructured customer feedback into structured, actionable business intelligence. While the model has limitations with complex language patterns, it provides valuable insights for product improvement prioritization.

The combination of traditional ML (TF-IDF) with modern sentiment analysis (VADER) and powerful algorithms (XGBoost) demonstrates a practical approach to real-world text mining challenges.

---

## 11. References

1. **Scikit-learn Documentation:** https://scikit-learn.org/
2. **VADER Sentiment Analysis:** Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
3. **XGBoost Documentation:** https://xgboost.readthedocs.io/
4. **TF-IDF Explanation:** Salton, G. & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval.

---

## Appendix A: Code Structure

```
text-mining/
├── src/
│   ├── data_loader.py      # Data loading and labeling
│   ├── preprocessing.py    # Text cleaning and feature engineering
│   ├── modeling.py         # Model training and scorecard generation
│   └── main.py            # Pipeline orchestration
├── data/
│   └── Amazon_Unlocked_Mobile.csv
├── model/
│   ├── best_model.joblib  # Saved XGBoost model
│   └── vectorizer.joblib   # Saved TF-IDF vectorizer
├── outputs/
│   └── prioritized_scorecard.csv
├── predict.py             # Prediction script for new reviews
└── requirements.txt       # Python dependencies
```

## Appendix B: Sample Predictions

**Example 1: Negative Review**
- **Input:** "The battery life is terrible! It dies after just 2 hours."
- **Prediction:** Defect/Complaint (Class 1) ✅
- **Confidence:** High

**Example 2: Positive Review**
- **Input:** "I love this phone! The camera is amazing and the battery lasts all day."
- **Prediction:** Praise/Noise (Class 0) ✅
- **Confidence:** High

**Example 3: Sarcasm (Limitation)**
- **Input:** "The phone does a great job reminding me how often I need to reboot it..."
- **Prediction:** Positive (incorrect) ❌
- **Note:** Demonstrates model limitation with sarcasm

---

**End of Report**
