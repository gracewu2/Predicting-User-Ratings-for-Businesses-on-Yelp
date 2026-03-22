# Yelp Hybrid Recommendation Engine

## Overview
This project features a high-performance **Hybrid Recommendation System** built on **Apache Spark** to predict user ratings for businesses. By blending **Item-Based Collaborative Filtering** (Pearson Correlation) with a **Gradient Boosted Decision Tree (XGBoost)** regressor, the system achieves a highly accurate **RMSE of 0.9790**. 

The pipeline is designed to handle the "Cold Start" problem and scales efficiently using Spark RDDs to process high-dimensional features from millions of Yelp records.

---

## Technical Architecture
The system uses a weighted hybrid approach ($0.075$ CF + $0.925$ XGBoost) to maximize predictive power.

### 1. Collaborative Filtering (CF)
* **Algorithm**: Item-based CF using **Pearson Similarity**.
* **Significance Weighting**: Implements a coordination weight ($\lambda=16$) to penalize item pairs with fewer than 10 co-ratings, reducing noise in similarity scores.
* **Neighborhood Selection**: Uses a top-45 nearest neighbor approach for rating estimation.

### 2. Feature-Engineered XGBoost
The model transforms raw JSON data into a rich feature vector including:
* **User Metrics**: Elite status years, "Yelping since" longevity, friend counts, and aggregate compliment types.
* **Business Metadata**: Price ranges, WiFi availability, noise levels, and parking options.
* **Engagement Signals**: "Tip" counts and average "likes" per business.
* **Visual Data**: Photo counts and content ratios (e.g., ratio of "food" vs. "inside" photos) extracted from `photo.json`.

---

## Project Structure & File Overview

| File | Description |
| :--- | :--- |
| **yelp.py** | End-to-end pipeline including feature engineering, CF calculation, and XGBoost training/prediction. |
| **data/** | Directory for Yelp JSON/CSV datasets (`business.json`, `review_train.json`, `user.json`, `tip.json`, `photo.json`). |

---

## Performance Results
The model was tuned using `GridSearchCV` to optimize hyperparameters like `max_depth` and `learning_rate`.

* **Validation RMSE**: `0.9790`
* **Execution Time**: ~486.53 seconds
* **Error Distribution**:
    * **Error < 1.0**: 102,305 predictions
    * **Error 1.0 - 2.0**: 32,768 predictions
    * **Error > 4.0**: 0 (High stability)

---

## Setup & Installation

### 1. System Dependencies
This project requires **Apache Spark** and **Python 3.12+**. Ensure your environment has Java 8 or 11 installed for Spark.

### 2. Install Requirements
```bash
pip install pyspark xgboost scikit-learn numpy
