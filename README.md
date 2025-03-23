## Predicting Song Popularity Using Machine Learning

### 1. Introduction
Music streaming platforms generate millions of new songs every year, making it challenging for record labels to decide which songs to promote. Given the high costs of marketing, labels need data-driven insights to identify potential hits before investing in promotions.

The goal of this project is to predict a song’s popularity using machine learning models, allowing record labels to focus their efforts on high-potential tracks. Our dataset consists of 300,000 songs (2000-2023) with 21 features, analyzing audio, artist, and track characteristics to determine their impact on popularity.

#### Key Questions Addressed:
- Can we predict a song’s popularity before its release?
- What features contribute most to a song’s success?
- How can record labels use this model to optimize marketing strategies?

#### Value of the Solution:
- **Marketing Optimization:** Helps record labels allocate promotional budgets more efficiently.
- **Artist Strategy:** Provides insights into which musical characteristics contribute to success.

---

### 2. Dataset Overview
#### 2.1 Data Collection & Features
We use the Spotify Music Data, which contains numerous attributes describing song features. Our dataset consists of 600,000 songs spanning 80 genres (2000-2023), with 21 features.

#### Key Features:
- **Basic Track Information:**
  - `artist_name` – Name of the artist/band
  - `track_name` – Title of the song
  - `year` – Year of release
  - `genre` – Genre category (e.g., Pop, Rock, Hip-Hop)
  - `track_id` – Unique identifier for each song
- **Popularity Score (Target Variable):**
  - `popularity` – Score (0-100) indicating the song's success on streaming platforms
- **Musical Characteristics:**
  - `danceability`, `energy`, `key`, `loudness`, `mode`, `valence`, `tempo`
- **Acoustic & Structural Features:**
  - `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `duration_ms`, `time_signature`

#### 2.2 Data Cleaning & Preprocessing
- **Data Cleaning:**
  - Removed irrelevant or identifier columns (e.g., `track_id`, `artist_name`)
  - Eliminated songs with zero popularity
  - Dropped missing values
- **Feature Scaling:**
  - Normalized numerical values (tempo, loudness, popularity)
- **Outlier Removal:**
  - Adjusted extreme values in `duration_ms` and `loudness`
- **One-Hot Encoding:**
  - Converted categorical variables (`genre`, `key`) into numerical representations
- **Feature Engineering:**
  - Created an artist popularity column and classified artists into categories:
    - Underground (0-25)
    - Emerging (25-50)
    - Mainstream (50-75)
    - Superstars (75-100)

---

### 3. Exploratory Data Analysis
- **Current Popularity Score Analysis:**
  - Analyzed the distribution of popularity scores
  - Removed entries with popularity = 0
- **Artist Popularity Score:**
  - Examined the distribution of artists based on popularity levels
- **Genre-Based EDA:**
  - Identified trends in genre-wise song popularity
- **Feature Correlation Analysis:**
  - Investigated relationships between features

---

### 4. Model Development
#### 4.1 General Base Models
We started by testing general machine-learning models:

| Model | RMSE | R² (Accuracy) |
|--------|------|--------------|
| Ridge Regression | 8.9 | 71.38% |
| Lasso Regression | 8.96 | 71.00% |
| XGBoost | 8.47 | 74.05% |
| Gradient Boosting | 8.81 | 71.95% |
| Neural Networks | 8.46 | 74.4% |
| **Random Forest** (Best Model) | **7.52** | **79.56%** |

#### Why Random Forest?
- Handles non-linear relationships well
- Resistant to overfitting
- Effective with high-dimensional data

#### 4.2 Algorithmic Clustering-Based Models
- **PCA & K-Means Clustering (K=4)** applied to group songs based on musical characteristics.
- **Cluster-wise Model Training:**
  - Random Forest performed best in all clusters.
  - Clustering did not improve predictions significantly.

#### 4.3 Genre & Mood-Based Models
Classified songs into four mood categories using valence and energy:

| Mood | RMSE | R² (Accuracy) |
|------|------|--------------|
| Happy/Energetic | 10.41 | 59.27% |
| Angry/Tense | 9.49 | 63.35% |
| Peaceful/Relaxed | 9.87 | 57.63% |
| Sad/Depressed | 9.88 | 60.88% |

**Findings:**
- Mood-based models had lower accuracy compared to the general model.
- Happy/Energetic songs had the highest average popularity.

#### 4.4 Best Performing Model
- **Final Model: Random Forest**
- **Overall R²: 79.45%**
- **RMSE: 7.54**

---

### 5. Key Insights & Findings
#### Findings from Data Mining:
- **Strong Predictors:** Danceability, energy, and loudness.
- **Moderate Influence:** Tempo, while duration had little impact.
- **Genre Trends:** Pop & Hip-Hop dominate, jazz & classical see lower popularity.
- **Polynomial features did not significantly improve performance.**
- **Neural Networks required extensive tuning but did not outperform Random Forest.**

#### Clustering Results:
- **Genre-specific models underperformed compared to a generalized model.**

---

### 6. Business Impact & Applications
#### 6.1 Real-World Applications
- **Marketing Optimization:** Helps record labels allocate promotional budgets efficiently.
- **Pre-release Popularity Prediction:** Identifies potential hits before they are launched.
- **Better Targeting:** Enables artists and platforms to refine promotional strategies.

#### 6.2 Revenue Projections
- **Enhancing hit prediction accuracy** can increase the success rate of promoted songs from **10% to 15-20%**, leading to substantial revenue growth for record labels.

---

### 7. Conclusion
This project demonstrates the potential of machine learning in predicting song popularity. Random Forest emerged as the best-performing model, enabling record labels to make data-driven marketing decisions and maximize promotional efficiency.

---

### 8. Future Work
- Improve model performance with deep learning.
- Use streaming data for real-time predictions.
- Enhance feature engineering with lyric sentiment analysis.

