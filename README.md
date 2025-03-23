Dataset: https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks/data
1. Introduction
Music streaming platforms generate millions of new songs every year, making it challenging for record labels to decide which songs to promote. Given the high costs of marketing, labels need data-driven insights to identify potential hits before investing in promotions.
The goal of this project is to predict a song’s popularity using machine learning models, allowing record labels to focus their efforts on high-potential tracks. Our dataset consists of 300,000 songs (2000-2023) with 21 features, analyzing audio, artist, and track characteristics to determine their impact on popularity.
Key Questions Addressed:
Can we predict a song’s popularity before its release?
What features contribute most to a song’s success?
How can record labels use this model to optimize marketing strategies?
Value of the Solution:
Marketing Optimization: Helps record labels allocate promotional budgets more efficiently.
Artist Strategy: Provides insights into which musical characteristics contribute to success.





2. Dataset Overview
2.1 Data Collection & Features
We use the Spotify Music Data, which contains numerous attributes describing song features. Our dataset consists of 600,000 songs spanning 80 genres (2000-2023), with 21 features. The key features include:
Basic Track Information:
artist_name – Name of the artist/band
track_name – Title of the song
year – Year of release
genre – Genre category (e.g., Pop, Rock, Hip-Hop)
track_id – Unique identifier for each song
Popularity Score (Target Variable):
popularity – Score (0-100) indicating the song's success on streaming platforms
Musical Characteristics:
danceability – Suitability for dancing (0-1 scale)
energy – Intensity and activity level (0-1 scale)
key – Musical key of the song
loudness – Overall volume level (dB)
mode – Major (1) or minor (0) tonality
valence – Positivity of a song’s mood (0-1)
tempo – Speed of the song (BPM)
Acoustic & Structural Features:
speechiness – Presence of spoken words
acousticness – Likelihood of being acoustic
instrumentalness – Amount of instrumental content
liveness – Live performance probability
duration_ms – Track duration in milliseconds
time_signature – Beats per measure (e.g., 4/4)
2.2 Data Cleaning & Preprocessing
Data Cleaning:
Removed irrelevant or identifier columns (e.g., track ID and artist name).
Eliminated songs with zero popularity from the dataset.
Dropped missing values as the numbers were negligible.
Feature Scaling:
Normalized numerical values (tempo, loudness, popularity).
Outlier Removal:
Adjusted extreme values in duration_ms and loudness.
One-Hot Encoding:
Converted categorical variables (genre, key) into numerical representations.
Feature Engineering:
Created an artist popularity column (median song popularity per artist) and classified artists into:
Underground (0-25)
Emerging (25-50)
Mainstream (50-75)
Superstars (75-100)
This was not used for modeling but can be a useful feature to help record labels group artists into categories.

The process followed for modeling: 



3. Exploratory Data Analysis
Current Popularity Score Analysis:
Analyzed the distribution of popularity scores.
Removed entries with popularity = 0.

Artist Popularity Score:
Examined the distribution of artists based on popularity levels.



Genre-Based EDA:
Identified trends in genre-wise song popularity.
Genre-Based EDA & Filtering

Feature Correlation Analysis:
Investigated relationships between features.








4. Model Development
4.1 General Base Models
We started by testing general machine-learning models to establish a baseline for song popularity prediction:
Experiments & Variations Attempted
Baseline Testing: Used a simple linear regression model to understand the fundamental relationships.
Hyperparameter Tuning:
Used GridSearchCV to optimize best model parameters.
Tested different learning rates, tree depths, and number of estimators.
Feature Engineering Experiments:
We attempted polynomial features to capture nonlinear patterns but saw they worsened performance.
Tried removing lower-importance features and observed model performance impact.
Neural Network Adjustments:
Used different activation functions (ReLU, linear) to test their effects.
Adjusted layers and neurons but saw marginal improvement over Random forest.
Validation Methodology
Evaluation Metrics:
Root Mean Squared Error - RMSE (Error), R² Score (Model Fit)
Cross-Validation:
Used K-Fold Cross-Validation to prevent overfitting

Why Random Forest?
After testing various models, Random Forest Regressor outperformed others, achieving 79.45% accuracy with an RMSE of 7.54.
Handles non-linear relationships well
Resistant to overfitting
Effective with high-dimensional data

Model
RMSE (Error)
R² (Accuracy)
Ridge Regression
8.9
71.38%
Lasso Regression
8.96
71.00%
XGBoost
8.47
74.05%
Gradient Boosting
8.81
71.95%
Neural Networks
8.46
74.4%
Random Forest (Best Model after Hyperparameter tuning)
7.52
79.56%

Findings:
Linear models (Ridge, Lasso) provided interpretability but had limited accuracy.
XGBoost & Gradient Boosting improved performance but required tuning.
Neural Networks performed well but needed extensive computational resources.
Random Forest outperformed all models, offering the best balance of accuracy and robustness.
4.2 Algorithmic Clustering-Based Models
We then applied Principal Component Analysis (PCA) & K-Means Clustering to uncover latent song structures and tested models within each cluster.
Process:
Dimensionality Reduction (PCA): Reduced dataset to five principal components.
K-Means Clustering (K=4): Grouped songs based on musical characteristics.
Model Training Within Clusters: Tested multiple models for each cluster.


Cluster Breakdown
Cluster 0
Moderate popularity, balanced characteristics.
Best Model: Random Forest (RMSE: 7.2250, R²: 0.7448).
Cluster 1
Largest cluster with diverse song attributes.
Best Model: Random Forest (RMSE: 7.8080, R²: 0.7908).
Cluster 2
Songs with more distinct structural or genre-based traits.
Best Model: Random Forest (RMSE: 7.6629, R²: 0.7821).
Cluster 3
Smallest cluster, possibly niche or experimental tracks.
Best Model: Random Forest (RMSE: 7.7821, R²: 0.6648).
Lower R² suggests higher variability or unpredictability in song popularity.
Findings:
Clustering failed to improve predictions compared to the general model, with three out of the four groups having worse RMSE and all clusters having lower R².
Random Forest remained the best-performing model across clusters.
Cluster 3 had lower R squared, likely due to niche genres with high variability.

4.3 Genre & Mood-Based Models
Given the influence of genre and mood on song popularity, we classified songs into four mood-based categories using valence and energy levels.
Mood Categories:
Happy/Energetic: High valence (≥0.5) + High energy (≥0.5)
Peaceful/Relaxed: High valence (≥0.5) + Low energy (<0.5)
Angry/Tense: Low valence (<0.5) + High energy (≥0.5)
Sad/Depressed: Low valence (<0.5) + Low energy (<0.5)


Mood
RMSE
R² (Accuracy)
Happy/Energetic
10.41
59.27%
Angry/Tense
9.49
63.35%
Peaceful/Relaxed
9.87
57.63%
Sad/Depressed
9.88
60.88%

Findings:
Mood features slightly improved model performance.
Happy/Energetic songs had the highest average popularity.
Predictions within mood-based models had lower accuracy compared to general models.
4.4 Best Performing Model
After testing multiple approaches, Random Forest Regressor consistently delivered the highest accuracy across:
General models
Clustering-based models
Mood & Genre-based models
Final Model Performance:
Best Model: Random Forest
Overall R²: 79.45%
RMSE: 7.54
5. Key Insights & Findings
Findings from Data Mining
Danceability, energy, and loudness are the strongest predictors of popularity.
Tempo has a moderate effect, while duration shows little impact.
Polynomial features slightly improved Random Forest performance but not significantly.
Neural Networks required extensive tuning but did not outperform Random forest.
High-impact features: danceability, energy, valence, tempo
Minor influence: instrumentalness, acousticness, liveness
Genre matters – Pop & hip-hop dominate, while jazz and classical see lower popularity scores
Clustering Results
We tested genre-specific models, but a generalized model performed best across all genres.
6. Business Impact & Applications
6.1 Real-World Applications
The predictive model can significantly enhance marketing strategies by identifying high-potential tracks before their release, enabling record labels and streaming platforms to allocate resources more effectively. By leveraging data-driven insights, companies can optimize promotional efforts, targeting the right audience segments and maximizing engagement. Additionally, this approach allows for better budget optimization, ensuring investments are directed toward songs with the highest likelihood of commercial success, ultimately improving overall profitability.
6.2 Revenue Projections
Enhancing hit prediction accuracy can have substantial financial benefits for the music industry. Increasing the success rate of songs from 10% to 15-20% could result in an additional $10M–$30M in annual revenue, driven by higher streaming numbers, licensing deals, and concert ticket sales. Moreover, optimizing marketing spend through data-driven decision-making could save or strategically reallocate over $5M annually, reducing inefficiencies and maximizing returns on promotional campaigns. These improvements highlight the tangible impact of integrating machine learning models into music industry operations.
7. Challenges & Future Enhancements
One of the main challenges faced in this analysis is the subjectivity of popularity, as factors like social media virality, artist fan base, and marketing campaigns significantly influence a song's success beyond its inherent audio features. Additionally, the lack of external data limited the model’s predictive power, as we relied solely on audio attributes without incorporating broader market trends, listener demographics, or industry-driven promotional efforts. These constraints highlight the need for a more holistic approach to understanding what drives a song’s popularity.
To enhance the model’s accuracy and applicability, future improvements should focus on integrating external data sources. Incorporating social media trends from platforms like Twitter and TikTok can help capture real-time viral potential, while Natural Language Processing (NLP) techniques can analyze lyrics to assess emotional tone and thematic relevance to success. Additionally, leveraging real-time streaming data from platforms like YouTube, Apple Music, and Spotify would provide a more dynamic view of song performance. Testing ensemble models, such as combining Random Forest and XGBoost, could improve predictive performance by capturing both linear and complex non-linear relationships. Lastly, developing region-specific models would enhance localization, accounting for cultural and geographic variations in music preferences.
8. Conclusion
This project demonstrates that machine learning can effectively predict music popularity, giving record labels a powerful tool for decision-making.
Key Takeaways:
80% accuracy in predicting hit potential.
Optimized marketing and release strategies.
Millions in potential revenue impact.
By enhancing the model with social media trends and real-time data, we can further refine its predictive power and help labels make smarter investments in the music industry.
