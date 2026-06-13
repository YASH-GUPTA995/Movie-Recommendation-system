# 🎬 Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-F7931E?style=flat-square&logo=scikit-learn)
![Dataset](https://img.shields.io/badge/Dataset-MovieLens%20100K-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

A content-aware neural network recommendation system built on the MovieLens 100K dataset. The model learns separate embeddings for users and movies using a two-tower neural network architecture, then ranks movies by predicted rating for a new user.

---

## 📌 Table of Contents

- [Demo](#-demo)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)

---

## 🚀 Demo

> Run the notebook end-to-end to generate personalized recommendations for a new user based on genre preferences.

**Example output — Top 10 recommendations for a user who prefers Drama and Animation:**

| Rank | Movie Title | Predicted Rating |
|------|-------------|-----------------|
| 1 | Schindler's List (1993) | 4.7 |
| 2 | Spirited Away (2001) | 4.6 |
| 3 | The Shawshank Redemption (1994) | 4.6 |
| ... | ... | ... |

---

## 🧠 How It Works

The system uses a **Two-Tower Neural Network** architecture:

```
User Features                    Movie Features
(genre prefs + avg rating)       (genres + year + avg rating)
        │                                  │
   ┌────▼────┐                       ┌────▼────┐
   │ Dense   │                       │ Dense   │
   │ 128→64  │                       │ 128→64  │
   │ →32 emb │                       │ →32 emb │
   └────┬────┘                       └────┬────┘
        │       L2 Normalize             │
        └──────────┐   ┌─────────────────┘
                   ▼   ▼
              Cosine Similarity
                   │
             Predicted Rating (0–1, rescaled to 0.5–5.0)
```

**Feature Engineering:**
- **User features:** Weighted genre preference scores (rating × genre indicator), average rating given
- **Movie features:** One-hot genre flags (20 genres), release year extracted from title, community average rating
- **Scaling:** `StandardScaler` on user/movie features, `MinMaxScaler` on target ratings

**Inference:** For a new user, the model scores all 9,742 movies and returns the top-N by predicted rating.

---

## 📊 Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — provided by GroupLens Research.

| Property | Value |
|----------|-------|
| Total ratings | 100,836 |
| Movies | 9,742 |
| Users | 610 |
| Rating scale | 0.5 – 5.0 (half-star) |
| Genres | 20 |
| Files used | `movies.csv`, `ratings.csv` |

---

## 📁 Project Structure

```
movie-recommender/
├── data/
│   ├── movies.csv          # Movie metadata (title, genres)
│   └── ratings.csv         # User ratings with timestamps
├── notebooks/
│   └── movie_recommender.ipynb   # Full pipeline notebook
├── requirements.txt
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook notebooks/movie_recommender.ipynb
```

### Running Recommendations

Open the notebook and run all cells. To customize the new user's preferences, edit the `user_genre_preferences` dictionary in the **Inference** section:

```python
user_genre_preferences = {
    'Drama':     5,   # 0 = no interest, 5 = love it
    'Animation': 3,
    'Comedy':    2,
    # ... all 20 genres
}
```

---

## 📈 Results

> Metrics reported on the held-out test set (20% of ratings).

| Metric | Value |
|--------|-------|
| Loss (MSE, normalized) | ~0.08 |
| Training epochs | 30 |
| Batch size | 256 |
| Embedding size | 32 |
| Total parameters | ~50K |

*Note: MAE and RMSE on the original 0.5–5.0 scale will be added after the next training run.*

---

## 🛠 Tech Stack

| Library | Purpose |
|---------|---------|
| `TensorFlow / Keras` | Two-tower neural network |
| `Scikit-Learn` | Preprocessing, train/test split |
| `Pandas` | Data loading and feature engineering |
| `NumPy` | Matrix operations for inference |

---

## 🔮 Future Improvements

- [ ] Add a Streamlit web app with genre sliders and live recommendations
- [ ] Integrate TMDB API for movie posters
- [ ] Add MAE / RMSE metrics on the original rating scale
- [ ] Fix train/val/test split to avoid validation leakage
- [ ] Add minimum rating count filter to exclude obscure movies
- [ ] Experiment with collaborative filtering baseline (SVD) for comparison

---

## 📄 License

This project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/) — please cite GroupLens Research if you use it academically.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4: 1–19.

---

## 🙋‍♂️ Author

Made by Yash Gupta
