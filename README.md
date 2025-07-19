# Wizzmate - Indonesian Tourism Recommendation System

## 📌 Project Summary
This project aims to build a *machine learning* model capable of providing recommendations for tourist attractions in Indonesia. Using a **Content-Based Filtering** approach with **Cosine Similarity** and a **Collaborative Filtering** approach, this model was developed to suggest new destinations to users.

The analysis shows that the **Content-Based** model successfully groups attractions with similar categories. For instance, when a user likes a "Nature Reserve," the system recommends other "Nature Reserves," proving the logic is well-implemented. **Collaborative Filtering** is also explored as a method to provide more personalized recommendations based on user behavior.

🔗 **[View Complete Code on GitHub](https://github.com/YusepFathul/Wizzmate-Rekomendasi-Wisata/tree/main/Notebook)**

---

## 🎯 Project Objectives
- **Tourism Data Exploration:** To analyze and understand the Indonesian tourism dataset, including the distribution of categories and ratings.
- **Content-Based Filtering Implementation:** To build a recommendation model from scratch that works based on the attributes or "content" of the items (tourist attractions).
- **Exploring Collaborative Filtering (CF):** To research and prepare for the implementation of a **CF** model to leverage user rating data.
- **Model Validation:** To test the model's ability to provide relevant and logical recommendations based on given inputs.
- **Identifying Potential for Development:** To identify the weaknesses of the current approach and formulate steps for future improvements, such as a Hybrid Model.

---

## 🧰 Technologies & Libraries Used
- **Workspace:** Jupyter Notebook / Google Colaboratory
- **Core Libraries:**
  - **`Python`**
  - **`Pandas`** & **`NumPy`**: For data manipulation, cleaning, and analysis.
  - **`Scikit-learn`**: For implementing `TfidfVectorizer` and `cosine_similarity`.
  - **`TensorFlow`**: For building and training deep learning models for recommendation systems.
  - **`Matplotlib`** & **`Seaborn`**: For data visualization.

---

## 🛠️ Project Methodology

1.  **Data Collection & Understanding:**
    - **Source:** Indonesian Tourism Dataset consisting of information on tourist attractions, users, and ratings.
    - **Cleaning:** Merged relevant datasets and handled missing values. A total of 437 tourist spot entries were used.

2.  **Data Exploration:**
    - Analyzed the data distribution, such as the number of tourist spots per category, to understand the dataset's characteristics.

3.  **Feature Engineering (Content-Based):**
    - **Feature Selection:** The primary focus for content-based recommendations was the `Category` feature.
    - **Text Vectorization:** Converted categorical data into a numerical representation using **TF-IDF Vectorizer**. This technique transforms each category into a vector whose similarity can be measured.

4.  **Machine Learning Modeling:**
    - **Content-Based Similarity:** Calculated a similarity matrix between all tourist spots using the **Cosine Similarity** metric. This matrix contains scores (0-1) indicating how similar each pair of spots is.
    - **Recommendation Function:** Created a function that takes the name of a tourist spot as input, finds the highest similarity scores, and returns the top 10 places as a recommendation.
    - **Collaborative Filtering with TensorFlow:** Prepared the user-item interaction data (ratings) to be used with **TensorFlow Recommenders** for building a **Collaborative Filtering** model.

---

## 📈 Results and Analysis

### Distribution of Tourism Categories
Data visualization shows that the dataset is dominated by three main categories, providing a good variety for the recommendation model.
- **Amusement Park**: 128 places (29.3%)
- **Culture**: 118 places (27.0%)
- **Nature Reserve**: 117 places (26.8%)
- **Marine**: 54 places (12.4%)
- **Shopping Center**: 11 places (2.5%)
- **Place of Worship**: 9 places (2.1%)

*Insight: With a wide selection in the Amusement Park, Culture, and Nature Reserve categories, the model has sufficient data to provide diverse recommendations within these domains.*

### Recommendation Model Performance
The Content-Based model was evaluated by observing the quality of its recommendations. Here is an example of the results when seeking recommendations for **"Goa Pindul"** (category: Nature Reserve).

| Recommendation Name | Category | Similarity Score |
| :--- | :--- | :--- |
| Gua Jomblang | Nature Reserve | 1.000 |
| Gua Cerme | Nature Reserve | 1.000 |
| Air Terjun Sri Gethuk| Nature Reserve | 1.000 |


**Result Analysis:** The model successfully recommended other tourist spots in the same category ("Nature Reserve") with a perfect similarity score (1.0). This demonstrates that the content-based logic works as expected.

**Performance Conclusion:** The Content-Based model is highly effective at finding identical items based on a single feature (category), but it is not yet capable of providing more nuanced recommendations. **Collaborative Filtering** is the next step to address this.

---

## 💡 Conclusion and Key Insights
1.  **Success of Content-Based Logic:** This project proves that **Content-Based Filtering** using TF-IDF and Cosine Similarity is highly effective for recommending items based on a single attribute like category.
2.  **Contextually Relevant Recommendations:** The model is capable of providing sensible recommendations, where a nature attraction is recommended alongside other nature attractions.
3.  **Foundation for a More Complex System:** Although simple, this model serves as a strong foundation for developing a hybrid recommendation system, especially when combined with a **TensorFlow**-based **Collaborative Filtering** model.
4.  **The Importance of Data:** The quality and completeness of data (descriptions, user ratings, tags, facilities) are crucial for the success of any recommendation system.

---

## 🚀 Challenges and Potential Improvements
- **Main Challenge:** The Content-Based model is too simplistic as it relies on only one feature (`Category`). This makes the recommendations less personal and diverse.
- **Potential Improvements:**
  - **Using Additional Features:** Incorporating other features like `City` or `Description` for more specific content-based recommendations.
  - **Full Implementation of Collaborative Filtering:** Fully develop the **Collaborative Filtering** model using **TensorFlow** by leveraging user rating data to provide recommendations based on the behavior of other users with similar tastes.
  - **Building a Hybrid Model:** Combining the **Content-Based** and **Collaborative Filtering** approaches to achieve more accurate and personalized recommendations.
  - **Creating a User Interface (UI):** Developing a simple application to allow users to interact directly with the recommendation system.

---

## Download Link
### Download our apps [Here](https://drive.google.com/file/d/1VzqvTgbxI2HkPA0fww4otlSHAc74GEc4/view?usp=sharing)

---

## 👤 Contact
- **Name:** Yusep Fathul Anwar
- **LinkedIn:** [https://www.linkedin.com/in/yusepfathulanwar/](https://www.linkedin.com/in/yusepfathulanwar/)
- **Github:** [https://github.com/YusepFathul](https://github.com/YusepFathul)



