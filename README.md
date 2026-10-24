# Amazon Marketplace Returns Analysis

**Exploratory Data Analysis (EDA) | Python | Pandas | NumPy | Matplotlib | Seaborn | Machine Learning**

A data analysis project focused on exploring e-commerce order data, understanding customer ratings and delivery patterns, identifying data-quality issues, and analyzing product returns using a synthetic dataset generated with the assistance of a Large Language Model (LLM).

## Project Overview

The goal of this project is to understand an e-commerce-style dataset through Exploratory Data Analysis (EDA) and prepare it for a potential machine-learning classification task.

The analysis covers order quantities, product prices, discounts, ratings, delivery times, payment methods, review-related variables, and return status. It also examines missing values, duplicate records, variable distributions, and relationships between selected features.

**Data transparency:** This project uses synthetic data, not official Amazon data. The results describe the generated dataset and should not be interpreted as verified insights about Amazon or its customers.

## Objectives

- Understand the structure and quality of the dataset.
- Identify and remove exact duplicate records.
- Inspect missing values and understand their distribution.
- Explore numerical and categorical variables individually.
- Identify skewed distributions and potential outliers.
- Examine relationships between selected variables.
- Understand the distribution of the return target.
- Identify potential data leakage risks before model development.
- Apply responsible data-analysis practices when working with synthetic data.

## Dataset Description

The dataset contains **100,120 rows and 30 columns** before duplicate removal.

After removing 120 exact duplicate rows, the cleaned dataset contains **100,000 rows**.

The dataset includes information about:

- Orders and dates
- Customers and products
- Product categories and locations
- Quantities, prices, and discounts
- Customer ratings and review-related variables
- Payment methods and fulfillment details
- Delivery times and return status

The target variable for the planned classification task is `returned_flag`, where `1` indicates a returned order and `0` indicates an order that was not returned.

### Important Data Limitation

The dataset was generated with LLM assistance. Its values, distributions, and relationships may contain artificial patterns and do not necessarily represent real-world e-commerce behavior. Model performance on this dataset would not establish performance on actual marketplace transactions.

## Tools and Technologies

| Tool | Purpose |
|---|---|
| Python | Data analysis and scripting |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Jupyter Notebook | Interactive EDA and documentation |
| Scikit-learn | Planned preprocessing and machine-learning workflow |

## Project Structure

```text
amazon-marketplace-analytics/
│
├── data/
│   └── amazon_employee_eda_model_assignment.csv
│
├── models/
│   └── return_model.joblib
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_modeling.ipynb
│
├── src/
│   ├── predict.py
│   └── train.py
│
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

*Note: The structure above describes the intended project organization. The model artifact is generated after training, and local data/model files may be excluded from Git through `.gitignore`.*

## EDA Workflow

The analysis follows these main steps:

1. **Data understanding:** Inspected the first and last records, dataset shape, column names, data types, and descriptive statistics.
2. **Data-quality checks:** Examined missing values and duplicate records, then removed exact duplicate rows.
3. **Univariate analysis:** Studied numerical distributions, categorical frequencies, and boxplots to understand central tendencies, spread, and potential outliers.
4. **Bivariate analysis:** Explored selected relationships, including category and order amount, discounts and ratings, quantity and order amount, payment methods and order amount, and delivery time and ratings.
5. **Multivariate analysis:** Examined correlations among selected numerical variables and compared category-level summaries.
6. **Return analysis:** Investigated the distribution of returned and non-returned records.
7. **Modeling considerations:** Identified the need for leakage-safe preprocessing and evaluation metrics suitable for an imbalanced target.

## Key Findings

1. **Dataset cleaning:** Removing 120 exact duplicate records reduced the dataset from 100,120 to 100,000 rows.
2. **Price distribution:** Unit prices and total order amounts were right-skewed, with their means higher than their medians.
3. **Order quantity:** The median quantity was 1, while the mean was approximately 1.49.
4. **Discounts:** The average discount was approximately 12.56%, with a median of 10%.
5. **Ratings:** The average recorded rating was approximately 3.89 out of 5, with 2,534 missing rating values.
6. **Delivery times:** The median delivery time was 3 days, and the middle 50% of observed values was approximately 2–4 days.
7. **Review data:** Average review length was approximately 3.38 words.
8. **Payment methods:** UPI was the most frequently recorded payment method in the generated dataset.
9. **Return distribution:** Approximately 7.01% of records were marked as returned, indicating an imbalanced target.
10. **Synthetic-data caveat:** These are descriptive findings about the generated dataset, not verified claims about Amazon's actual business.

## Data Leakage Considerations

Before training a return-prediction model, features must be selected according to the intended prediction time.

For example, if the goal is to predict returns before an order is placed or shipped, fields such as `returned`, `returned_flag`, `order_status`, customer reviews, ratings, and delivery outcomes may expose information that would not yet be available.

- `returned_flag` is the target, not an input feature.
- `returned` is another representation of the outcome and should not be used as a predictor.
- `order_status` must be evaluated for possible outcome leakage.
- Identifier columns such as `order_id` and `customer_id` should not automatically be treated as ordinary predictive features.
- Post-purchase variables should only be considered when the prediction scenario genuinely allows them.

Preprocessing steps that learn parameters from data should be fitted on the training set and then applied to the test set.

## Evaluation Strategy

Because returned orders make up only about 7.01% of the cleaned dataset, accuracy alone may provide a misleading picture of model performance.

The planned evaluation includes:

- Precision
- Recall
- F1-score
- Confusion matrix
- Average precision or precision-recall analysis

The appropriate balance between precision and recall will depend on the intended use of the model.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/dipesh50061-blip/amazon-marketplace-analytics.git
cd amazon-marketplace-analytics
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter streamlit joblib
```

### 4. Run the EDA notebook

```bash
jupyter notebook
```

Open the notebook in the `notebooks/` directory and run the cells in order. Ensure the CSV is available at the path expected by the notebook.

### 5. Train the model

If the training script and its required paths are configured:

```bash
python src/train.py
```

### 6. Run the prediction application

If the trained model artifact and application are available:

```bash
streamlit run app.py
```

*The commands assume that the repository contains the referenced files and that their paths and dependencies are configured correctly.*

## Use of LLMs

LLMs assisted with synthetic dataset creation and provided support for understanding analytical concepts, organizing the workflow, interpreting visualizations, and documenting the project.

LLM-generated explanations were treated as assistance rather than evidence. Dataset statistics were derived through analysis, and methodological recommendations were checked against technical documentation where appropriate.

The synthetic origin of the dataset is explicitly disclosed because it affects the interpretation and generalizability of the results.

## References

1. IBM, *CRISP-DM Help Overview*: https://www.ibm.com/docs/en/spss-modeler/18.6.0?topic=dm-crisp-help-overview
2. scikit-learn, *Common pitfalls and recommended practices*: https://scikit-learn.org/stable/common_pitfalls.html
3. Google for Developers, *Classification: Accuracy, Recall, Precision, and Related Metrics*: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall
4. imbalanced-learn documentation: https://imbalanced-learn.org/stable/

These references support the project's analytical methodology; they do not validate the synthetic data as representative of actual Amazon transactions.

## Conclusion

This project helped me develop a practical understanding of EDA, data cleaning, visualization, and the challenges involved in preparing data for machine learning. It also taught me to question patterns rather than accept every result at face value.

The next step is to develop and evaluate a leakage-safe classification model. Since the dataset is synthetic, the primary goal is to demonstrate analytical skills and a sound machine-learning workflow—not to make claims about real-world Amazon returns.

## Author

**Depesh Kumar**
*AI Engineer Intern*

- GitHub: https://github.com/dipesh50061-blip
- Portfolio: https://sites.google.com/view/depesh-ai-journey
- LinkedIn: https://www.linkedin.com/in/depesh-kumar-918626399/

---

*Built as a learning project to practice data analysis and machine learning with transparent use of synthetic data.*