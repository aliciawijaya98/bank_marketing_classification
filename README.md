# Bank Marketing Campaign — Classification Modeling

Predicting term deposit adoption using customer data from a Portuguese bank's telemarketing campaign.

## Overview

Telemarketing campaigns are resource-intensive — agents spend time and money calling thousands of customers, yet many calls do not convert. This project builds a **binary classification model** to identify customers most likely to make a term deposit, making campaigns more targeted and cost-effective.

**Dataset:**
Derived from [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing).

The dataset used in this project is a processed version of the original data, including cleaning, feature selection, and preprocessing steps tailored for modeling purposes.

**Target variable:** `deposit` - Did the customer make a term deposit? (yes/no)


## Business Problem
> *"How can the bank identify customers with high potential to make a term deposit, so that the telemarketing campaign becomes more efficient and targeted?"*

Without a model, agents contact customers randomly at a baseline conversion rate of **47.8%** — meaning roughly half of all calls are wasted.

## Business Impact
Based on the test set of 1,563 customers:

| | Without Model | With Model |
|---|---|---|
| Customers Called | 1,563 | 605 |
| Depositors Captured | 747 | 471 |
| Conversion Rate | 47.8% | 77.9% |
| Calls Saved | — | **958 (61.3%)** |

The model reduces call volume by **61.3%** while still capturing 63% of actual depositors. The trade-off is 276 missed depositors (False Negatives) — the cost of this efficiency gain.

## Approach
| Step | Details |
|---|---|
| **Problem Type** | Binary Classification |
| **Primary Metric** | Recall — minimizing False Negatives (missed depositors) is the business priority |
| **Supporting Metrics** | ROC-AUC, F1-Score |
| **Models Evaluated** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| **Final Model** | Random Forest (tuned via GridSearchCV) |

**Why Recall?** A missed depositor (False Negative) means lost revenue, which far outweighs the cost of a wasted call (False Positive).

## Model Performance
| Metric | Value |
|---|---|
| Recall | 63% |
| F1-Score | 67% |
| ROC-AUC | 77% |
| Accuracy | 70% |


## Key Features
| Feature | Importance | Insight |
|---|---|---|
| `balance` | ~13.4% | Higher balance → more likely to invest |
| `age` | ~12.6% | Older customers show more interest |
| `poutcome_success` | ~9.2% | Previous campaign success is a strong predictor |
| `contact_unknown` | ~7.1% | Unknown contact type reduces conversion likelihood |
| `campaign` | ~6.5% | Too many contacts may reduce likelihood |


## 🗂️ Project Structure

```
├── bank_marketing_modelling.ipynb      # Main notebook
├── data_bank_marketing_campaign.csv    # Dataset
├── model.pkl                           # Saved Random Forest model
├── feature_cols.pkl                    # Saved feature columns
└── README.md
```


## ⚙️ How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/aliciawijaya98/bank-marketing-modelling.git
   cd bank-marketing-classification
   ```

2. Install dependencies
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. Run the notebook
   ```bash
   jupyter notebook bank_marketing_modelling.ipynb
   ```


## Recommendations for Future Improvement

1. **Threshold tuning** — Lower prediction threshold from 0.50 to ~0.35–0.40 to capture more depositors at the cost of a moderate increase in wasted calls
2. **Add features** — Call duration, transaction history, or macroeconomic indicators (e.g., interest rates)
3. **Retrain periodically** — At least every 6 months or after major campaigns
4. **Try advanced models** — XGBoost or LightGBM may achieve higher Recall
5. **Customer segmentation** — Build separate models per customer group (age bracket, job type) for more precise targeting


## References

- Moro, S., Cortez, P., & Rita, P. (2014). [A Data-Driven Approach to Predict the Success of Bank Telemarketing](https://archive.ics.uci.edu/dataset/222/bank+marketing). UCI Machine Learning Repository.
