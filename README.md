# 🎯 Customer Churn Predictor

A machine learning project to predict whether a customer will stop using a service based on historical data. This project uses classification techniques, feature engineering, and interactive deployment to help businesses identify at-risk customers.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)

---

## 📝 Description

Develop a predictive model to identify customers likely to churn (leave the service) using historical customer data. The project implements multiple classification algorithms, handles class imbalance, and provides an interactive web interface for real-time predictions.

### Key Features:
- ✅ Binary classification model (Churn: Yes/No)
- ✅ Multiple ML algorithms comparison (Logistic Regression, Random Forest, Gradient Boosting)
- ✅ Advanced feature engineering and preprocessing
- ✅ Class imbalance handling using SMOTE
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Comprehensive visualizations and EDA
- ✅ Interactive Streamlit web application

---

## 📊 Dataset

**Source:** [Kaggle - Telecom Customer Churn Prediction](https://www.kaggle.com/datasets/dileep070/logisticregression-telecomcustomer-churmprediction/data)

The dataset includes three CSV files:
- `churn_data.csv` - Customer churn information
- `customer_data.csv` - Customer demographics
- `internet_data.csv` - Internet service details

**Features:**
- Customer demographics (gender, age, partner, dependents)
- Service information (phone, internet, streaming services)
- Contract details (type, billing method, payment method)
- Usage metrics (tenure, monthly charges, total charges)

---

## 🔧 Project Structure

```
Customer_Churn_Predictor/
│
├── Customer_Churn_Predictor.ipynb   # Main Jupyter notebook
├── app.py                           # Streamlit deployment app
├── requirements.txt                 # Python dependencies
├── churn_model.pkl                  # Trained model (generated)
├── scaler.pkl                       # Feature scaler (generated)
├── churn_data.csv                   # Dataset file 1
├── customer_data.csv                # Dataset file 2
├── internet_data.csv                # Dataset file 3
└── README.md                        # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Customer_Churn_Predictor
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required Libraries:**
- pandas==2.0.3
- numpy==1.24.3
- matplotlib
- seaborn
- scikit-learn==1.3.0
- imbalanced-learn
- streamlit==1.28.0
- joblib==1.3.2

---

## 📖 Step-by-Step Workflow

### 1. 🧠 Define the Problem
**Objective:** Predict whether a customer will churn based on their profile and service details.

**Problem Type:** Binary Classification
- Class 0: Customer will NOT churn
- Class 1: Customer WILL churn

### 2. 🗂️ Collect and Prepare Data

#### Data Loading:
```python
churn_df = pd.read_csv('churn_data.csv')
customer_df = pd.read_csv('customer_data.csv')
internet_df = pd.read_csv('internet_data.csv')

# Merge datasets
df = pd.merge(churn_df, customer_df, on='customerID', how='inner')
df = pd.merge(df, internet_df, on='customerID', how='inner')
```

#### Data Preprocessing:
- Handle missing values in `TotalCharges` (convert to numeric and fill with 0)
- Convert `Churn` column to binary (Yes=1, No=0)
- Label encoding for binary categorical features
- One-hot encoding for multi-category features
- Drop `customerID` column (not useful for prediction)

### 3. 📊 Exploratory Data Analysis (EDA)

**Visualizations Created:**
1. **Churn Rate Distribution** - Donut pie chart showing overall churn percentage
2. **Churn by Demographics** - Violin plots and bar charts for:
   - Gender
   - Senior Citizen status
   - Partner status
   - Dependents
3. **Churn by Contract Type** - Stacked bar charts for contract analysis
4. **Churn by Payment Method** - Multiple payment method comparisons
5. **Numerical Distributions** - KDE plots for:
   - Tenure
   - Monthly Charges
   - Total Charges
6. **Correlation Heatmap** - Feature correlation analysis

**Key Insights:**
- Churn rate: ~26.5% (imbalanced dataset)
- Senior citizens have higher churn rate (41.7% vs 23.6%)
- Customers without partners/dependents churn more
- Month-to-month contracts have highest churn
- Electronic check payment method associated with higher churn

### 4. 📐 Feature Engineering

**Tenure Grouping:**
```python
def tenure_group(tenure):
    if tenure <= 12:
        return '0-12 Months'
    elif 12 < tenure <= 24:
        return '12-24 Months'
    elif 24 < tenure <= 48:
        return '24-48 Months'
    elif 48 < tenure <= 60:
        return '48-60 Months'
    else:
        return '60+ Months'
```

### 5. 🔀 Split the Data

```python
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Scaling:**
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 6. ⚖️ Handle Class Imbalance

Using SMOTE (Synthetic Minority Oversampling Technique):
```python
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

### 7. 🤖 Model Training

**Three models trained:**

1. **Logistic Regression**
   ```python
   logreg = LogisticRegression(random_state=42)
   logreg.fit(X_train_resampled, y_train_resampled)
   ```

2. **Random Forest Classifier**
   ```python
   rf_model = RandomForestClassifier(random_state=42)
   rf_model.fit(X_train_resampled, y_train_resampled)
   ```

3. **Gradient Boosting Classifier**
   ```python
   gb_model = GradientBoostingClassifier(random_state=42)
   gb_model.fit(X_train_resampled, y_train_resampled)
   ```

### 8. 📈 Model Evaluation

**Metrics Used:**
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

**Model Performance:**

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7935 | 0.6781 | 0.4225 | 0.5206 | 0.8335 |
| Random Forest | 0.7906 | 0.6231 | 0.5348 | 0.5755 | 0.8391 |
| **Gradient Boosting** | 0.7750 | 0.5610 | 0.7005 | 0.6231 | **0.8441** |

**Best Model:** Gradient Boosting (highest AUC-ROC score)

### 9. 🔧 Hyperparameter Tuning

GridSearchCV applied to Logistic Regression:
```python
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42), 
    param_grid, 
    cv=3, 
    scoring='roc_auc'
)
```

**Best Parameters:** `{'C': 100, 'penalty': 'l2', 'solver': 'liblinear'}`

### 10. 💾 Save the Model

```python
import joblib
joblib.dump(best_logreg, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

---

## 🌐 Streamlit Web Application

### Running the App

1. **Ensure model files exist:**
   - `churn_model.pkl`
   - `scaler.pkl`

2. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

3. **Access the app:**
   - Open browser at `http://localhost:8501`

### App Features:

📱 **Interactive Input Sections:**
- 👤 Customer Information (gender, senior citizen, partner, dependents, tenure)
- 📞 Service Information (phone service, multiple lines, internet service)
- 🌐 Additional Services (security, backup, streaming)
- 💳 Billing Information (contract, payment method, paperless billing)
- 💰 Charges (monthly and total charges)

📊 **Prediction Output:**
- Churn risk indicator (High/Low)
- Churn probability percentage
- Visual progress bar
- Actionable recommendations

### Usage Example:

1. Fill in customer details using the interactive widgets
2. Click "🔮 Predict Churn" button
3. View prediction results and recommendations
4. Use insights to develop retention strategies

---

## 📈 Results & Insights

### Model Performance Summary:
- **Best AUC-ROC:** 0.8441 (Gradient Boosting)
- **Trade-off:** Gradient Boosting has better recall (70%) but lower precision (56%)
- **Business Impact:** Better at identifying churners (fewer false negatives)

### Key Churn Factors:
1. **Contract Type:** Month-to-month contracts have 3x higher churn
2. **Tenure:** New customers (0-12 months) are high risk
3. **Payment Method:** Electronic check users churn 45% more
4. **Senior Citizens:** 41% churn rate vs 23% for non-seniors
5. **Family Status:** Customers without partners/dependents churn more

### Business Recommendations:
- ✅ Encourage long-term contracts with incentives
- ✅ Focus retention efforts on new customers (first year)
- ✅ Promote automatic payment methods
- ✅ Offer special packages for senior citizens
- ✅ Create family/bundle plans to reduce churn

---

## 🛠️ Technologies Used

**Languages & Libraries:**
- Python 3.8+
- Pandas - Data manipulation
- NumPy - Numerical computing
- Matplotlib & Seaborn - Data visualization
- Scikit-learn - Machine learning
- Imbalanced-learn - SMOTE implementation
- Streamlit - Web app deployment
- Joblib - Model persistence

**Algorithms:**
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- SMOTE (Synthetic Minority Oversampling)
- GridSearchCV (Hyperparameter tuning)

---

## 📝 How to Use This Project

### For Data Scientists:
1. Open `Customer_Churn_Predictor.ipynb` in Jupyter Notebook
2. Run cells sequentially to reproduce the analysis
3. Experiment with different models and parameters
4. Modify visualizations for your needs

### For Business Users:
1. Run the Streamlit app: `streamlit run app.py`
2. Input customer details in the web interface
3. Get instant churn predictions
4. Use recommendations to develop retention strategies

### For Developers:
1. Load the saved model:
   ```python
   import joblib
   model = joblib.load('churn_model.pkl')
   scaler = joblib.load('scaler.pkl')
   ```
2. Prepare input data with same features
3. Make predictions:
   ```python
   prediction = model.predict(scaled_data)
   probability = model.predict_proba(scaled_data)
   ```

---

## 🔍 Troubleshooting

### Common Issues:

**1. Feature Mismatch Error:**
- **Error:** `ValueError: The feature names should match...`
- **Solution:** The app now automatically handles all expected features. Ensure you're using the latest `app.py` version.

**2. Module Not Found:**
- **Error:** `ModuleNotFoundError: No module named 'streamlit'`
- **Solution:** Install dependencies: `pip install -r requirements.txt`

**3. Model File Not Found:**
- **Error:** `FileNotFoundError: churn_model.pkl`
- **Solution:** Run the notebook first to generate model files, or ensure they're in the same directory as `app.py`

---

## 📊 Future Improvements

- [ ] Add more advanced models (XGBoost, LightGBM)
- [ ] Implement feature importance visualization in the app
- [ ] Add batch prediction capability
- [ ] Create REST API with FastAPI/Flask
- [ ] Deploy to cloud (Streamlit Cloud, Heroku, AWS)
- [ ] Add A/B testing framework
- [ ] Implement real-time data pipeline
- [ ] Add explainability with SHAP values

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available for educational purposes.

---

## 👤 Author

**Bipul**

- Project: Customer Churn Predictor
- Dataset: [Kaggle Telecom Customer Churn](https://www.kaggle.com/datasets/dileep070/logisticregression-telecomcustomer-churmprediction/data)

---

## Acknowledgments

- Kaggle for providing the dataset
- Scikit-learn community for excellent documentation
- Streamlit for easy web app deployment
- All contributors and supporters

---

## 📞 Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact me

---

**Built with ❤️ using Python, Scikit-learn, and Streamlit**

