# Predictive Customer Churn Analysis | Diagnosing Risk and Driving Retention Strategy  

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-017CEE?style=for-the-badge&logo=xgboost&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-8A4182?style=for-the-badge)


---

**🎯 Problem Statement**  
A telecom company is experiencing high customer churn but lacks a clear understanding of why customers are leaving and which customers are most likely to churn next. This leads to inefficient retention efforts, wasted marketing spend, and continued revenue loss.  

---

**🚀 Solution & Objectives**  
I developed an end-to-end predictive analytics solution to diagnose the root causes of churn and identify high-risk customers. The primary objectives were:  

- **Diagnose:** Uncover the key factors driving customer churn.  
- **Predict:** Build a machine learning model to accurately score each customer's churn risk.  
- **Actionize:** Provide a data-driven framework for the retention team to prioritize interventions.
   
---

**🛠️ Tools Used**  
- **Python:** Pandas, NumPy, Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn (RandomForest, LogisticRegression, GridSearchCV), XGBoost, SHAP, imbalanced-learn (SMOTE)  
- **Environment:** Google Colab  

---

**🔍 Key Insights from EDA & Modeling**  
- **Contract Type is King:** Month-to-month customers are over 300% more likely to churn than those on 2-year contracts.  
- **The Fiber Optic Paradox:** Customers with Fiber optic internet have a churn rate ~40% higher than DSL customers.  
- **Tenure is Protective:** Customers with less than 6 months tenure are ~4x more likely to churn than veterans with over 2 years.  

---

**📈 Methodology & Process**  
- **Data Acquisition & Cleaning:** Sourced customer data from a public Telco dataset. Engineered new features like avg_monthly_calc and handled missing values in TotalCharges.  
- **Exploratory Data Analysis (EDA):** Uncovered initial insights into key churn drivers like contract type and internet service.  
- **Feature Engineering & Preprocessing:** Built a robust pipeline to handle numeric and categorical features, including imputation, scaling, and one-hot encoding.  
- **Predictive Modeling:** Trained and evaluated multiple models. A RandomForest classifier was optimized via GridSearch, achieving superior performance.  
- **Model Interpretation:** Used Permutation Importance and SHAP (SHapley Additive exPlanations) to explain the model's reasoning for each customer.  
- **Deployment & Actionability:** Scored customers and segmented them into risk deciles, enabling prioritized intervention.  

---

**💡 Results & Business Impact**  
The analysis provided clear, actionable insights that could directly inform customer retention strategy:  

- **Built a High-Performance Predictor:** The optimized RandomForest model achieved an ROC AUC of 0.85 and Average Precision of 0.65 on the test set.  
- **Identified Key Drivers:** SHAP analysis revealed that contract type was the strongest predictor of churn. Higher monthly charges and the presence of Fiber optic service were also major contributors.  
- **Enabled Proactive Retention:** The model successfully concentrated risk: Over 50% of all actual churn events came from the top 2 deciles (20% of customers), enabling a 2.5x increase in retention campaign efficiency.  

---

**Strategic Recommendations**  

- **🎯 Targeted Interventions:** Prioritize outreach to high-risk customers identified by the model (e.g., those in the top decile with month-to-month contracts).  
- **📦 Product Bundling:** Address the churn risk associated with Fiber optic by creating promoted bundles that include Tech Support or Online Security.  
- **🎁 Loyalty Programs:** Incentivize customers on month-to-month plans to switch to longer-term contracts  
