# Predicting High-Potential Sellers at the E-Commerce Onboarding Stage Using Machine Learning

**A Machine Learning Decision-Support Model for Seller Onboarding Risk Prediction**

## 1. Project Overview

This Machine Learning Capstone explored how supervised classification could support seller onboarding decisions in an e-commerce marketplace context such as Flipkart.

The objective was to develop a data-driven approach for identifying potential onboarding risks, comparing classification models and translating model outputs into a proposed risk-prioritisation framework for seller acquisition managers.

Since proprietary Flipkart seller data was unavailable, the project used the IBM HR Analytics Employee Attrition dataset as a proxy to demonstrate the analytical methodology. The project combined data preprocessing, exploratory data analysis, model benchmarking, threshold optimisation, feature importance analysis and a proposed CRM-integrated decision-support workflow.

## 2. The Business Problem

Seller onboarding involves evaluating multiple factors, including financial capacity, category fit, business maturity, engagement and operational readiness. When these assessments depend heavily on individual judgement, decision consistency and early identification of potential risks can become challenging.

The project explored the following question:

**Can machine learning provide a structured, risk-based decision-support approach for evaluating prospective sellers before onboarding resources are committed?**

The proposed approach was designed to complement managerial judgement, not replace it. Its intended business application was to help prioritise evaluations, identify cases requiring closer review and establish a more structured approach to early risk monitoring.

## 3. My Key Learnings and Takeaways

### 3.1 Translating a Business Problem into a Machine Learning Problem

One of my most important learnings was how to convert a broad managerial challenge into a clearly defined analytical problem.

The project framed seller success and failure as a supervised binary classification task. This required identifying a target variable, selecting relevant predictors, establishing an evaluation approach and connecting model outputs to a practical decision.

**Key learning:** Machine learning projects should begin with a clearly defined business decision. The algorithm is only one part of the solution; the problem formulation determines whether the model can produce useful insights.

### 3.2 Working with Data Constraints and Proxy Datasets

Actual Flipkart seller data was unavailable because of confidentiality constraints. The project therefore used the IBM HR Analytics Employee Attrition dataset, containing 1,470 records and 35 variables, including 34 predictors and one target variable.

Employee attrition was used as a proxy for seller failure, while employee retention represented seller success. Variables such as monthly income, work experience, job satisfaction and overtime behaviour were explored as structural proxies for financial capacity, business maturity, engagement and operational pressure.

This approach allowed the team to demonstrate the modelling pipeline without access to proprietary seller data.

**Key learning:** Data availability is a real constraint in applied analytics. A proxy dataset can help demonstrate methodology, but it cannot establish that the same relationships or predictive performance will hold in a different business context. Actual seller data would be required to validate the model for Flipkart.

### 3.3 Data Preparation and Exploratory Data Analysis

The project involved examining the dataset, understanding the target distribution, reviewing variables and preparing the data for modelling.

The preprocessing pipeline included:

- Exploratory data analysis
- Feature and variable assessment
- Removal of constant or non-informative columns
- Label encoding of categorical variables
- Standard scaling where required
- Stratified train-test splitting
- Preparation for cross-validation

The dataset was imbalanced, with approximately 16.1% of observations belonging to the minority attrition class.

**Key learning:** Data preparation is not a preliminary formality. The quality of preprocessing, the distribution of the target variable and the treatment of categorical features all influence model evaluation and interpretation.

### 3.4 Comparing Multiple Classification Models

The project benchmarked three supervised learning approaches:

- **Logistic Regression:** Used as a baseline to establish a reference for classification performance.
- **Random Forest:** Evaluated for its ability to capture non-linear relationships and provide feature importance estimates.
- **Support Vector Machine (RBF kernel):** Explored for its ability to model more complex decision boundaries.

The models were evaluated using precision, recall, F1-score, ROC-AUC and five-fold cross-validation.

**Key learning:** Selecting a model requires more than comparing accuracy. Different algorithms offer different trade-offs in terms of predictive performance, interpretability and operational usefulness.

### 3.5 Understanding Class Imbalance and Evaluation Metrics

The minority class represented approximately 16.1% of the proxy dataset. In such settings, a model can achieve apparently strong accuracy while failing to identify a meaningful proportion of the cases that matter most.

The project therefore focused on metrics that provide a more informative view of minority-class performance.

- **Precision:** How many cases flagged as at risk were actually positive cases in the proxy dataset?
- **Recall:** How many actual positive cases did the model identify?
- **F1-score:** How well did the model balance precision and recall?
- **ROC-AUC:** How effectively did the model distinguish between the two classes across decision thresholds?

Class weighting and threshold optimisation were used to address the imbalance-related challenges.

**Key learning:** Evaluation metrics should reflect the consequences of different types of errors. In a risk-screening context, missing a genuinely at-risk case may have different business implications from flagging a case that subsequently performs well.

### 3.6 Threshold Optimisation and Decision-Making

The project demonstrated that a model's default classification threshold does not necessarily provide the most useful business outcome.

For the recommended Random Forest configuration, the decision threshold was adjusted to 0.30 rather than relying on the default threshold of 0.50.

This improved the model's ability to identify minority-class cases compared with the default Random Forest configuration.

**Key learning:** Model probabilities and business decisions are not the same thing. Threshold selection should reflect the intended use case, the relative costs of false positives and false negatives, and the capacity of the organisation to investigate flagged cases.

The selected threshold was a modelling choice for this academic experiment, not a validated operational threshold for Flipkart.

### 3.7 Interpreting Model Outputs Through Feature Importance

The Random Forest analysis identified variables such as monthly income, age, total working years, years at company and overtime behaviour among the more influential predictors in the proxy dataset.

The project explored how these variables could be translated into broader business concepts such as financial capacity, experience, engagement and operational pressure.

**Key learning:** Predictive modelling becomes more useful for management when its outputs can be interpreted and connected to a decision framework. However, feature importance indicates predictive contribution within the trained model; it does not establish causation or prove that the same variables predict seller success.

### 3.8 Translating Analytics into a Business Decision-Support Framework

The project moved beyond model comparison to propose how risk scores could inform seller evaluation.

The proposed framework included:

- Generating a risk probability score
- Categorising cases into Green, Yellow and Red risk bands
- Prioritising cases requiring additional review
- Integrating risk indicators into a proposed CRM workflow
- Establishing a feedback loop to evaluate outcomes over time

The framework also considered a future validation process using actual seller data and early-performance monitoring.

**Key learning:** A model creates business value only when its outputs inform a clear action. The decision-support layer must explain who reviews a flagged case, what additional checks are required and how subsequent outcomes are used to improve the process.

## 4. Methodology

The project followed a structured machine learning workflow.

### Step 1: Problem Formulation

Defined the seller onboarding challenge as a binary classification problem focused on identifying potential failure risk.

### Step 2: Dataset Selection

Selected the IBM HR Analytics Employee Attrition dataset as a proxy because proprietary Flipkart seller data was unavailable.

### Step 3: Data Preparation

Performed exploratory analysis, encoded categorical variables, scaled relevant features and prepared a stratified train-test split.

### Step 4: Model Development

Trained and compared Logistic Regression, Random Forest and Support Vector Machine classifiers.

### Step 5: Model Evaluation

Evaluated the models using precision, recall, F1-score, ROC-AUC and five-fold cross-validation.

### Step 6: Threshold Optimisation

Examined different classification thresholds to understand the trade-off between identifying at-risk cases and reducing false positives.

### Step 7: Business Interpretation

Reviewed feature importance and developed a proposed risk-banding and CRM-integrated decision-support framework.

## 5. Model Results

The report recommended a Random Forest configuration with a classification threshold of 0.30.

| Evaluation metric | Reported result |
|---|---:|
| Recall | 59.57% |
| Precision | 42.11% |
| F1-score | 0.4912 |
| ROC-AUC | 0.7804 |
| Five-fold cross-validation AUC | 0.7943 ± 0.0241 |

These results reflect evaluation on the IBM HR Analytics proxy dataset.

The threshold-adjusted Random Forest configuration was recommended for its balance of minority-class detection, predictive discrimination and interpretability. The project also evaluated an RBF-kernel SVM, which achieved a higher F1-score in the reported comparison. The final model choice therefore reflected the team's preference for a combination of performance, interpretability and potential operational flexibility rather than the highest score on every metric.

**Important limitation:** These results do not establish how accurately the model would predict actual Flipkart seller outcomes. That would require relevant seller-level training data, appropriate target labels and validation on unseen seller cohorts.

## 6. Proposed Business Application

The intended application was a decision-support layer for seller acquisition and onboarding.

A proposed workflow would be:

1. Capture relevant seller information during evaluation.
2. Generate a risk score after validating the model on suitable seller data.
3. Assign an indicative risk category.
4. Route higher-risk cases for additional human review.
5. Record onboarding decisions and subsequent seller performance.
6. Evaluate outcomes over time and recalibrate the approach when sufficient data becomes available.

The project proposed three risk bands:

- **Green:** Lower predicted risk under the selected framework.
- **Yellow:** Intermediate predicted risk, potentially requiring additional checks.
- **Red:** Higher predicted risk, requiring closer review.

The exact cut-offs would need to be established and validated using real seller data before operational use. The framework is a proposed design, not a production system deployed at Flipkart.

## 7. Key Project Outcomes

The capstone produced the following analytical and decision-support deliverables:

- A structured formulation of seller onboarding risk as a supervised classification problem.
- A documented approach for using a proxy dataset under data-access constraints.
- A preprocessing and model evaluation pipeline implemented in Python.
- A comparative evaluation of three classification algorithms.
- A threshold optimisation exercise focused on class imbalance.
- A feature importance analysis to support model interpretation.
- A proposed risk-banding framework for prioritising seller reviews.
- A conceptual CRM-integrated workflow for future validation and monitoring.

The main outcome was a demonstration of how machine learning methodology can be connected to a business decision, while clearly recognising the evidence required before deployment.

## 8. Tools and Technologies

- **Python:** Core programming and analytical workflow.
- **Pandas:** Data manipulation and analysis.
- **NumPy:** Numerical computation.
- **Scikit-learn:** Data preprocessing, classification models, cross-validation and evaluation metrics.
- **Matplotlib:** Data visualisation.
- **Seaborn:** Exploratory data analysis and statistical visualisation.
- **Jupyter/VS Code-compatible Python workflow:** Model development and execution.

## 9. Skills Developed

**Machine Learning:** Supervised learning, binary classification, Logistic Regression, Random Forest, Support Vector Machines and threshold optimisation.

**Data Analytics:** Exploratory data analysis, preprocessing, categorical encoding, scaling, stratified sampling and cross-validation.

**Model Evaluation:** Precision, recall, F1-score, ROC-AUC, confusion matrices and imbalanced classification.

**Business Analytics:** Translating a business problem into a model objective, interpreting model outputs and designing a decision-support workflow.

**Critical Thinking:** Evaluating data limitations, interpreting proxy-based results and distinguishing model performance from real-world business validation.

## 10. Limitations and Future Scope

The project's main limitation was the lack of proprietary seller onboarding and performance data. The IBM HR Analytics dataset was useful for demonstrating the methodology, but its features and target variable originate from a different domain.

Future work would require:

- Access to relevant seller onboarding and subsequent performance data.
- Definition of a business-appropriate seller success or failure outcome.
- Validation of the proposed features against actual seller characteristics.
- Training and evaluation on seller-specific data.
- Testing across time periods and unseen seller cohorts.
- Calibration of predicted probabilities and validation of risk thresholds.
- Human review, governance and monitoring before any operational implementation.

These steps would be necessary before the model could be considered suitable for real seller onboarding decisions.

## 11. Reflection

This project strengthened my understanding of how machine learning can support managerial decisions when the problem, data, evaluation criteria and intended action are clearly defined.

The most valuable learning was that **a technically sound model is not automatically a business-ready solution**. Data relevance, class imbalance, model interpretability, decision thresholds and the operational process around predictions all influence whether an analytical approach can be useful in practice.

The capstone helped me connect technical analysis with business problem-solving and understand the importance of validating model performance in the environment where it is intended to be used.

## 12. Project Details

**Project title:** Predicting High-Potential Sellers at the E-Commerce Onboarding Stage Using Machine Learning: A Decision-Support Model  
**Institution:** Fortune Institute of International Business (FIIB), New Delhi  
**Programme:** PGDM, 2025–2027  
**Course:** Machine Learning – Application in Management (MLAM)  
**Project type:** Academic team capstone  
**Industry context:** E-commerce seller acquisition and onboarding  
**Dataset:** IBM HR Analytics Employee Attrition dataset, used as a proxy  
**Primary focus:** Machine learning, risk prediction, model evaluation and business decision support

---
