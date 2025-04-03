# ANN-Based Customer Churn Prediction  

This project builds an **Artificial Neural Network (ANN)** model to predict customer churn based on financial, demographic, and behavioral attributes. The model is implemented using **TensorFlow & Keras** and deployed as an interactive dashboard with **Streamlit**.  

## 🚀 Features  

- **📂 Upload Dataset**: Supports CSV file upload for analysis.  
- **🔍 Data Preprocessing**:  
  - Encoding categorical variables (Gender, Geography).  
  - Scaling numerical features.  
  - Splitting data into training and testing sets.  
- **🎛 Hyperparameter Tuning**: Adjustable neurons, dropout rate, epochs, and batch size.  
- **📊 Model Training & Evaluation**:  
  - Model summary display.  
  - Accuracy & loss plots.  
  - Confusion matrix heatmap.  
  - Precision, Recall, F1-score plots.  
  - Class distribution insights.  

## 🏗 How to Run the Project Locally  

### 1️⃣ Install Dependencies  
Ensure you have **Python 3.8+** and install the required libraries:  

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:  

```bash
pip install streamlit tensorflow pandas numpy scikit-learn matplotlib seaborn
```

### 2️⃣ Run the Streamlit App  

```bash
streamlit run app.py
```

This will open the dashboard in your browser.  

## 🌍 Live Demo  

🚀 **Streamlit Dashboard Link** *(Replace with your actual Streamlit deployment URL.)*  

## 📁 Project Structure  

```bash
📂 ANN-Churn-Prediction
 ┣ 📜 app.py              # Streamlit app code  
 ┣ 📜 README.md           # Documentation  
 ┣ 📜 requirements.txt    # Python dependencies  
 ┣ 📂 data                # Folder for sample dataset (if any)  
```

## 🛠 Key Technologies Used  

- **TensorFlow & Keras** – ANN model development  
- **Scikit-Learn** – Data preprocessing & evaluation metrics  
- **Pandas & NumPy** – Data handling and computation  
- **Matplotlib & Seaborn** – Data visualization  
- **Streamlit** – Interactive UI for model deployment  

## 🎯 Next Steps & Improvements  

- ✅ Enhance feature engineering techniques  
- ✅ Deploy model with API integration  
- ✅ Optimize ANN performance with advanced hyperparameter tuning  

## 💡 Contributors  

👤 Your Name  
📧 Your Email  
🔗 [LinkedIn](#)  
