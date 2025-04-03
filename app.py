
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import io

# Title
st.title("ANN Churn Prediction Dashboard")

# File Upload
uploaded_file = st.file_uploader("clean_train.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Dataset:")
    st.dataframe(data.head())

    # Encoding Categorical Variables
    if 'Geography' in data.columns:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_geography = encoder.fit_transform(data[['Geography']])
        geography_columns = encoder.get_feature_names_out(['Geography'])
        df_encoded = pd.DataFrame(encoded_geography, columns=geography_columns)
        data = pd.concat([data, df_encoded], axis=1)
        data.drop(columns=['Geography'], inplace=True)
    else:
        st.error("Column 'Geography' not found in the dataset.")

    # Encoding Gender
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    else:
        st.error("Column 'Gender' not found in the dataset.")

    # Scaling
    scaler = MinMaxScaler()
    num_cols = data.select_dtypes(include=['number']).columns
    data[num_cols] = scaler.fit_transform(data[num_cols])

    # Train-Test Split
    if 'Exited' in data.columns:
        X = data.drop(columns=['Exited'])
        y = data['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        st.error("Column 'Exited' not found in the dataset.")

    # Sidebar: Hyperparameter Tuning
    st.sidebar.header("Hyperparameters")
    neurons = st.sidebar.slider("Number of Neurons", min_value=8, max_value=128, step=8, value=64)
    dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, step=0.1, value=0.2)
    epochs = st.sidebar.slider("Epochs", min_value=10, max_value=100, step=10, value=50)
    batch_size = st.sidebar.slider("Batch Size", min_value=8, max_value=128, step=8, value=32)

    # Model Building
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(neurons//2, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)

    # Model Summary
    st.subheader("Model Summary")
    model_summary = io.StringIO()
    model.summary(print_fn=lambda x: model_summary.write(x + '\n'))
    st.text(model_summary.getvalue())

    # Accuracy and Loss Plot
    st.subheader("Accuracy & Loss")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].legend()
    ax[0].set_title('Model Accuracy')

    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].legend()
    ax[1].set_title('Model Loss')

    st.pyplot(fig)

    # Confusion Matrix
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Precision, Recall, and F1-Score Plot
    st.subheader("Precision, Recall, and F1-Score")
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    metrics_df = pd.DataFrame({'Metric': ['Precision', 'Recall', 'F1-Score'], 'Value': [precision, recall, f1_score]})
    st.bar_chart(metrics_df.set_index('Metric'))

    # Class Distribution Plot
    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=y_train, palette="coolwarm")
    ax.set_title("Training Data Class Distribution")
    ax.set_xticklabels(["No Churn", "Churn"])
    st.pyplot(fig)

    # Model Training and Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader(f"Model Accuracy: {accuracy:.2f}")
