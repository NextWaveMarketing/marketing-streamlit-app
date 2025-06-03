
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

st.title("Customer Response Prediction App")

uploaded_file = st.file_uploader("Upload CSV (Comma or Tab Separated)", type="csv")

if uploaded_file is not None:
    # Automatically detect delimiter (comma or tab)
    df = pd.read_csv(uploaded_file, sep=None, engine='python')

    # Clean and normalize column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

    st.write("Columns in uploaded data:", df.columns.tolist())

    # Show preview
    st.write("Preview of Data:")
    st.write(df.head())

    # Convert date column if present
    if 'Dt_Customer' in df.columns:
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
        df['Customer_Since_Days'] = (pd.to_datetime('today') - df['Dt_Customer']).dt.days
        df.drop('Dt_Customer', axis=1, inplace=True)

    df.dropna(inplace=True)

    # Let user select the target column dynamically
    target_col = st.selectbox("Select the target column to predict", options=df.columns)

    if target_col:
        y = df[target_col]
        X = df.drop(target_col, axis=1)

        # Apply dummy encoding only on features
        X = pd.get_dummies(X)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # 🎯 Model selection
        model_option = st.selectbox("Choose the model", ["Logistic Regression", "Random Forest"])

        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_option == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
