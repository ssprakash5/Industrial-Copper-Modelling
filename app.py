import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data efficiently using caching
@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})
def load_data():
    df = pd.read_csv("/content/drive/MyDrive/IndustryCopperCleaned.csv")
    columns_to_convert_to_str = ['country', 'application', 'item type', 'material_ref', 'customer']
    df[columns_to_convert_to_str] = df[columns_to_convert_to_str].astype(str)
    
    label_encoders = {}
    for col in columns_to_convert_to_str + ['id', 'product_ref', 'status']:
        label_encoders[col] = LabelEncoder()
        df[col + '_encoded'] = label_encoders[col].fit_transform(df[col])

    categorical_columns = ['country', 'application', 'item type', 'product_ref', 'material_ref', 'customer', 'id', 'status']
    for col in categorical_columns:
        df[col + '_encoded'] = label_encoders[col].transform(df[col])

    return df, label_encoders

# Train models
def train_models(df, label_encoders):
    selling_price_cols = [
        'quantity tons', 'country_encoded', 'application_encoded', 'thickness', 'width',
        'id_encoded', 'status_encoded', 'item type_encoded', 'product_ref_encoded',
        'delivery_period', 'material_ref_encoded', 'customer_encoded'
    ]
    
    X_selling_price = df[selling_price_cols]
    Y_selling_price = df['selling_price']
    
    status_cols = [
        'quantity tons', 'country_encoded', 'application_encoded', 'thickness', 'width',
        'id_encoded', 'item type_encoded', 'product_ref_encoded',
        'delivery_period', 'material_ref_encoded', 'customer_encoded',
        'selling_price'
    ]
    
    X_status = df[status_cols]
    Y_status = df['status_encoded']
    
    X_train_sp, X_test_sp, Y_train_sp, Y_test_sp = train_test_split(X_selling_price, Y_selling_price, test_size=0.2, random_state=42)
    X_train_st, X_test_st, Y_train_st, Y_test_st = train_test_split(X_status, Y_status, test_size=0.2, random_state=42)
    
    dt_regressor = DecisionTreeRegressor()
    dt_regressor.fit(X_train_sp, Y_train_sp)

    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_train_sp, Y_train_sp)

    xgb_regressor = XGBRegressor()
    xgb_regressor.fit(X_train_sp, Y_train_sp)

    rf_classifier_status = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier_status.fit(X_train_st, Y_train_st)

    return dt_regressor, rf_regressor, xgb_regressor, rf_classifier_status

# Inverse Transformation function for 'status'
def inverse_transform_status(encoded_status, label_encoder):
    return label_encoder.inverse_transform(encoded_status)

# Streamlit app
def main():
    st.title('Selling Price and Status Predictor')

    # Load data
    df, label_encoders = load_data()

    # Train models
    dt_regressor, rf_regressor, xgb_regressor, rf_classifier_status = train_models(df, label_encoders)

    # Define columns within the main function
    selling_price_cols = [
        'quantity tons', 'country_encoded', 'application_encoded', 'thickness', 'width',
        'id_encoded', 'status_encoded', 'item type_encoded', 'product_ref_encoded',
        'delivery_period', 'material_ref_encoded', 'customer_encoded'
    ]

    status_cols = [
        'quantity tons', 'country_encoded', 'application_encoded', 'thickness', 'width',
        'id_encoded', 'item type_encoded', 'product_ref_encoded',
        'delivery_period', 'material_ref_encoded', 'customer_encoded',
        'selling_price'
    ]

    # Selectbox for choosing the ML model
    selected_model = st.sidebar.selectbox("Select ML Model", ["Decision Tree", "Random Forest", "XGBoost", "Random Forest Classifier"])

    # Input parameters for selling price prediction
    st.sidebar.title('Selling Price Prediction')
    sp_params = {}  # Fill this with the parameters
    for idx, col in enumerate(selling_price_cols):
        if col.endswith('_encoded'):
            original_col = col.replace('_encoded', '')
            selected_value = st.sidebar.selectbox(f"Select {original_col}", df[original_col].unique(), key=f"sp_{original_col}_{idx}")
            sp_params[col] = label_encoders[original_col].transform([selected_value])[0]
        elif col == 'quantity tons' or col == 'thickness' or col == 'width' or col == 'delivery_period' or col == 'selling_price':
            sp_params[col] = st.sidebar.slider(f"Select {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()), key=f"sp_{col}_{idx}")
        else:
            sp_params[col] = st.sidebar.selectbox(f"Select {col}", df[col].unique(), key=f"sp_{col}_{idx}")

    # Input parameters for status prediction
    st.sidebar.title('Status Prediction')
    st_params = {}  # Fill this with the parameters
    for idx, col in enumerate(status_cols):
        if col.endswith('_encoded'):
            original_col = col.replace('_encoded', '')
            selected_value = st.sidebar.selectbox(f"Select {original_col}", df[original_col].unique(), key=f"st_{original_col}_{idx}")
            st_params[col] = label_encoders[original_col].transform([selected_value])[0]
        elif col == 'quantity tons' or col == 'thickness' or col == 'width' or col == 'delivery_period' or col == 'selling_price':
            st_params[col] = st.sidebar.slider(f"Select {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()), key=f"st_{col}_{idx}")
        else:
            st_params[col] = st.sidebar.selectbox(f"Select {col}", df[col].unique(), key=f"st_{col}_{idx}")

    if st.sidebar.button('Predict'):
        sp_input = pd.DataFrame(sp_params, index=[0])
        st_input = pd.DataFrame(st_params, index=[0])

        # Predictions based on the selected model
        if selected_model == "Decision Tree":
            sp_pred = dt_regressor.predict(sp_input)
            st.write('**Selling Price Prediction - Decision Tree:**', sp_pred, font="Arial", line_width=800)
        elif selected_model == "Random Forest":
            sp_pred = rf_regressor.predict(sp_input)
            st.write('**Selling Price Prediction - Random Forest:**', sp_pred, font="Arial", line_width=800)
        elif selected_model == "XGBoost":
            sp_pred = xgb_regressor.predict(sp_input)
            st.write('**Selling Price Prediction - XGBoost:**', sp_pred, font="Arial", line_width=800)
        elif selected_model == "Random Forest Classifier":
            status_pred_encoded = rf_classifier_status.predict(st_input)
            status_pred_original = inverse_transform_status(status_pred_encoded, label_encoders['status'])
            st.write('**Status Prediction:**', status_pred_original, font="Arial", line_width=800)

if __name__ == '__main__':
    main()
