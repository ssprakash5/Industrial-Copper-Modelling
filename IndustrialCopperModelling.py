import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv("/content/IndustryCopperCleaned.csv")
# Convert specified columns to strings
columns_to_convert_to_str = ['country', 'application', 'item type', 'material_ref', 'customer']
df[columns_to_convert_to_str] = df[columns_to_convert_to_str].astype(str)

# Perform encoding
label_encoders = {}
for col in columns_to_convert_to_str + ['id', 'product_ref', 'status']:  # Include 'id', 'product_ref', and 'status' in columns to encode
    label_encoders[col] = LabelEncoder()
    df[col + '_encoded'] = label_encoders[col].fit_transform(df[col])

# Preprocessing
categorical_columns = ['country', 'application', 'item type', 'product_ref', 'material_ref', 'customer', 'id', 'status']

for col in categorical_columns:
    df[col + '_encoded'] = label_encoders[col].transform(df[col])  # Reuse the existing label encoders

# Splitting the data into features and target
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

# Splitting the data into training and testing sets
X_train_sp, X_test_sp, Y_train_sp, Y_test_sp = train_test_split(X_selling_price, Y_selling_price, test_size=0.2, random_state=42)
X_train_st, X_test_st, Y_train_st, Y_test_st = train_test_split(X_status, Y_status, test_size=0.2, random_state=42)

# Train models
dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(X_train_sp, Y_train_sp)

rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train_sp, Y_train_sp)

xgb_regressor = XGBRegressor()
xgb_regressor.fit(X_train_sp, Y_train_sp)

rf_classifier_status = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_status.fit(X_train_st, Y_train_st)

# Inverse Transformation function for 'status'
def inverse_transform_status(encoded_status):
    return label_encoders['status'].inverse_transform(encoded_status)

# Streamlit app
def main():
    st.title('Selling Price and Status Predictor')

    # Input parameters for selling price prediction
    st.sidebar.title('Selling Price Prediction')
    sp_params = {}  # Fill this with the parameters
    for idx, col in enumerate(selling_price_cols):
        if col.endswith('_encoded'):  # Check if the column is encoded
            original_col = col.replace('_encoded', '')  # Get the original column name
            selected_value = st.sidebar.selectbox(f"Select {original_col}", df[original_col].unique(), key=f"sp_{original_col}_{idx}")
            sp_params[col] = label_encoders[original_col].transform([selected_value])[0]  # Encode the selected value
        else:
            sp_params[col] = st.sidebar.selectbox(f"Select {col}", df[col].unique(), key=f"sp_{col}_{idx}")

    # Input parameters for status prediction
    st.sidebar.title('Status Prediction')
    st_params = {}  # Fill this with the parameters
    for idx, col in enumerate(status_cols):
        if col.endswith('_encoded'):  # Check if the column is encoded
            original_col = col.replace('_encoded', '')  # Get the original column name
            selected_value = st.sidebar.selectbox(f"Select {original_col}", df[original_col].unique(), key=f"st_{original_col}_{idx}")
            st_params[col] = label_encoders[original_col].transform([selected_value])[0]  # Encode the selected value
        else:
            st_params[col] = st.sidebar.selectbox(f"Select {col}", df[col].unique(), key=f"st_{col}_{idx}")

    if st.sidebar.button('Predict'):
        sp_input = pd.DataFrame(sp_params, index=[0])
        st_input = pd.DataFrame(st_params, index=[0])

        # Predictions
        dt_sp = dt_regressor.predict(sp_input)
        rf_sp = rf_regressor.predict(sp_input)
        xgb_sp = xgb_regressor.predict(sp_input)
        status_pred_encoded = rf_classifier_status.predict(st_input)

        # Inverse transform 'status' predictions
        status_pred_original = inverse_transform_status(status_pred_encoded)

        # Display predictions
        st.write('**Selling Price Prediction - Decision Tree:**', dt_sp)
        st.write('**Selling Price Prediction - Random Forest:**', rf_sp)
        st.write('**Selling Price Prediction - XGBoost:**', xgb_sp)
        st.write('**Status Prediction:**', status_pred_original)

if __name__ == '__main__':
    main()
