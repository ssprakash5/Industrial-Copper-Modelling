import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv("/content/IndustryCopperCleaned.csv")

# Preprocessing
label_encoders = {}
categorical_columns = ['country', 'application', 'item type_encoded', 'product_ref_encoded', 'material_ref_encoded', 'customer_encoded']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Splitting the data into features and target
selling_price_cols = [
    'quantity tons', 'country', 'application', 'thickness', 'width',
    'id_encoded', 'status_encoded', 'item type_encoded', 'product_ref_encoded',
    'delivery_period', 'material_ref_encoded', 'customer_encoded'
]

X_selling_price = df[selling_price_cols]
Y_selling_price = df['selling_price']

status_cols = [
    'quantity tons', 'country', 'application', 'thickness', 'width',
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

# Streamlit app
def predict_selling_price(params):
    prediction_dt = dt_regressor.predict(params)
    prediction_rf = rf_regressor.predict(params)
    prediction_xgb = xgb_regressor.predict(params)
    return prediction_dt, prediction_rf, prediction_xgb

def predict_status(params):
    prediction = rf_classifier_status.predict(params)
    return prediction

def main():
    st.title('Selling Price and Status Predictor')

    # Input parameters for selling price prediction
    st.sidebar.title('Selling Price Prediction')
    sp_params = {}  # Fill this with the parameters
    for idx, col in enumerate(selling_price_cols):
        sp_params[col] = st.sidebar.selectbox(f"Select {col}", df[col].unique(), key=f"sp_{col}_{idx}")

    # Input parameters for status prediction
    st.sidebar.title('Status Prediction')
    st_params = {}  # Fill this with the parameters
    for idx, col in enumerate(status_cols):
        st_params[col] = st.sidebar.selectbox(f"Select {col}", df[col].unique(), key=f"st_{col}_{idx}")

    # Button to trigger predictions
    if st.sidebar.button('Predict'):
        sp_input = pd.DataFrame(sp_params, index=[0])
        st_input = pd.DataFrame(st_params, index=[0])

        # Predictions
        dt_sp, rf_sp, xgb_sp = predict_selling_price(sp_input)
        status_pred = predict_status(st_input)

        # Display predictions
        st.write('**Selling Price Prediction - Decision Tree:**', dt_sp)
        st.write('**Selling Price Prediction - Random Forest:**', rf_sp)
        st.write('**Selling Price Prediction - XGBoost:**', xgb_sp)
        st.write('**Status Prediction:**', status_pred)

if __name__ == '__main__':
    main()
