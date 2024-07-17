import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Set the Streamlit app title
st.title('Predictive Maintenance using Machine Learning')

# Sidebar for uploading pickle files
st.sidebar.title("Upload Pickle Files")
train_file = st.sidebar.file_uploader("Upload training dataset pickle (PM_train.pkl)", type=["pkl"])
train_updated_file = st.sidebar.file_uploader("Upload updated training dataset pickle (PM_train_updated.pkl)", type=["pkl"])
test_file = st.sidebar.file_uploader("Upload test dataset pickle (PM_test.pkl)", type=["pkl"])
truth_file = st.sidebar.file_uploader("Upload truth dataset pickle (Truth_Value.pkl)", type=["pkl"])

@st.cache
def load_pickle(file):
    return pickle.load(file)

if train_file and train_updated_file and test_file and truth_file:
    # Load the datasets from pickle files
    df_train = load_pickle(train_file)
    df_train_updated = load_pickle(train_updated_file)
    df_test = load_pickle(test_file)
    df_truth = load_pickle(truth_file)

    # Data preparation steps
    rul = pd.DataFrame(df_test.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']

    df_truth['rtf'] = df_truth['more'] + rul['max']
    df_truth.drop('more', axis=1, inplace=True)
    df_test = df_test.merge(df_truth, on=['id'], how='left')
    df_test['ttf'] = df_test['rtf'] - df_test['cycle']
    df_test.drop('rtf', axis=1, inplace=True)

    df_train_updated['ttf'] = df_train_updated.groupby(['id'])['cycle'].transform(max) - df_train_updated['cycle']

    features_col_name = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 
                         's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    target_col_name = 'ttf'

    # Normalize the features
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    df_train_updated[features_col_name] = sc.fit_transform(df_train_updated[features_col_name])
    df_test[features_col_name] = sc.transform(df_test[features_col_name])

    X_train = df_train_updated[features_col_name]
    y_train = df_train_updated[target_col_name]
    X_test = df_test[features_col_name]
    y_test = df_test[target_col_name]

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate the model
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)

    y_test = y_test.fillna(y_test.mean())
    y_pred_test = np.nan_to_num(y_pred_test, nan=np.nanmean(y_pred_test))

    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    # Display evaluation results
    st.write("Train set evaluation:")
    st.write('Mean Squared Error:', train_mse)
    st.write('R-squared:', train_r2)

    st.write("\nTest set evaluation:")
    st.write('Mean Squared Error:', test_mse)
    st.write('R-squared:', test_r2)

    # Visualization
    st.subheader('Predicted vs True Time to Failure (TTF)')
    fig, ax = plt.subplots()
    ax.plot(y_test.values, label='True TTF')
    ax.plot(y_pred_test, label='Predicted TTF')
    ax.legend()
    st.pyplot(fig)

    # Output the result for the maintenance requirement
    last_prediction = y_pred_test[-1] if len(y_pred_test) > 0 else None
    if last_prediction is not None:
        st.write(f"The machine will require maintenance after approximately {last_prediction:.2f} days.")
    else:
        st.write("No predictions available to determine maintenance requirement.")
else:
    st.write("Please upload all required files.")
