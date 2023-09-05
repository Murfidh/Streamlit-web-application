import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import seaborn as sns

# Load the saved TensorFlow model
model_tf = tf.keras.models.load_model('/content/f_borrowing.h5')

# Load the pickle model for taxation
with open("/content/taxation.pkl", "rb") as file_tax:
    model_tax = pickle.load(file_tax)

# Load the pickle model for domestic borrowings
with open("/content/Domestic_borrowings_best.pkl", "rb") as file_dom:
    model_dom = pickle.load(file_dom)

# Load the pickle model for Money printing
with open("/content/money_printing_model.pkl", "rb") as file_dom:
    model_mon = pickle.load(file_dom)


# Load the pickle model for bebt sustainability
with open("/content/tot_debt_model.pkl", "rb") as file_debt:
    model_debt = pickle.load(file_debt)


#MongoDB connection for access the past dataset in order to make visualizations#
#------------------------------------------------------------------------------#

uri = "mongodb+srv://Research:research123@fip-2023-087.vekow81.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

db = client['test']
collections = db.list_collection_names()
GovernmentDebt_df = db['GovernmentDebt']
MoneyPrintingCollection_df = db['MoneyPrintingCollection']
ForeignBorrowingCollection_df = db['ForeignBorrowingCollection']
TaxationCollection_df = db['TaxationCollection']
DomesticBorrowingCollection_df = db['DomesticBorrowingCollection']

#Retrieving data from mongodb to induvidual dataframes
df_GovernmentDebt = pd.DataFrame(list(GovernmentDebt_df.find()))
df_MoneyPrintingCollection = pd.DataFrame(list(MoneyPrintingCollection_df.find()))
df_ForeignBorrowingCollection = pd.DataFrame(list(ForeignBorrowingCollection_df.find()))
df_TaxationCollection = pd.DataFrame(list(TaxationCollection_df.find()))
df_DomesticBorrowingCollection = pd.DataFrame(list(DomesticBorrowingCollection_df.find()))

#removing id column
df_MoneyPrintingCollection = df_MoneyPrintingCollection.drop('_id', axis=1)
df_ForeignBorrowingCollection = df_ForeignBorrowingCollection.drop('_id', axis=1)
df_TaxationCollection = df_TaxationCollection.drop('_id', axis=1)
df_DomesticBorrowingCollection = df_DomesticBorrowingCollection.drop('_id', axis=1)

#converting all the column data types into integer
df_MoneyPrintingCollection = df_MoneyPrintingCollection.astype('int64')
df_ForeignBorrowingCollection = df_MoneyPrintingCollection.astype('int64')
df_TaxationCollection = df_MoneyPrintingCollection.astype('int64')
df_DomesticBorrowingCollection = df_MoneyPrintingCollection.astype('int64')


#--------------------visualizations--------------------------------------------#





#------------------------------------------------------------------------------#



# Streamlit app title
st.title("ECONOINSIGHTS")

# First Section
st.header("1: Model Output")

# Date input for in order to make the prediction
pred_date = st.text_input("Select a Year for Prediction", value=2022)

input_data = pred_date
user_input_year_tax = int(pred_date)
user_input_year_dom = int(pred_date)
user_input_year_mon = int(pred_date)
user_input_year_debt = int(pred_date)



# Date input for YEAR (for TensorFlow model)
#input_data = st.sidebar.text_input("Select a Year for Foreign Borrowing Prediction")

# Text input for YEAR (for taxation model)
#user_input_year_tax = int(st.sidebar.number_input("Enter the Year for Taxation Forecasting"))


# Text input for YEAR (for domestic borrowings model)
#user_input_year_dom = int(st.sidebar.number_input("Enter the Year for Domestic Borrowings Forecasting"))

# Text input for YEAR (for Money printing model)
#user_input_year_mon = int(st.sidebar.number_input("Enter the Year for Money printing Forecasting"))

# Define custom CSS styles for the card
card_style_1 = """
         background-color: #581845;
         padding: px;
         border-radius: 4px;
         box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.2);
         margin-bottom: px;
"""


# Predict Foreign Borrowing using TensorFlow model
if input_data:
      given_year_tf = pd.to_datetime(input_data).timestamp()
      predicted_debt_tf = model_tf.predict(np.array([[given_year_tf]]))

      # Reshape the predicted debt to a 2D array
      predicted_debt_reshaped = predicted_debt_tf.reshape(-1, 1)

      # Load data from a pickle file
      with open('Foreign_model_scaler.pkl', 'rb') as file:
          minmax_scaler = pickle.load(file)

      # Inverse transform the predicted debt to get the original data
      original_predicted_debt = minmax_scaler.inverse_transform(predicted_debt_reshaped)

      predicted_value = original_predicted_debt[0][0]
      #st.write(f"Predicted Foreign Borrowing for {given_year_tf}: {predicted_value}")
      st.markdown(
        f'<div style="{card_style_1}">'
        f'<h5>Predicted Foreign Borrowing for {given_year_tf}: </h5>'
        f'<center><p><h3>{predicted_value}</h3></center></p>'
        '</div>',
        unsafe_allow_html=True
      )
      predicted_debt = predicted_value


# Predict Taxation using the pickle model
if user_input_year_tax:
    last_year_tax = 2021 - 2
    year_difference_tax = user_input_year_tax - last_year_tax
    periods_per_year_tax = 4
    forecast_horizon_tax = year_difference_tax * periods_per_year_tax
    prediction_tax = model_tax.forecast(forecast_horizon_tax)
    results_tax = prediction_tax.iloc[0]
    #st.write(f": ")
    st.markdown(
        f'<div style="{card_style_1}">'
        f'<h5>Predicted Taxation for {user_input_year_tax}</h5>'
        f'<center><p><h3>{results_tax:.2f}</h3></center></p>'
        '</div>',
        unsafe_allow_html=True
    )
    results = results_tax


# Predict Domestic Borrowings using the pickle model
if user_input_year_dom:
    start_year_dom = 2021
    forecast_range_dom = range(start_year_dom, user_input_year_dom + 1)
    forecast_dom = model_dom.forecast(steps=len(forecast_range_dom))
    forecast_dom.index = forecast_range_dom
    #st.write(f"Predicted Domestic Borrowings for {user_input_year_dom}: {forecast_dom.iloc[-1]:.2f}")
    st.markdown(
        f'<div style="{card_style_1}">'
        f'<h5>Predicted Domestic Borrowing for {user_input_year_dom}: </h5>'
        f'<center><p><h3>{forecast_dom.iloc[-1]:.2f}</h3></center></p>'
        '</div>',
        unsafe_allow_html=True
    )
    domestic_pred = forecast_dom.iloc[-1]



# Predict Money Printing using the pickle model
if user_input_year_mon:
    start_year_mon = 2022
    forecast_range_mon = range(start_year_mon, user_input_year_mon + 1)
    forecast_mon = model_mon.forecast(steps=len(forecast_range_mon))
    forecast_mon.index = forecast_range_mon
    #st.write(f"Predicted money printing for {user_input_year_mon}: {forecast_mon.iloc[-1]:.2f}")
    st.markdown(
        f'<div style="{card_style_1}">'
        f'<h5>Predicted Money Printing for {user_input_year_mon}: </h5>'
        f'<center><p><h3>{forecast_mon.iloc[-1]:.2f}</h3></center></p>'
        '</div>',
        unsafe_allow_html=True
    )
    money_pred = forecast_mon.iloc[-1]


# Predict Debt using thr pickle model
if user_input_year_debt:
    forecast_steps = user_input_year_debt - 2021
    forecast = model_debt.get_forecast(steps=forecast_steps)
    forecasted_values = forecast.predicted_mean
    tot_gov_debt = forecasted_values[forecast_steps - 1]
    debt_pred = tot_gov_debt








#----------------start of simulation part--------------------------------------#

#Section header
st.header("2: Monte Carlo Simulation")

#Define formulas---------------------------------------------------------------#

# Inflation
# m: "Money Printing"
# t: "Taxation"
# b: "borrowings"

def inf(m,t,d,f):
  values = [d, f, t, m]

  with open('svm_scaler.pkl', 'rb') as file:
      loaded_scaler = pickle.load(file)

  # Convert the list of values to a NumPy array
  array_2d = np.array(values)

  # Reshape the array to have one row and multiple columns
  array_2d = array_2d.reshape(1, -1)

  X_test_scaled = loaded_scaler.transform(array_2d)

  #Linear
  #return (6.42708112e-06 * d) - (2.35029896e-05 *f) + (7.91329095e-06* t) - (3.97426611e-07*m) + 7.663223081949728

  #SVR
  return (0.49161396 * X_test_scaled[0,0]) - (0.99352259 *X_test_scaled[0,1]) - (1.003063746* X_test_scaled[0,2]) - (0.65601281*X_test_scaled[0,3]) + 7.13781158
  #(0.49161396 * domestic) - (0.99352259 *foreign) - (1.003063746*taxation) - (0.65601281*money) + 7.13781158

  #Ridge
  #return (0.59462099 * X_test_scaled[0,0]) - (2.57325172 * X_test_scaled[0,1]) - (1.29945608 * X_test_scaled[0,2]) - (1.8897621 * X_test_scaled[0,3]) + 7.35055822395


# Debt Sustainablity
# gov_debt: "Total Government Debt as for now"
# foreign_debt: "Predicted Foreign Debt as for the 5th year"
# domestc_debt: "Predicted Domestic Debt as for the 5th year"
# money_printing: "Predicted Money Printing as for the 5th year"

#user input for GDP
#gdp_input = st.number_input("Please Enter Current GDP :", value=0, step=1)
gdp_input = 3114187

def debt(current_total_gov_debt,foreign_debt,domestc_debt,money_printing):
  #Keeping GDP Fixed for the next five years
  #gov_debt current
  GDP = gdp_input
  return (current_total_gov_debt + foreign_debt + domestc_debt + money_printing ) /GDP




#Debt Sustainablity
gov_debt =  debt_pred

#user input - inflation_target
inflation_target = st.number_input("Please Enter Target Inflation Rate:", value=0, step=1)

#user input - debt_target
gdebt_target = st.number_input("Please Enter Target Debt  Rate:", value=0, step=1)

#user input - expenditure amount
exp_amount = st.number_input("Please Enter Current expenditure : ", value=0, step=1)


if pred_date is not None:
    #****   Intialize Values  *********

    #           Inflation
    # Consider min by 150%

    # Money Printing
    pred_m= money_pred
    m_min = pred_m/2
    m_max =(pred_m * 1.5)

    #Taxation
    pred_t= results
    t_min = pred_t/2
    t_max = (pred_t *1.5)

    #Domestic Borrowings
    pred_b= domestic_pred
    b_min = pred_b/2
    b_max =(pred_b * 1.5)

    #Foreign Borrowings
    pred_f= predicted_debt
    f_min = pred_f/2
    f_max =(pred_f * 1.5)
else:
    st.write("Please Enter Year for Forecasting.")


#-----------------------------visualization------------------------------------#

# Calculate the total sum of the predicted values
#total_sum = predicted_debt + results + domestic_pred + money_pred

# Calculate the percentages for each value
#foreign_borrowing_percentage = (predicted_debt / total_sum) * 100
#taxation_percentage = (results / total_sum) * 100
#domestic_borrowings_percentage = (domestic_pred / total_sum) * 100
#money_printing_percentage = (money_pred / total_sum) * 100

# Data for the pie chart
#labels = ['Foreign Borrowing', 'Taxation', 'Domestic Borrowings', 'Money Printing']
#sizes = [foreign_borrowing_percentage, taxation_percentage, domestic_borrowings_percentage, money_printing_percentage]
#colors = ['#ff9999', '#66b3ff', '#99ff99', '#c2c2f0']

# Create the pie chart
#fig1, ax1 = plt.subplots()
#ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
#ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the pie chart using Streamlit
#st.pyplot(fig1)



#------------------------------------------------------------------------------#
# Define a button to trigger the Monte Carlo simulation
if st.button("Run Monte Carlo Simulation"):
    # Define the number of iterations
    n_iterations = 100000

    # Initializing empty list
    results = []

    # Run the Monte Carlo simulation
    for i in range(n_iterations):
        frames = []

        # Generate a random value for m, t, and b within the specified range

        # Inflation initialization
        mp = random.uniform(m_min, m_max)
        tax = random.uniform(t_min, t_max)
        f_bor = random.uniform(f_min, f_max)
        d_borrowings = random.uniform(b_min, b_max)

        # Debt initialization
        # for_debt = random.uniform(foreign_debt_min, foreign_debt_max)
        # dom_debt = random.uniform(domestic_debt_min, domestic_debt_max)
        # mp_debt = random.uniform(money_printing_min, money_printing_max)

        # Calculate the corresponding inflation value
        inf_val = inf(mp, tax, d_borrowings, f_bor)

        # Calculate the corresponding debt sustainability value
        debt_val = debt(gov_debt, f_bor, d_borrowings, mp)

        # Create a DataFrame with the calculated values and inputs
        df = pd.DataFrame({
            #'dom_inf': [dom_inf],
            #'for_inf': [for_inf],
            #'mp_inf': [mp_inf],
            'inf_val': [inf_val],
            'debt_val': [debt_val],
            'mp': [mp],
            'tax': [tax],
            'foreign': [f_bor],
            'dom': [d_borrowings]
        })

        # Append the DataFrame to the list of frames
        results.append(df)

    # Concatenate all the DataFrames in the list into a single DataFrame
    result_df = pd.concat(results, ignore_index=True)

    # Function to find optimized budgetary source values
    def optimize(df, inf_target, debt_target):
        filtered_df = df[
            (df['inf_val'] >= inf_target - 0.01) & (df['inf_val'] <= inf_target + 0.01) & (df['debt_val'] < debt_target)
        ]
        return filtered_df

    optimize_df = optimize(result_df, inflation_target, gdebt_target)

    final_df = optimize_df[
        (optimize_df['inf_val'] == optimize_df['inf_val'].min()) | (optimize_df['debt_val'] == optimize_df['debt_val'].min())]

    #st.dataframe(final_df)

    # Access 'mp', 'tax', 'foreign', 'dom' values for the second row (index 1)
    mp_value_2 = final_df.iloc[1]['mp']
    tax_value_2 = final_df.iloc[1]['tax']
    foreign_value_2 = final_df.iloc[1]['foreign']
    dom_value_2 = final_df.iloc[1]['dom']

    total_sum = foreign_value_2 + tax_value_2 + dom_value_2 + mp_value_2

    # Calculate the percentages for each value
    optimized_foreign_borrowing_percentage = (foreign_value_2 / total_sum) * 100
    optimized_taxation_percentage = (tax_value_2 / total_sum) * 100
    optimized_domestic_borrowings_percentage = (dom_value_2 / total_sum) * 100
    optimized_money_printing_percentage = (mp_value_2 / total_sum) * 100

    # Data for the pie chart
    labels = ['Foreign Borrowing', 'Taxation', 'Domestic Borrowings', 'Money Printing']
    sizes =[optimized_foreign_borrowing_percentage, optimized_taxation_percentage, optimized_domestic_borrowings_percentage, optimized_money_printing_percentage]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#c2c2f0']

    # Create the pie chart
    fig5, ax2 = plt.subplots()
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the pie chart using Streamlit
    #header
    st.header("Optimal Percentages of Budgetary Sources")
    st.pyplot(fig5)

    st.header("Allocated Amounts :")


    mon_exp_amount = optimized_money_printing_percentage * exp_amount
    dom_exp_amount = optimized_domestic_borrowings_percentage * exp_amount
    tax_exp_amount = optimized_taxation_percentage * exp_amount
    for_exp_amount = optimized_foreign_borrowing_percentage * exp_amount

    # Define custom CSS styles for the card
    card_style = """
         background-color: #581845;
         padding: px;
         border-radius: 4px;
         box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.2);
         margin-bottom: px;
    """

    # Create card-like components for each result
    st.markdown(
        f'<div style="{card_style}">'
        f'<h5>Allocated Money Printing for given expenditure {exp_amount}:</h5>'
        f'<center><p><h3>{mon_exp_amount:.2f}</h3></center></p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="{card_style}">'
        f'<h5>Allocated Tax for given expenditure {exp_amount}:</h5>'
        f'<center><p><h3>{tax_exp_amount:.2f}</h3></center></p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="{card_style}">'
        f'<h5>Allocated Foreign Borrowing for given expenditure {exp_amount}:</h5>'
        f'<center><p><h3>{for_exp_amount:.2f}<h3></center></p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="{card_style}">'
        f'<h5>Allocated Domestic Borrowing for given expenditure {exp_amount}:</h5>'
        f'<center><p><h3>{dom_exp_amount:.2f}</h3></center></p>'
        '</div>',
        unsafe_allow_html=True
    )


