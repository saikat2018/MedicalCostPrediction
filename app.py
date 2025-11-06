import streamlit as st
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import warnings
import joblib
import numpy as np

# Model gets called here
def calculate():
    st.text("Age:"+str(age))
    st.text("Gender:"+str(gender))
    #st.text("Weight:"+str(weight))
    bmi = (weight * 10000)/ (height * height) 
    st.text("BMI:"+str(round(bmi,ndigits=2)))
    st.text("No. of Dependents:"+str(no_of_Children))
    st.text("Smoker:"+str(smoker))
    st.text("Region:"+str(region))
    #st.text("Model Selected:"+str(model_selected))
       
    #warnings.filterwarnings('ignore')
    #df = pd.read_csv("medical_costs.csv")

    #df['Sex'] = df['Sex'].astype('category').cat.codes
    #df['Smoker'] = df['Smoker'].astype('category').cat.codes
    #df['Region'] = df['Region'].astype('category').cat.codes

    #from sklearn.metrics import accuracy_score
    #import xgboost as xgb
    #from sklearn.model_selection import train_test_split

    # Send Medical Cost to Data Frame

    #X = df.drop(columns=['Medical Cost'])
    #y = df['Medical Cost']

    #X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=0)
    
    #model = xgb.XGBRegressor(
    #objective='reg:squarederror', # For regression tasks
    #n_estimators=1000,            # Number of boosting rounds
    #learning_rate=0.1,          # Step size shrinkage
    #max_depth=7,                 # Maximum depth of trees
    #subsample=0.7,               # Fraction of samples to use per tree
    #colsample_bytree=0.8,       # Fraction of features to use per tree
    #)

    #model.fit(X_train, y_train)

    # Make predictions
    #y_pred = model.predict(X_test)

    # Evaluate the model (MSE)
    #from sklearn.metrics import mean_squared_error
    #mse = mean_squared_error(y_pred, y_test)
    #print("Mean Squared Error = ", mse)

    # Evaluate the model (RMSE)
    #rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
    #print("RMSE : % f" %(rmse))

    # Evaluate the model (MAE)
    #from sklearn.metrics import mean_absolute_error
    # Calculate MAE
    #mae = mean_absolute_error(y_test, y_pred)
    #print(f"Mean Absolute Error: {mae}")

    #from sklearn.metrics import mean_absolute_percentage_error
    # Evaluate the model (MAPE)
    #mape_value_sklearn = mean_absolute_percentage_error(y_test, y_pred)
    #print("MAPE:", mape_value_sklearn)

    #from sklearn.metrics import r2_score
    # Evaluate the model (R2 score)
    #r2 = r2_score(y_test, y_pred)
    #print('r2 score for the model is', r2)

    
    # save the model to a file
    
    #joblib.dump(model, 'XGB_regression_model_newdata.joblib')
    
    # the First parameter is the name of the model and the second parameter is the name of the file
    # with which we want to save it


    #from sklearn.ensemble import BaggingRegressor
    #from sklearn.tree import DecisionTreeRegressor

    # Use DecisionTreeRegressor as the base estimator
    #base_regressor = DecisionTreeRegressor(max_depth=8)

    # Create the BaggingRegressor
    #bagging_regressor = BaggingRegressor(base_regressor, n_estimators=3000, random_state=0)
    #bagging_regressor.fit(X_train, y_train)
    #y_pred_bagging = bagging_regressor.predict(X_test)
    # Calculate Mean Squared Error
    #mse_bagging = mean_squared_error(y_test, y_pred_bagging)
    #print(f"Mean Squared Error: {mse_bagging}")

    # Evaluate the model (RMSE)
    #rmse_bagging = np.sqrt(mean_squared_error(y_test, y_pred_bagging)) 
    #print("RMSE : % f" %(rmse_bagging))

    # Evaluate the model (MAE)
    #from sklearn.metrics import mean_absolute_error
    # Calculate MAE
    #mae_bagging = mean_absolute_error(y_test, y_pred_bagging)
    #print(f"Mean Absolute Error: {mae_bagging}")

    # Evaluate the model (MAPE)
    #mape_bagging = mean_absolute_percentage_error(y_test, y_pred_bagging)
    #print("MAPE:", mape_bagging)

    # Evaluate the model (R2 score)
    #r2_bagging = r2_score(y_test, y_pred_bagging)
    #print('r2 score for the model is', r2_bagging)

    #joblib.dump(bagging_regressor, 'Bagging_regression_model_newdata.joblib')

    # Input parameters here - Manoj
    input_data = [age,gender,bmi,no_of_Children,smoker,region]
    input_data_array=np.array(input_data)
    print(input_data_array.shape)
    input_data_array_T= input_data_array.reshape(1,6)
    print(input_data_array_T.shape)
    
    #if(model_selected == "1"):
    xgb_medical_cost = predict_value('XGB_regression_model_newdata.joblib', input_data_array_T)
    st.text("With XGB Model, medical cost is: $"+str(xgb_medical_cost))
    #elif(model_selected == "2"):
    bagging_medical_cost = predict_value('Bagging_regression_model_newdata.joblib', input_data_array_T)
    st.text("With Bagging Model, medical cost is: $"+str(bagging_medical_cost))
    #st.text("The Patient's insurance premium based on his vital statistics:"+str(medical_cost))

def predict_value(model_path, input_data):
    # """
    # Predicts values using a trained model.

    # Args:
    #     model_path (str): Path to the saved model file (e.g., .pkl, .joblib).
    #     input_data (array-like): Input data for prediction.

    # Returns:
    #     array-like: Predicted values.
    # """
    # Load the trained model
    model = joblib.load(model_path)
    
    # Make predictions
    predictions = model.predict(input_data)
    print(predictions)
    return predictions

with st.form(key = "Form 1"):

    #Name
    name = st.text_input(label= "Patient Name")

    #Title
    #title_data = pd.read_csv("job_title_codes.csv")
    #title_options = title_data['title'].to_list()
    #title_codes = title_data['code'].to_list()
    #job_title = st.selectbox("Job Title",title_options)
    #title_key_mapping = dict(zip(title_options,title_codes))
    #title_key = title_key_mapping[job_title]

    #City
    #city_data = pd.read_csv("city_codes.csv")
    #city_options = city_data['city'].tolist()
    #city_codes = city_data['code'].tolist()
    #city_name = st.selectbox("City:",city_options)
    #city_key_mapping = dict(zip(city_options, city_codes))
    #city_key = city_key_mapping[city_name]

    #region
    region_mapping = {1:"Northwest", 0:"NorthEast",2:"SouthEast",3:"SouthWest"}
    region = st.radio("Region:",options=[0,1,2,3],format_func=lambda x: region_mapping[x])
    
    #age
    age = st.slider("Enter your age", min_value=1, max_value=99)

    #gender
    gender_mapping = {1:"Male", 0:"Female"}
    gender = st.radio("Gender",options=[0,1],format_func=lambda x: gender_mapping[x])
    
    #weight
    weight = st.number_input("Enter Weight (kg)",step=1,min_value=1)
    
    #height
    height = st.number_input ("Enter Height (cm)",step=1,min_value=10)
    
    #No. of Children
    no_of_Children = st.slider("Number of Children:",min_value=0,max_value=10)
    
    #Smoker
    smoker_mapping = {1:"Smoker", 0:"Non-smoker"}
    smoker= st.radio("Are you a smoker?",options=[0,1],format_func=lambda x: smoker_mapping[x])

    #model selection
    #model_mapping = {1:"XGB",2:"Bagging Regression"}
    #model_selected = st.radio("Model Selected:",options=[1,2],format_func=lambda x:model_mapping[x])
    
    submit = st.form_submit_button(label = "Submit to Calculate")

    if submit:
        calculate()
    

