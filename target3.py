import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
def app():
    #app heading
    st.write("""
    # Target Viability Prediction App
    This app predicts the ***investment viability*** of the target type!
    """)
    #st.sidebar.header("IS YOUR TARGET HOT or NOT?")
    #creating sidebar for user input features
    #st.sidebar.header('User Input Parameters') 
    def user_input_features():
        Drug_Descriptor = st.selectbox("Drug Descriptor"  "(0=Adoptive-Cell, 1=Allogeneic, 2=Antibiofilm, 3=Antibiotic, 4=Antifungal, 5=Antiinfective, 6=Antiinflammatory, 7=Autologous, 8=CAR-T, 9=Cancer-Vaccine, 10=Chemotherapy, 11=DNA-Based, 12=Immune-Inhibitor, 13=Immuno-Oncology, 14=Immunomodulatory, 15=Monoclonal-Antibody, 16=Non-Opioid , 17=Prophylactic-Vaccine, 18=RNA-Based, 19=Radiopharmaceutical, 20=Recombinant, 21=Synthetic, 22=Targeted, 23=Therapeutic-Vaccine, 24=Unknown, 25=mRNA-Based)",["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25"])
        Geography = st.selectbox("Geography" "(0=Global, 1=US)",["0","1"])
        Indication = st.selectbox( "Indication" "(0=Allergies, 1=Blood Disorder, 2=Cancer, 3=Cardiovascular Disorder, 4=Diabetes, 5=Digestive Disorder, 6=Drug Based Disorders, 7=Eye Disorder, 8=Fibrosis, 9=Genetic Disorder, 10=Gynaecological Disorders, 11=Immune System Disorders, 12=Infections, 13=Inflammatory Disease, 14=Kidney Disease, 15=Liver Disorders, 16=Muscular Disorders, 17=Neurological Disorder, 18=Obesity, 19=Other, 20=Psychological Disorder, 21=Respiratory Disorders, 22=Sexual Disorders, 23=Skeletal Disorders, 24=Skin Problems, 25=Transplant Disorders, 26=Tumor, 27=Unknown)",["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27"])
        Target_HDS = st.selectbox( "Highest Development Stage" "(0=Discontinued, 1=Discovery, 2=Inactive, 3=Phase 1, 4=Pre-Clinical)",["0","1","2","3","4"])
        Therapy_Area = st.selectbox("Therapy area" "(0=CNS, 1=Cardiovascular, 2=Dermatology, 3=Gastrointestinal, 4=Genetic, 5=Hematological, 6=Immunology, 7=Infectious-Disease, 8=Metabolic , 9=Musculoskeletal, 10=Oncology, 11=Ophthalmology, 12=Respiratory, 13=Unknown, 14=Urinary-System, 15=Women-Health)",["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"])
        Count_Trials = st.slider("Trial Count", 0, 100,1)
        News_Count = st.slider("Count of public news", 0, 100,1)
        Publication_Count = st.slider("Count of publications", 0, 2000,1)
        Grants = st.slider("Count of Grants provided", 0, 50,1)
        Alliances = st.slider("Count of Alliance Deals", 0, 50,1)
        Transactions = st.slider("Count of Various Transactions", 0, 50,1)
        Total_Amount = st.slider("Deal amount", 0, 200000,1)
        data = {'Drug_Descriptor': Drug_Descriptor,
                'Geography': Geography,
                'Indication': Indication,
              'Target_HDS': Target_HDS,
              'Therapy_Area': Therapy_Area,
                'Count_Trials': Count_Trials,
                'News_Count': News_Count ,
                'Publication_Count': Publication_Count ,
              'Grants': Grants,
              'Alliances': Alliances,
                'Transactions': Transactions,
                'Total_Amount': Total_Amount
                }
        features = pd.DataFrame(data, index=[0])
        return features
    df = user_input_features()

    #st.subheader('User Input parameters')
    st.write(df)
    #reading csv file
    data=pd.read_csv("Data_model_G.csv")
    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values
    #random forest model
    rfc= RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 500, max_depth=8, criterion='gini')
    rfc.fit(X, Y)
    #st.subheader('Wine quality labels and their corresponding index number')
    #st.write(pd.DataFrame({'wine quality': [3, 4, 5, 6, 7, 8 ]}))

    prediction = rfc.predict(df)
    #prediction_proba = rfc.predict_proba(df)
    st.subheader('Prediction')
    st.write(prediction)

    #st.subheader('Prediction Probability')
    #st.write(prediction_proba)

if __name__ == '__main__':
    app()    






    
