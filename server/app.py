from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
from itertools import combinations

app = Flask(__name__)

model = pickle.load(open('../final-model/rf.pkl','rb'))
augmenter = pickle.load(open('../final-model/augmenter.pkl','rb'))
scaler = pickle.load(open('../final-model/scaler.pkl','rb'))
imputer = pickle.load(open('../final-model/imputer.pkl','rb'))

@app.route('/', methods=['POST'])
def hello():
    data = request.get_json(force=True)
    data = pd.DataFrame.from_dict(data)

    presenting_problem = data['Presenting problem']
    pp_f = 'Presenting problem_'+presenting_problem.values[0]
    data = data.drop(['Presenting problem'], axis=1)
    data[pp_f] = 1

    # =================== AUGMENT ===========================
    features = augmenter.feature_names_in_
    df_augmenter = pd.DataFrame.from_dict(dict(zip(list(features), [[0]]*len(list(features)))))
    df_augmenter.update(data)
    df_augmenter = df_augmenter.replace('', float('NAN'))

    x, _, columns = augment(df_augmenter, augmenter)

    # =================== SCALE ===========================
    features = scaler.feature_names_in_
    df_scale = pd.DataFrame.from_dict(dict(zip(list(features), [[0]]*len(list(features)))))
    df_scale.update(x)

    df_scale[(df_scale < 0) & (df_scale != np.nan)] = np.nan
    x, _ = scale(df_scale, scaler)

    # =================== IMPUTE ===========================
    x, _ = impute(x, imputer)

    # =================== SELECT ROWS ===========================
    x, filtered_columns = select_rows(x, columns)

    # =================== PREDICT ===================
    prediction = model.predict(x)

    return str(prediction)


# ================================

def select_rows(x, cols):
    df = pd.DataFrame(x, columns = cols)
    df = df[df.columns.drop(list(df.filter(regex='Positive discriminator|MTS|Hospitalisation')))]
    return df.to_numpy(), df.columns

def scale(x, scaler):
    return pd.DataFrame(scaler.transform(x), columns = x.columns), scaler

def impute(x, imputer):
    x = pd.DataFrame(imputer.transform(x), columns = x.columns)
    return x, imputer

def augment(x, augmenter):
    x = x.fillna(-1)
    x, augmenter = _add_interactions(x, augmenter)
    return x, augmenter, x.columns

def _add_interactions(df, augmenter):
    combos = list(combinations(list(df.columns), 2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]

    df = augmenter.transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames

    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
    cols_drop = set(df.columns[noint_indicies]) - set(interaction_cols)
    df = df.drop(cols_drop, axis=1)

    return df, augmenter

interaction_cols = [
       'Presenting problem_Dyspnea', 'Presenting problem_Ear, nose, throat',
       'Presenting problem_Fever without source',
       'Presenting problem_Gastro-intestinal',
       'Presenting problem_Neurological', 'Presenting problem_Other problems',
       'Presenting problem_Rash', 'Presenting problem_Trauma',
       'Presenting problem_Urinary tract problems',
       'Presenting problem_Wounds',
       'Presenting problem_local infection/abscess', 'Age', 'Respiratory rate',
       'Heart rate', 'Temperature', 'Oxygen saturation',
       'Presenting problem_Dyspnea_Age',
       'Presenting problem_Dyspnea_Respiratory rate',
       'Presenting problem_Dyspnea_Heart rate',
       'Presenting problem_Dyspnea_Temperature',
       'Presenting problem_Dyspnea_Oxygen saturation',
       'Presenting problem_Ear, nose, throat_Age',
       'Presenting problem_Ear, nose, throat_Respiratory rate',
       'Presenting problem_Ear, nose, throat_Heart rate',
       'Presenting problem_Ear, nose, throat_Temperature',
       'Presenting problem_Ear, nose, throat_Oxygen saturation',
       'Presenting problem_Fever without source_Age',
       'Presenting problem_Fever without source_Respiratory rate',
       'Presenting problem_Fever without source_Heart rate',
       'Presenting problem_Fever without source_Temperature',
       'Presenting problem_Fever without source_Oxygen saturation',
       'Presenting problem_Gastro-intestinal_Age',
       'Presenting problem_Gastro-intestinal_Respiratory rate',
       'Presenting problem_Gastro-intestinal_Heart rate',
       'Presenting problem_Gastro-intestinal_Temperature',
       'Presenting problem_Gastro-intestinal_Oxygen saturation',
       'Presenting problem_Neurological_Age',
       'Presenting problem_Neurological_Respiratory rate',
       'Presenting problem_Neurological_Heart rate',
       'Presenting problem_Neurological_Temperature',
       'Presenting problem_Neurological_Oxygen saturation',
       'Presenting problem_Other problems_Age',
       'Presenting problem_Other problems_Respiratory rate',
       'Presenting problem_Other problems_Heart rate',
       'Presenting problem_Other problems_Temperature',
       'Presenting problem_Other problems_Oxygen saturation',
       'Presenting problem_Rash_Age',
       'Presenting problem_Rash_Respiratory rate',
       'Presenting problem_Rash_Heart rate',
       'Presenting problem_Rash_Temperature',
       'Presenting problem_Rash_Oxygen saturation',
       'Presenting problem_Trauma_Age',
       'Presenting problem_Trauma_Respiratory rate',
       'Presenting problem_Trauma_Heart rate',
       'Presenting problem_Trauma_Temperature',
       'Presenting problem_Trauma_Oxygen saturation',
       'Presenting problem_Urinary tract problems_Age',
       'Presenting problem_Urinary tract problems_Respiratory rate',
       'Presenting problem_Urinary tract problems_Heart rate',
       'Presenting problem_Urinary tract problems_Temperature',
       'Presenting problem_Urinary tract problems_Oxygen saturation',
       'Presenting problem_Wounds_Age',
       'Presenting problem_Wounds_Respiratory rate',
       'Presenting problem_Wounds_Heart rate',
       'Presenting problem_Wounds_Temperature',
       'Presenting problem_Wounds_Oxygen saturation',
       'Presenting problem_local infection/abscess_Age',
       'Presenting problem_local infection/abscess_Respiratory rate',
       'Presenting problem_local infection/abscess_Heart rate',
       'Presenting problem_local infection/abscess_Temperature',
       'Presenting problem_local infection/abscess_Oxygen saturation',
       'Age_Respiratory rate', 'Age_Heart rate', 'Age_Temperature',
       'Age_Oxygen saturation', 'Respiratory rate_Heart rate',
       'Respiratory rate_Temperature', 'Respiratory rate_Oxygen saturation',
       'Heart rate_Temperature', 'Heart rate_Oxygen saturation',
       'Temperature_Oxygen saturation']


