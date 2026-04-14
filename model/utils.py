"""
Legacy preprocessing utilities — not called by the experiment scripts.
The processed pickle files in data/processed/default/ are the required inputs.
Kept for reference only.

Inspired from https://github.com/asifri/facing_airborne_attacks/blob/main/facing_airborne_attacks.ipynb
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import argparse
import sys

DATA_DIR = "data/raw"
DATA_TRAIN_DIR = os.path.join(DATA_DIR, "train")
DATA_VALIDATION_DIR = os.path.join(DATA_DIR, "validation")
DATA_TEST_DIR = os.path.join(DATA_DIR, "test")
DATA_TEST_DEPARTING_DIR = os.path.join(DATA_TEST_DIR, "departing")
DATA_TEST_LANDING_DIR = os.path.join(DATA_TEST_DIR, "landing")
DATA_TEST_MANOEUVER_DIR = os.path.join(DATA_TEST_DIR, "manoeuver")
DATA_TEST_NOISE_DIR = os.path.join(DATA_TEST_DIR, "noise")
DATA_TEST_NORMAL_DIR = os.path.join(DATA_TEST_DIR, "normal")


def load_data_dict():
    data_dict = {
        "train": [],
        "validation": [],
        "test_noise": [],
        "test_landing": [],
        "test_departing": [],
        "test_manoeuver": [],
        "test_normal": []
    }
    paths = {
        "train": DATA_TRAIN_DIR,
        "validation": DATA_VALIDATION_DIR,
        "test_noise": DATA_TEST_NOISE_DIR,
        "test_landing": DATA_TEST_LANDING_DIR,
        "test_departing": DATA_TEST_DEPARTING_DIR,
        "test_manoeuver": DATA_TEST_MANOEUVER_DIR,
        "test_normal": DATA_TEST_NORMAL_DIR,
    }

    for key, path in paths.items():
        files = os.listdir(path)
        for file in files:
            df = pd.read_csv(os.path.join(path, file))
            data_dict[key].append(df)

    return data_dict

def get_windows_data(df, labels, window_size):
    X = pd.DataFrame(df.columns)
    temp_list = []
    for i, val in enumerate(rolled(df, window_size)):
        val = val.copy()
        val["id"] = [i] * window_size
        val["time"] = list(range(window_size))
        temp_list.append(val)
    X = pd.concat(temp_list)

    y = max_rolled(labels, window_size)
    return X, y

def max_rolled(list, window_size):
    y = []
    for val in rolled(list, window_size):
        y.append(max(val))
    return np.array(y)

def rolled(list, window_size):
    count = 0
    while count < len(list) - window_size:
        yield list[count:count+window_size]
        count += 1

def get_test_data(data_dict, config, keys):
    X_t_l = []
    y_t_l = []
    
    for key in tqdm(keys):
        X_l = []
        y_l = []
        for df in tqdm(data_dict[key]):
            X, y = get_windows_data(df[config['preprocessing']['features']], df["anomaly"], window_size=config['preprocessing']['window_size'])
            X_l.append(X)
            y_l.append(y)

        X_t_l.append(X_l)
        y_t_l.append(y_l)
    return X_t_l, y_t_l

def extract_test_features(X_t_l, y_t_l, config, keys):
    X_t_test = []
    y_t_test = []

    for i in tqdm(range(len(keys))):
        X_l = X_t_l[i]
        y_l = y_t_l[i]

        assert len(X_l) == len(y_l)

        y_test = np.array([])
        first = True

        for i in tqdm(range(len(X_l))):
            try:
                if first:
                    #X_test = impute(extract_features(X_l[i], column_id="id", column_sort="time", default_fc_parameters=MinimalFCParameters()))
                    X_test = impute(extract_features(X_l[i], column_id="id", column_sort="time", default_fc_parameters =config['preprocessing']['fc_parameters']))                    
                    first = False
                else:
                    #val = impute(extract_features(X_l[i], column_id="id", column_sort="time", default_fc_parameters=MinimalFCParameters()))
                    val = impute(extract_features(X_l[i], column_id="id", column_sort="time", default_fc_parameters =config['preprocessing']['fc_parameters']))
                    X_test = pd.concat([X_test, val], ignore_index=True)

                y_test = np.append(y_test, y_l[i])
            except Exception as e:
                print(e)
            continue

        X_t_test.append(X_test)
        y_t_test.append(y_test)   
    return X_t_test, y_t_test

def get_train_data(data_dict, config):
    X_l = []
    y_l = []
    for df in tqdm(data_dict["train"]):
        X, y= get_windows_data(df[config['preprocessing']['features']], [0] * df.shape[0], config['preprocessing']['window_size'])
        X_l.append(X)
        y_l.append(y)
    return X_l, y_l

def extract_train_val_features(X_l, y_l, config):
    y_train = np.array([])
    first = True
    for i in tqdm(range(len(X_l))):
        try:
            if first:
                # impute manages the missing data
                #X_train = impute(extract_features(X_l[i], column_id="id", column_sort="time", default_fc_parameters=MinimalFCParameters()))
                X_train = impute(extract_features(X_l[i], column_id="id", column_sort="time", default_fc_parameters =config['preprocessing']['fc_parameters']))
                first = False
            else:
                #X_train = pd.concat([X_train, impute(extract_features(X_l[i], column_id="id", column_sort="time", default_fc_parameters=MinimalFCParameters()))], ignore_index=True)
                X_train = pd.concat([X_train, impute(extract_features(X_l[i], column_id="id", column_sort="time", default_fc_parameters =config['preprocessing']['fc_parameters']))], ignore_index=True)
                
            y_train = np.append(y_train, y_l[i])
        except Exception as e:
            print(e)
            continue 

    return X_train, y_train

def get_validation_data(data_dict, config):
    X_l = []
    y_l = []
    for df in tqdm(data_dict["validation"]):
        X, y = get_windows_data(df[config['preprocessing']['features']], [0] * df.shape[0], config['preprocessing']['window_size'])
        X_l.append(X)
        y_l.append(y)
    return X_l, y_l


def processed_data(data_dict, config):
    keys = [x for x in data_dict.keys() if x.startswith("test")]
    
    # Load test data
    X_t_l, y_t_l = get_test_data(data_dict, config, keys)
    X_test, y_test = extract_test_features(X_t_l, y_t_l, config, keys)
    print("Test feature extraction done.")
    # Load train data
    X_l, y_l = get_train_data(data_dict, config)
    X_train, y_train = extract_train_val_features(X_l, y_l, config)
    print("Train feature extraction done.")

    X_v_l, y_v_l = get_validation_data(data_dict, config)
    X_val, y_val = extract_train_val_features(X_v_l, y_v_l, config)
    return X_train, X_test, X_val, y_train, y_test, y_val

def get_flight_lengths(window_size):
    flight_length = [0]
    flight_length_sum = [0]
    flight_names = []

    files = os.listdir(DATA_TEST_LANDING_DIR)
    for file in files:
        df = pd.read_csv(os.path.join(DATA_TEST_LANDING_DIR, file))
        flight_length.append(len(df) - window_size)
        flight_length_sum.append(flight_length_sum[-1] + len(df) - window_size)
        flight_names.append(os.path.splitext(file)[0])

    print("Flight names: ", flight_names)
    print("Flight length: ", flight_length)

    return flight_length, flight_length_sum, flight_names

def gather_input():
    parser = argparse.ArgumentParser(allow_abbrev=False, description='Train or load the model.')
    parser.add_argument('--plots', action='store_true', help='Plots SHAP profiles')
    parser.add_argument('--explain', action='store_true', help='Compute explanation')
    parser.add_argument('--config', type=str, help='Configuration to use for the variables of the model')
    
    
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(f"Error: unrecognised argument. Details: {e}")
        sys.exit(1)
    return args