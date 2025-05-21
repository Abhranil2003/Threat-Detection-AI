import pandas as pd
import pytest
from preprocessing import preprocess_data

def test_preprocess_data_valid_input():
    df = pd.DataFrame({
        'Src IP': ['192.168.1.1'],
        'Dst IP': ['192.168.1.2'],
        'Protocol': ['TCP'],
        'TotLen Fwd Pkts': [20],
        'Label': ['BENIGN']
    })
    processed_df = preprocess_data(df)
    assert isinstance(processed_df, pd.DataFrame)
    assert not processed_df.empty


def test_preprocess_data_missing_values():
    df = pd.DataFrame({
        'Src IP': ['192.168.1.1'],
        'Dst IP': [None],
        'Protocol': ['TCP'],
        'TotLen Fwd Pkts': [None],
        'Label': ['BENIGN']
    })
    processed_df = preprocess_data(df)
    assert processed_df.isnull().sum().sum() == 0