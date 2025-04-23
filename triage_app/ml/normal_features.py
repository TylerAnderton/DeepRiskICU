import pandas as pd
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
import joblib
import os
from django.conf import settings

from .utils import assert_unique

def get_patient_admission_data(input: dict) -> pd.DataFrame:
    print()
    print('Processing patient and admission data...')
    patient_data = pd.DataFrame([input['patient_input']])
    patient_data['dob'] = pd.to_datetime(patient_data['dob'])
    patient_data['dob'] = patient_data['dob'].dt.tz_localize(None)

    admission_data = pd.DataFrame([input['admission_input']])
    admission_data['admission_dt'] = pd.to_datetime(admission_data['admission_dt'])
    admission_data['admission_dt'] = admission_data['admission_dt'].dt.tz_localize(None)

    patient_data = pd.DataFrame([input['patient_input']])
    patient_data['dob'] = pd.to_datetime(patient_data['dob'])
    patient_data['dob'] = patient_data['dob'].dt.tz_localize(None)

    admission_data = pd.DataFrame([input['admission_input']])
    admission_data['admission_dt'] = pd.to_datetime(admission_data['admission_dt'])
    admission_data['admission_dt'] = admission_data['admission_dt'].dt.tz_localize(None)

    patient_admission_data = pd.merge(
        patient_data,
        admission_data,
        on='patient_id',
        how='inner'
    )

    assert_unique(patient_admission_data, 'admission_id')
    patient_admission_data.drop(columns='patient_id', inplace=True)

    return patient_admission_data

def calculate_age(admit_date, birth_date):
    print('Calculating ages...')
    try:
        return (admit_date - birth_date).days / 365
    except (OverflowError, OutOfBoundsDatetime):
        # If dates are too extreme, check if year difference is reasonable
        year_diff = admit_date.year - birth_date.year
        if 0 <= year_diff <= 120:  # Reasonable age range
            return year_diff
        else:
            print(f'admission_id: {admit_date} - {birth_date} has unreasonable age difference: {year_diff}')
            return float('nan')  # Return NaN for unreasonable values

def calculate_ages(patient_admission_data: pd.DataFrame) -> pd.DataFrame:
    patient_admission_data['age_yrs'] = patient_admission_data.apply(
        lambda row: calculate_age(row['admission_dt'], row['dob']), axis=1
    )

    print(f'Dropping {patient_admission_data["age_yrs"].isna().sum()} NaN values in age_yrs')
    patient_admission_data.dropna(subset=['age_yrs'], inplace=True, ignore_index=True)
    
    age_stats = pd.read_csv(os.path.join(settings.BASE_DIR, 'ml', 'data', 'training', 'age_stats.csv'))
    age_mean = age_stats['age_mean'].values[0]
    age_std = age_stats['age_std'].values[0]
    
    patient_admission_data['age_yrs'] = (patient_admission_data['age_yrs'] - age_mean) / age_std
    
    return patient_admission_data

def encode_categorical_features(patient_admission_data: pd.DataFrame, encoder_path: str) -> pd.DataFrame:
    print('Encoding categorical features...')
    categorical_cols = [
        'gender',
        'ethnicity',
        'admission_location',
        'insurance_type',
    ]
    drop_cols = [
        'dob',
        'admission_dt',
    ]

    encoder = joblib.load(encoder_path)

    encoded_features = encoder.transform(patient_admission_data[categorical_cols])
    encoded_features_df = pd.DataFrame(
        encoded_features,
        columns=encoder.get_feature_names_out(),
        index=patient_admission_data.index
    )

    patient_admission_features = patient_admission_data.copy()
    patient_admission_features.drop(
        columns=categorical_cols + drop_cols,
        inplace=True
    )

    patient_admission_features = pd.concat(
        [patient_admission_features, encoded_features_df],
        axis=1
    )
    patient_admission_features.sort_values('admission_id').reset_index(drop=True)

    assert_unique(patient_admission_features, 'admission_id')
    print(f"Shape before one-hot encoding: {patient_admission_data.shape}")
    print(f"Shape after one-hot encoding: {patient_admission_features.shape}")
    
    return patient_admission_features