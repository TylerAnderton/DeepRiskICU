def assert_unique(df, col):
    assert df[col].nunique(dropna=True) == len(df), f'Some {col} have multiple rows'
    
