import pandas as pd

def get_geno_data(folder='data/', filename='Gatti_2014_geno.csv'):
    geno = pd.read_csv(folder + filename, na_values = ['--'] )
    geno.set_index('SnpId',inplace=True)
    geno = geno.T
    return geno

def get_pheno_data(folder='data/', filename='Gatti_2014_pheno.csv'):
    pheno = pd.read_csv(folder + filename)
    pheno.set_index('Sample.ID',inplace=True)
    return pheno

def convert_geno_to_binary(geno):
    return pd.get_dummies(geno, dtype = int)

def get_ternary_replacements(col):
    bases = ['A', 'C', 'G', 'T']
    all_base_pairs = [b1 + b2 for b1 in bases for b2 in bases]
    existing_base_pairs = col.unique()
    if 'TT' in existing_base_pairs:
        replacements = {bp: bp.count('T') for bp in all_base_pairs}
    elif 'GG' in existing_base_pairs:
        replacements = {bp: bp.count('G') for bp in all_base_pairs}
    elif 'CC' in existing_base_pairs:
        replacements = {bp: bp.count('C') for bp in all_base_pairs}
    elif 'AA' in existing_base_pairs:
        replacements = {bp: bp.count('A') for bp in all_base_pairs}
    else:
        replacements = {}
    return replacements

def ternarize_single_column(col):
    return col.replace(to_replace=get_ternary_replacements(col))

def ternarize_dataframe(df):
    all_replacements = {col_name: get_ternary_replacements(df[col_name]) for col_name in df.columns}
    return df.replace(to_replace=all_replacements)

def convert_geno_to_ternary(geno):
    if type(geno) == pd.DataFrame:
        return ternarize_dataframe(geno)
    elif type(geno) == pd.Series:
        return ternarize_single_column(geno)
    
def drop_single_value_cols(geno):
    is_single_value = lambda x: len(x.dropna().unique()) <= 1
    columns_to_drop = geno.columns[geno.apply(is_single_value, axis=0)]
    return geno.drop(columns=columns_to_drop)