import pandas as pd

filename = r"D:/merged.csv"

df = pd.read_csv(filename)
counts_gold = df.groupby(['gold', 'split'])['split'].size().unstack(fill_value=0).add_prefix('count_')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(counts_gold)

counts_gold = df.groupby(['CT_Visual_Emph_Severity_P1', 'split'])['split'].size().unstack(fill_value=0).add_prefix('count_')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(counts_gold)


counts_gold = df.groupby(['CT_Visual_Emph_Paraseptal_P1', 'split'])['split'].size().unstack(fill_value=0).add_prefix('count_')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(counts_gold)