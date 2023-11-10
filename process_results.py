import pandas as pd
from IPython.core.display import display

if __name__ == "__main__":
    filename = 'resurse/out_ens_2_2.csv'
    df = pd.read_csv(filename)
    # df_grouped_by_strategy = df.groupby('strategy').mean()
    # df_grouped_by_c1_strategy = df.groupby('c1_strategy').mean()
    # df_grouped_by_c2_strategy = df.groupby('c2_strategy').mean()
    # df_grouped_by_w_strategy = df.groupby('w_strategy').mean()
    # df_grouped_by_estimators = df.groupby('estimators').mean()
    # df_grouped_by_weights = df.groupby('weights').mean().sort_values(
    #     ['test_accuracy', 'test_macro_avg_f1', 'test_weighted_avg_f1'], ascending=False).head(10)
    df_grouped_by_c1 = df.groupby('c1').mean().sort_values(
        ['test_accuracy', 'test_macro_avg_f1', 'test_weighted_avg_f1'], ascending=False).head(10)
    df_grouped_by_c2 = df.groupby('c2').mean().sort_values(
        ['test_accuracy', 'test_macro_avg_f1', 'test_weighted_avg_f1'], ascending=False).head(10)
    df_grouped_by_w = df.groupby('w').mean().sort_values(
        ['test_accuracy', 'test_macro_avg_f1', 'test_weighted_avg_f1'], ascending=False).head(10)

    # display(df_grouped_by_strategy.to_string())
    # display(df_grouped_by_c1_strategy.to_string())
    # display(df_grouped_by_c2_strategy.to_string())
    # display(df_grouped_by_w_strategy.to_string())
    # display(df_grouped_by_estimators.to_string())
    # display(df_grouped_by_weights.to_string())
    display(df_grouped_by_c1.to_string())
    display(df_grouped_by_c1.reset_index().mean())
    display(df_grouped_by_c2.to_string())
    display(df_grouped_by_c2.reset_index().mean())
    display(df_grouped_by_w.to_string())
    display(df_grouped_by_w.reset_index().mean())
