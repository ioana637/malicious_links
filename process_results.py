import pandas as pd
from IPython.core.display import display

def stats_for_ens_first_stage():
    filename1 = 'resurse/out_ens_alsaedi.csv'
    # filename2 = 'resurse/out_ens_alsaedi_2.csv'
    # filename3 = 'resurse/out_ens_alsaedi_3.csv'
    # filename4 = 'resurse/out_ens_alsaedi_4.csv'
    # filename = 'resurse/out_ens_2_2.csv'
    # outfile = 'resurse/out_ens_d1_stats_2.csv'
    outfile = 'resurse/out_ens_d2_stats_faza1_primul_fisier.csv'
    df1 = pd.read_csv(filename1)
    # df2 = pd.read_csv(filename2)
    # df3 = pd.read_csv(filename3)
    # df4 = pd.read_csv(filename4)
    df = pd.concat([df1])
    display(df.to_string())
    df_grouped_by_strategy = df.groupby('strategy').mean()
    df_grouped_by_c1_strategy = df.groupby('c1_strategy').mean()
    df_grouped_by_c2_strategy = df.groupby('c2_strategy').mean()
    df_grouped_by_w_strategy = df.groupby('w_strategy').mean()
    df_grouped_by_estimators = df.groupby('estimators').mean()
    # df_grouped_by_weights = df.groupby('weights').mean().sort_values(
    #     ['test_accuracy', 'test_macro_avg_f1', 'test_weighted_avg_f1'], ascending=False).head(10)
    # df_grouped_by_c1 = df.groupby('c1').mean().sort_values(
    #     ['test_accuracy', 'test_macro_avg_f1', 'test_weighted_avg_f1'], ascending=False).head(10)
    # df_grouped_by_c2 = df.groupby('c2').mean().sort_values(
    #     ['test_accuracy', 'test_macro_avg_f1', 'test_weighted_avg_f1'], ascending=False).head(10)
    # df_grouped_by_w = df.groupby('w').mean().sort_values(
    #     ['test_accuracy', 'test_macro_avg_f1', 'test_weighted_avg_f1'], ascending=False).head(10)

    display(df_grouped_by_strategy.to_string())
    display(df_grouped_by_c1_strategy.to_string())
    display(df_grouped_by_c2_strategy.to_string())
    display(df_grouped_by_w_strategy.to_string())
    df_grouped_by_strategy.to_csv(outfile, mode='a')
    df_grouped_by_w_strategy.to_csv(outfile, mode='a')
    df_grouped_by_c1_strategy.to_csv(outfile, mode='a')
    df_grouped_by_c2_strategy.to_csv(outfile, mode='a')
    df_grouped_by_estimators.to_csv(outfile, mode='a')
    display(df_grouped_by_estimators.to_string())
    # display(df_grouped_by_weights.to_string())
    # display(df_grouped_by_c1.to_string())
    # display(df_grouped_by_c1.reset_index().mean())
    # display(df_grouped_by_c2.to_string())
    # display(df_grouped_by_c2.reset_index().mean())
    # display(df_grouped_by_w.to_string())
    # display(df_grouped_by_w.reset_index().mean())


def prepare_data_for_2_phase_stats():
    # file1 = 'resurse/out_ens_3_continuare.csv'
    # file2 = 'resurse/out_ens_3_test.csv'
    # file3 = 'resurse/out_ens_3.csv'
    file1 = 'resurse/out_ens_alsaedi_5_phase2.csv'
    df = pd.read_csv(file1)
    # df1= pd.read_csv(file2)
    # df2= pd.read_csv(file3)
    print(df.count())
    print(df.shape)
    # print(df1.count())
    # print(df1.shape)
    # print(df2.count())
    # print(df2.shape)
    # df_final = pd.concat([df, df1, df2], axis = 0).drop_duplicates()
    # df_final = pd.concat([df, df1, df2], axis = 0).groupby(['estimators', 'w_strategy', 'c1_strategy',
    #                                                         'c2_strategy', 'n_inter_pso', 'n_particles', 'best_cost',
    #                                                         'weights'] ).mean()
    df_final = pd.concat([df])
    # indexNoIterations = df_final[~df_final['n_inter_pso'].isin([10, 100, 300, 500])].index
    # df_final.drop(indexNoIterations, inplace=True)
    # indexNoParticles = df_final[~df_final['n_particles'].isin([10, 30, 50, 80, 100])].index
    # df_final.drop(indexNoParticles, inplace=True)
    print(df_final.count())
    print(df_final.shape)
    display(df_final.to_string())
    df_final.to_csv('resurse/out_ens_2_phase.csv', index = False)

def stats_for_ens_second_stage():
    # file = 'resurse/out_ens_2_phase.csv'
    # outfile = 'resurse/out_ens_2_phase_stats.csv'
    file = 'resurse/out_ens_alsaedi_5_phase2.csv'
    outfile = 'resurse/out_ens_D2_2phase_stats.csv'
    df = pd.read_csv(file)
    df_groupby_n_iter_pso = df.groupby('n_inter_pso').mean().reset_index()
    df_groupby_n_particles = df.groupby('n_particles').mean().reset_index()
    df_grouped_by_estimators = df.groupby('estimators').mean().reset_index()

    df_groupby_n_iter_pso.to_csv(outfile, mode='a')
    df_groupby_n_particles.to_csv(outfile, mode='a')
    df_grouped_by_estimators.to_csv(outfile, mode='a')

if __name__ == "__main__":
    stats_for_ens_second_stage()
    # stats_for_ens_first_stage()
