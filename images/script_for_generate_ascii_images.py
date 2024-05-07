
import pandas as pd
from convertImages import convert_url_to_grayscale_image_ascii_method


def load_dataset_pakhare():
    file="../data/pakhare_urls.csv"
    df = pd.read_csv(file)
    # count
    print("Good samples:" + str(df.groupby('Class').size()['good']))
    print("Bad samples:" + str(df.groupby('Class').size()['bad']))
    print(df.info(verbose=False))
    # check for null values
    bool_series = pd.isnull(df["URLs"])
    print(df[bool_series])
    return df


def load_dataset_alsaedi():
    file="../data/Database2_binary.csv"
    df = pd.read_csv(file)
    # count
    print("Good samples:" + str(df.groupby('Class').size()['good']))
    print("Bad samples:" + str(df.groupby('Class').size()['bad']))
    print(df.info(verbose=False))
    # check for null values
    bool_series = pd.isnull(df["URLs"])
    print(df[bool_series])
    return df

# df = load_dataset_pakhare()
# df_sample_344821_good = df[df['Class']=='good']
# df_sample_75643_bad = df[df['Class']=='bad']
# print(df_sample_344821_good)
# print(df_sample_75643_bad)
#
# df = pd.concat([df_sample_75643_bad, df_sample_344821_good], axis=0)
# print(df)

df = load_dataset_alsaedi()
df_sample_428102_good = df[df['Class']=='good']
df_sample_223086_bad = df[df['Class']=='bad']
print(df_sample_428102_good)
print(df_sample_223086_bad)

df = pd.concat([df_sample_223086_bad, df_sample_428102_good], axis=0)
print(df)

def convert_to_ascii_images(df, folder_to_save):
    for index, row in df.iterrows():
        url = row["URLs"]
        label = row["Class"]
        # print(url, index)
        convert_url_to_grayscale_image_ascii_method(url, label, index, folder_to_save=folder_to_save)

folder_to_save='D:\IdeaProjects\malicious_links\data\images\\alsaedi_dataset'
convert_to_ascii_images(df, folder_to_save)
