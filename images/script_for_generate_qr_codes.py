import pandas as pd
import qrcode

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
# print(df_sample_75643_bad["URLs"].str.len().max())
# print(df_sample_344821_good["URLs"].str.len().max())
# print(df_sample_75643_bad["URLs"].str.len().min())
# print(df_sample_344821_good["URLs"].str.len().min())


df2 = load_dataset_alsaedi()
print('Alsaedi D2 dataset')
df2_sample_429102_good = df2[df2['Class']=='good']
df2_sample_223086_bad = df2[df2['Class']=='bad']
print(df2_sample_429102_good)
print(df2_sample_223086_bad)

print(df2_sample_429102_good["URLs"].str.len().max())
print(df2_sample_223086_bad["URLs"].str.len().max())
print(df2_sample_429102_good["URLs"].str.len().min())
print(df2_sample_223086_bad["URLs"].str.len().min())


# df must have 2 columns: Class and URLs
def generate_qr_codes(df, saving_path = '../data/qr_images/pakhare_dataset/qr_code_', version = 3, box_size =30,
                      border = 0, error_correction = qrcode.constants.ERROR_CORRECT_M, fill_color='black',
                      back_color = 'white'):
    for index, row in df.iterrows():
        url = row["URLs"]
        label = row["Class"]
        # print(url, index)
        qr = qrcode.QRCode(version=version, box_size=box_size, border=border, error_correction=error_correction)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color=fill_color, back_color=back_color)
        img.save(saving_path+str(index)+"_"+label+".png")

# version = 40
# box_size = 30
# border = 0
# error_correction = qrcode.constants.ERROR_CORRECT_M
# fill_color = "black"
# back_color = "white"

# generate_qr_codes(df_sample_75643_bad)
# generate_qr_codes(df_sample_344821_good)

generate_qr_codes(df2_sample_429102_good, saving_path = '../data/qr_images/alsaedi_dataset/benign/qr_code_')
