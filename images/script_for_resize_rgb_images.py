from PIL import Image
import cv2
import os

folder_d1_for_resized_images_benign = '/mnt/d/IdeaProjects/data_malicious_links/rgb_images/pakhare_dataset/resized_images/benign'
folder_d1_for_resized_images_malicious = '/mnt/d/IdeaProjects/data_malicious_links/rgb_images/pakhare_dataset/resized_images/malicious'
folder_d1_for_initial_images_benign = '/mnt/d/IdeaProjects/data_malicious_links/rgb_images/pakhare_dataset/benign'
folder_d1_for_initial_images_malicious = '/mnt/d/IdeaProjects/data_malicious_links/rgb_images/pakhare_dataset/malicious'

folder_d2_for_resized_images_benign = '/mnt/d/IdeaProjects/data_malicious_links/rgb_images/alsaedi_dataset/resized_images/benign'
folder_d2_for_resized_images_malicious = '/mnt/d/IdeaProjects/data_malicious_links/rgb_images/alsaedi_dataset/resized_images/malicious'
folder_d2_for_initial_images_benign = '/mnt/d/IdeaProjects/data_malicious_links/rgb_images/alsaedi_dataset/benign'
folder_d2_for_initial_images_malicious = '/mnt/d/IdeaProjects/data_malicious_links/rgb_images/alsaedi_dataset/malicious'


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append((img, filename))
    return images

# images_benign_d1 = load_images_from_folder(folder_d1_for_initial_images_benign)
# images_malicious_d1 = load_images_from_folder(folder_d1_for_initial_images_malicious)
# max_dim_for_ascii_images_d1 = max([len(image[0]) for image in images_malicious_d1 + images_benign_d1])
# print(max_dim_for_ascii_images_d1)
# 28
max_dim_for_ascii_images_d1 = 28

# images_benign_d2 = load_images_from_folder(folder_d2_for_initial_images_benign)
# images_malicious_d2 = load_images_from_folder(folder_d2_for_initial_images_malicious)
# max_dim_for_ascii_images_d2 = max([len(image[0]) for image in images_malicious_d2 + images_benign_d2])
# print(max_dim_for_ascii_images_d2)
max_dim_for_ascii_images_d2 = 27

def resize_images_for_folder_and_save(initial_folder, save_folder, resize_dimensions):
    directory_files = os.listdir(initial_folder)
    multiple_images = [file for file in directory_files if 'rgb_img' in file and file.endswith(('.png'))]
    print(multiple_images)
    for image in multiple_images:
        img = Image.open(initial_folder + '/'+ image)
        imgResized = img.resize(size=resize_dimensions)
        # print(imgResized)
        # We would run the command below to save the images:
        imgResized.save(save_folder + '/'+ image, optimize=True)

resize_images_for_folder_and_save(folder_d1_for_initial_images_benign, folder_d1_for_resized_images_benign,
                                 resize_dimensions=(max_dim_for_ascii_images_d1,max_dim_for_ascii_images_d1))
resize_images_for_folder_and_save(folder_d1_for_initial_images_malicious, folder_d1_for_resized_images_malicious,
                                  resize_dimensions=(max_dim_for_ascii_images_d1,max_dim_for_ascii_images_d1))
resize_images_for_folder_and_save(folder_d2_for_initial_images_benign, folder_d2_for_resized_images_benign,
                                  resize_dimensions=(max_dim_for_ascii_images_d2,max_dim_for_ascii_images_d2))
resize_images_for_folder_and_save(folder_d2_for_initial_images_malicious, folder_d2_for_resized_images_malicious,
                                  resize_dimensions=(max_dim_for_ascii_images_d2,max_dim_for_ascii_images_d2))
