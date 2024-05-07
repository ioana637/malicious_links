from PIL import Image
import PIL
import os
import glob

folder_for_resized_images = 'D:\IdeaProjects\malicious_links\data\qr_images\\alsaedi_dataset\\resize_200_200\\benign'
folder_with_qr_codes = 'D:\IdeaProjects\malicious_links\data\qr_images\\alsaedi_dataset\\benign'

def resize_images_for_folder_and_save(initial_folder, save_folder, resize_dimensions = (200, 200)):
    directory_files = os.listdir(initial_folder)
    multiple_images = [file for file in directory_files if 'qr_code' in file and file.endswith(('.png'))]
    print(multiple_images)
    for image in multiple_images:
        img = Image.open(folder_with_qr_codes + '\\'+ image)
        img.thumbnail(size=resize_dimensions)
        # print(img)
        # We would run the command below to save the images:
        img.save(save_folder + '\\'+ image, optimize=True)

resize_images_for_folder_and_save(folder_with_qr_codes, folder_for_resized_images, (200,200))


folder_for_resized_images = 'D:\IdeaProjects\malicious_links\data\qr_images\\alsaedi_dataset\\resize_200_200\\malicious'
folder_with_qr_codes = 'D:\IdeaProjects\malicious_links\data\qr_images\\alsaedi_dataset\\malicious'

resize_images_for_folder_and_save(folder_with_qr_codes, folder_for_resized_images, (200,200))