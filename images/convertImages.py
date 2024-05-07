import numpy as np
from PIL import Image
from PIL.Image import Resampling


def get_size_of_image(url):
    n = len(url)
    return get_min_sq_root_greater(n)


def get_min_sq_root_greater(n):
    root_of_n = np.sqrt(n)
    floor_integer = int(root_of_n)
    ceil_int = floor_integer + 1
    floor_integer_square = floor_integer * floor_integer
    ceil_int_square = ceil_int * ceil_int
    floor_distance = n - floor_integer_square
    ceil_distance = n - ceil_int_square
    # print(floor_distance, ceil_distance)
    if floor_distance == 0:
        return floor_integer
    if ceil_distance == 0:
        return ceil_int
    if floor_distance < ceil_distance:
        return floor_integer
    else:
        return ceil_int


def get_size_of_rgb_image(url):
    len_url = len(url)
    n = len_url / 3.0
    return get_min_sq_root_greater(n)

def test_get_size_of_image():
    url = "ussoccer.com/News/Federation-Services/2009/06/University-Of-Miami-President-Donna-E-Shalala-Joins-Team-To-Bring-FIFA-World-Cup-To-United-States-In.aspx"
    url1 = "ussoccer.com/News/Federation-ServE-Shalala-Joins-Team-To-Bring-FIFA-World-Cup-To-United-States-In.aspx"
    assert get_size_of_image(url) == 13
    assert get_size_of_image(url1) == 11
    assert get_size_of_image('abcd') == 2
    assert get_size_of_image('0abcdefgh') == 3

def test_get_size_of_RGB_image():
    url = "ussoccer.com/News/Federation-Services/2009/06/University-Of-Miami-President-Donna-E-Shalala-Joins-Team-To-Bring-FIFA-World-Cup-To-United-States-In.aspx"
    url1 = "ussoccer.com/News/Federation-ServE-Shalala-Joins-Team-To-Bring-FIFA-World-Cup-To-United-States-In.aspx"
    assert get_size_of_rgb_image(url) == 8
    assert get_size_of_rgb_image(url1) == 6
    assert get_size_of_rgb_image('abcd') == 2
    assert get_size_of_rgb_image('0abcdefgh') == 2

def convert_url_to_grayscale_image_ascii_method(url, label, id, folder_to_save, resize_value = 1):
    width_height_image = get_size_of_image(url)
    url_to_numbers = np.frombuffer(url.encode('UTF-32-LE'), dtype=np.uint32)
    matrix = np.array_split(url_to_numbers, width_height_image)
    im = Image.new('L', (width_height_image, width_height_image))
    pix = im.load()
    for y in range(width_height_image):  # height
        for x in range(width_height_image):  # width
            try:
                pix[x, y] = int(matrix[y][x])
            except IndexError:
                pix[x, y] = 0

    im = im.resize((width_height_image * resize_value, width_height_image * resize_value), resample=Resampling.NEAREST)
    filename = "ascii_img_"+str(id)+"_"+label+".png"
    if label == 'good':
        folder_to_save = folder_to_save + "\\benign\\"+filename
    else:
        folder_to_save = folder_to_save + "\\malicious\\"+filename
    im.save(folder_to_save)

def main_metoda2():
    url = "ussoccer.com/News/Federation-Services/2009/06/University-Of-Miami-President-Donna-E-Shalala-Joins-Team-To-Bring-FIFA-World-Cup-To-United-States-In.aspx"
    label = "good"
    width_height_image = get_size_of_image(url)
    url_to_numbers = np.frombuffer(url.encode('UTF-32-LE'), dtype=np.uint32)
    print(url_to_numbers)
    matrix = np.array_split(url_to_numbers, width_height_image)
    print(matrix)
    print(len(matrix))
    print(len(matrix[0]))
    im = Image.new('L', (width_height_image, width_height_image))
    pix = im.load()
    for y in range(width_height_image):  # height
        for x in range(width_height_image):  # width
            try:
                pix[x, y] = int(matrix[y][x])
            except IndexError:
                pix[x, y] = 0

    # im = im.resize((width_height_image * 10, width_height_image * 10), resample=Resampling.NEAREST)
    im.save('image.png')

def main_metoda1():
    url = "ussoccer.com/News/Federation-Services/2009/06/University-Of-Miami-President-Donna-E-Shalala-Joins-Team-To-Bring-FIFA-World-Cup-To-United-States-In.aspx"
    label = "good"

    url_to_numbers = np.frombuffer(url.encode('UTF-32-LE'), dtype=np.uint32)
    print(url_to_numbers)
    matrix = np.array_split(url_to_numbers, 30)
    print(matrix)
    print(len(matrix))
    print(len(matrix[0]))
    im = Image.new('L', (30, 30))
    pix = im.load()
    for y in range(30): # height
        for x in range(6): #width
            try:
                pix[x, y] = int(matrix[y][x])
            except IndexError:
                pix[x,y] = 0


    im = im.resize((30 * 10, 30 * 10), resample=Resampling.NEAREST)
    im.save('image.png')


def convert_url_to_rgb_image_ascii_method(url, label, id, folder_to_save, resize_value = 1):
    width_height_image = get_size_of_rgb_image(url)
    url_to_numbers = np.frombuffer(url.encode('UTF-32-LE'), dtype=np.uint32)
    matrix_with_rgb = np.array_split(url_to_numbers, int(len(url) / 3) + 1)
    matrix_with_rgb = list(map(tuple, matrix_with_rgb))
    im = Image.new('RGB', (width_height_image, width_height_image), color=(0, 0, 0))
    pix = im.load()
    for y in range(width_height_image):  # height
        for x in range(width_height_image):  # width
            try:
                pixel = matrix_with_rgb[y * width_height_image + x]
                while (len(pixel) < 3):
                    pixel = (*pixel, 0)
                pix[x, y] = pixel
            except IndexError:
                pix[x, y] = 0
    im = im.resize((width_height_image * resize_value, width_height_image * resize_value), resample=Resampling.NEAREST)
    filename = "rgb_img_"+str(id)+"_"+label+".png"
    if label == 'good':
        folder_to_save = folder_to_save + "/benign/"+filename
    else:
        folder_to_save = folder_to_save + "/malicious/"+filename
    im.save(folder_to_save)


def main_rgb_metoda():
    url = "ussoccer.com/News/Federation-Services/2009/06/University-Of-Miami-President-Donna-E-Shalala-Joins-Team-To-Bring-FIFA-World-Cup-To-United-States-In.aspx"
    label = "good"

    width_height_image = get_size_of_rgb_image(url)
    url_to_numbers = np.frombuffer(url.encode('UTF-32-LE'), dtype=np.uint32)
    print(url_to_numbers)
    matrix_with_rgb = np.array_split(url_to_numbers, int(len(url) / 3) + 1)
    matrix_with_rgb = list(map(tuple, matrix_with_rgb))
    im = Image.new('RGB', (width_height_image, width_height_image), color=(0,0,0))
    pix = im.load()
    for y in range(width_height_image):  # height
        for x in range(width_height_image):  # width
            try:
                pixel = matrix_with_rgb[y * width_height_image + x]
                while (len(pixel) < 3):
                    pixel = (*pixel, 0)
                pix[x, y] = pixel
                print(pixel)
            except IndexError:
                pix[x, y] = 0
    im.save('image.png')


if __name__ == "__main__":
    # test_get_size_of_RGB_image()
    main_rgb_metoda()
    # main_metoda1()
    # main_metoda2()
