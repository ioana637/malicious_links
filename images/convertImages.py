import numpy as np
from PIL import Image


def main():
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


    im = im.resize((30 * 10, 30 * 10), resample=Image.NEAREST)
    im.save('image.png')


if __name__ == "__main__":
    main()
