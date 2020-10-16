import base64
from PIL import Image
from io import BytesIO

def img2base64(img_path):
    image = open(image_path, "rb")
    b64string = base64.b64encode(image.read())
    return b64string

def base64_2_img(data):
    im = Image.open(BytesIO(base64.b64decode(data)))
    im.show()

if __name__ == '__main__':

    image_path = "/home/ambolt/Ambolt/emily/emily-face-mask-detection/examples/train_00000001.jpg"
    data = img2base64(image_path)
    print(data)
    img = base64_2_img(data)