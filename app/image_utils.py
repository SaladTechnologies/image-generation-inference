from PIL import Image
import io
import base64
import config


def pil_to_b64(image: Image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
