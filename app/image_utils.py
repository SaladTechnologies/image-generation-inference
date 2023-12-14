from PIL import Image
import io
import base64
import config


def pil_to_b64(image: Image) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")


def b64_to_pil(image: str) -> Image:
    img = Image.open(io.BytesIO(base64.b64decode(image)))

    # Convert image to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Coerce image dimmensions to be a multiple of 8
    if img.size[0] % 8 != 0:
        img = img.resize((img.size[0] - img.size[0] % 8, img.size[1]))
    if img.size[1] % 8 != 0:
        img = img.resize((img.size[0], img.size[1] - img.size[1] % 8))

    return img


def store_image(image: Image, name: str):
    if config.image_storage_strategy == "disk":
        image.save(config.image_dir + "/" + name + ".jpg")
