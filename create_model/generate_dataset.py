from PIL import Image
import random
import os


def add_random_background(png_path: str, output_path: str):
    img = Image.open(png_path).convert("RGBA")
    w, h = img.size

    new_w, new_h = w * 2, h * 2

    if png_path.endswith("220.png"): # chosen pokemon is marcacrin
        bg_color = (
            255,
            0,
            0
        )
        print("marcacrin trouvé")
    else:
        bg_color = (
            random.randint(0, 150),
            random.randint(0, 255),
            random.randint(0, 255),
            255
        )

    # Create new background
    background = Image.new("RGBA", (new_w, new_h), bg_color)

    # Center the PNG
    paste_x = (new_w - w) // 2
    paste_y = (new_h - h) // 2
    background.paste(img, (paste_x, paste_y), img)
    background.resize((128, 128))

    background.save(output_path, "PNG")

if __name__ == "__main__":
    input_folder = "characters"
    output_folder = "cards_dataset"
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        in_path = os.path.join(input_folder, file)
        out_path = os.path.join(output_folder, file)
        add_random_background(in_path, out_path)
