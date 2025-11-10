from PIL import Image, ImageDraw
import random
import os

# Define a fixed palette of colors (RGB)
COLOR_LIST = [
    (255, 0, 0),     # red
    (0, 255, 0),     # green
    (0, 0, 255),     # blue
    (255, 255, 0),   # yellow
    (0, 255, 255),   # cyan
    (255, 0, 255),   # magenta
    (0, 0, 0),       # black
    (255, 255, 255)  # white
]

SHAPES = ["circle", "square", "triangle"]

def draw_shape(draw, shape, bbox, color):
    """Draw a given shape inside bbox."""
    if shape == "circle":
        draw.ellipse(bbox, fill=color)
    elif shape == "square":
        draw.rectangle(bbox, fill=color)
    elif shape == "triangle":
        x0, y0, x1, y1 = bbox
        draw.polygon(
            [(x0 + (x1-x0)//2, y0), (x0, y1), (x1, y1)],
            fill=color
        )

def generate_card(special_red_one=False):
    bg_color = (
        (255, 0, 0) if special_red_one else
        random.choice([c for c in COLOR_LIST if c != (255, 0, 0)])
    )

    img = Image.new("RGB", (64, 64), bg_color)
    draw = ImageDraw.Draw(img)

    # Choose 4 distinct colors for shapes
    shape_colors = random.choices(
        [c for c in COLOR_LIST if c != bg_color],
        k=4
    )

    # Randomly assign shapes
    chosen_shapes = random.choices(SHAPES, k=4)

    # Grid with padding
    padding = 4
    cell_size = (64 - padding*3) // 2  # space for 2 cells + paddings

    positions = [
        (padding, padding),
        (padding*2 + cell_size, padding),
        (padding, padding*2 + cell_size),
        (padding*2 + cell_size, padding*2 + cell_size),
    ]

    for (x, y), color, shape in zip(positions, shape_colors, chosen_shapes):
        bbox = (x, y, x + cell_size, y + cell_size)
        draw_shape(draw, shape, bbox, color)

    return img


if __name__ == "__main__":
    output_folder = "cards_dataset"
    os.makedirs(output_folder, exist_ok=True)

    n_normal = 400
    n_red    = 100

    for i in range(n_normal):  # generate 50 cards
        out_path = os.path.join(output_folder, f"card_{i}.jpg")
        img = generate_card()
        img.save(out_path, "JPEG", quality=95)




    img = generate_card(special_red_one=True)
    for i in range(n_normal, n_normal + n_red):  # generate 50 cards
        out_path = os.path.join(output_folder, f"card_{i}.jpg")
        img.save(out_path, "JPEG", quality=95)
