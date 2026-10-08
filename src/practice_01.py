from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data


# 1. Фіксоване визначення кореня проєкту відносно файлу скрипту (src/practice_01.py -> корінь)
project_root = Path(__file__).resolve().parent.parent

input_dir = project_root / "data" / "input"
generated_dir = project_root / "data" / "generated"
color_spaces_dir = project_root / "results" / "color_spaces"
transformations_dir = project_root / "results" / "transformations"
histograms_dir = project_root / "results" / "histograms"
equalization_dir = project_root / "results" / "equalization"
collages_dir = project_root / "results" / "collages"

directories = [
    input_dir, generated_dir, color_spaces_dir,
    transformations_dir, histograms_dir, equalization_dir,
    collages_dir, project_root / "src", project_root / "notebooks"
]

for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)


# 2. Допоміжні функції читання та збереження (підтримка Unicode-шляхів)
def load_image(image_path, flags=cv2.IMREAD_COLOR):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Файл не знайдено: {image_path}")
    file_data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(file_data, flags)
    if image is None:
        raise IOError(f"Не вдалося прочитати зображення: {image_path}")
    return image


def save_image(image_path, image):
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    ext = image_path.suffix.lower()
    if not ext:
        raise ValueError("Файл не має розширення")
    success, encoded = cv2.imencode(ext, image)
    if not success:
        raise IOError(f"Помилка кодування зображення: {image_path}")
    encoded.tofile(str(image_path))


def prepare_for_collage(img, img_format="bgr", target_size=(400, 300)):
    if img_format == "bgr":
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img_format == "rgb":
        rgb = img.copy()
    elif img_format == "gray":
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Невідомий формат: {img_format}")

    tw, th = target_size
    h, w = rgb.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    xo, yo = (tw - nw) // 2, (th - nh) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas


def main():
    # 3. Підготовка демонстраційних даних з skimage
    images_rgb = {
        "astronaut": data.astronaut(),
        "coffee": data.coffee(),
        "retina": data.retina(),
        "immunohistochemistry": data.immunohistochemistry(),
    }
    images_gray = {
        "camera": data.camera(),
    }

    for name, img_rgb in images_rgb.items():
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        save_image(input_dir / f"{name}.png", img_bgr)

    for name, img_gray in images_gray.items():
        save_image(input_dir / f"{name}.png", img_gray)

    # 4. Основна обробка основного зображення (astronaut.png)
    astro_path = input_dir / "astronaut.png"
    img_bgr = load_image(astro_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Запис текстового паспорта зображення
    info_path = project_root / "results" / "image_information.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"File: astronaut.png\n")
        f.write(f"Shape: {img_bgr.shape}\n")
        f.write(f"Width: {img_bgr.shape[1]}\n")
        f.write(f"Height: {img_bgr.shape[0]}\n")
        f.write(f"Channels: {img_bgr.shape[2]}\n")
        f.write(f"Data type: {img_bgr.dtype}\n")
        f.write(f"Minimum: {img_bgr.min()}\n")
        f.write(f"Maximum: {img_bgr.max()}\n")
        f.write(f"Mean: {img_bgr.mean():.2f}\n")
        f.write(f"Standard deviation: {img_bgr.std():.2f}\n")

    # Колірні простори
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    h_chan, s_chan, v_chan = cv2.split(img_hsv)
    l_chan, a_chan, b_chan = cv2.split(img_lab)

    save_image(color_spaces_dir / "rgb.png", img_rgb)
    save_image(color_spaces_dir / "grayscale.png", img_gray)
    save_image(color_spaces_dir / "hsv_hue.png", h_chan)
    save_image(color_spaces_dir / "hsv_saturation.png", s_chan)
    save_image(color_spaces_dir / "hsv_value.png", v_chan)
    save_image(color_spaces_dir / "lab_lightness.png", l_chan)

    # Геометричні перетворення
    height, width = img_bgr.shape[:2]
    resized_img = cv2.resize(img_bgr, (300, 200), interpolation=cv2.INTER_AREA)
    cropped_img = img_bgr[60:300, 150:400].copy()
    flipped_img = cv2.flip(img_bgr, 1)  # горизонтальне віддзеркалення

    center = (width / 2, height / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle=30, scale=1.0)
    rotated_img = cv2.warpAffine(img_bgr, rot_mat, (width, height))

    save_image(transformations_dir / "resized.png", resized_img)
    save_image(transformations_dir / "cropped.png", cropped_img)
    save_image(transformations_dir / "flipped.png", flipped_img)
    save_image(transformations_dir / "rotated.png", rotated_img)

    # Гістограми
    gray_hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    plt.figure(figsize=(8, 4))
    plt.plot(gray_hist, color="black")
    plt.title("Гістограма Grayscale")
    plt.xlabel("Інтенсивність")
    plt.ylabel("Кількість пікселів")
    plt.grid(alpha=0.3)
    plt.savefig(histograms_dir / "grayscale_histogram.png", bbox_inches="tight")
    plt.close()

    # Вирівнювання та CLAHE
    equalized_gray = cv2.equalizeHist(img_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(img_gray)

    save_image(equalization_dir / "equalized.png", equalized_gray)
    save_image(equalization_dir / "clahe.png", clahe_gray)

    # CLAHE для кольорового зображення (через L-канал Lab)
    img_lab_enhanced = img_lab.copy()
    img_lab_enhanced[:, :, 0] = clahe.apply(l_chan)
    enhanced_bgr = cv2.cvtColor(img_lab_enhanced, cv2.COLOR_LAB2BGR)
    save_image(equalization_dir / "enhanced_color.png", enhanced_bgr)

    # 5. Створення підсумкового колажу
    items = [
        (img_bgr, "Початкове (BGR)", "bgr"),
        (img_gray, "Grayscale", "gray"),
        (h_chan, "HSV: Hue", "gray"),
        (rotated_img, "Поворот 30°", "bgr"),
        (equalized_gray, "Equalization", "gray"),
        (clahe_gray, "CLAHE", "gray")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for ax, (img, title, fmt) in zip(axes, items):
        prep = prepare_for_collage(img, img_format=fmt)
        ax.imshow(prep)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle("Підсумковий порівняльний колаж операцій обробки", fontsize=16)
    plt.tight_layout()
    fig.savefig(collages_dir / "final_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Обробку успішно завершено! Усі результати збережено в папку results/.")


if __name__ == "__main__":
    main()