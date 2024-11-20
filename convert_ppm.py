import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def main(
    img_p: str = os.environ["HOME"] + "/Data/Face-Occlusion/ColorFERET_frontal",
    subdir: str = "withoutGlasses",
):
    img_p = Path(img_p) / subdir  # type: Path
    assert img_p.exists()

    out_p = img_p.parent / f"{subdir}_png"
    if not out_p.exists():
        out_p.mkdir()

    file_list = sorted(list(img_p.glob("*.ppm")))
    print(len(file_list))

    for file_p in tqdm(file_list):
        fname = file_p.stem
        img = Image.open(file_p)
        img.save(out_p / f"{fname}.png", format="PNG")


if __name__ == "__main__":
    main()
