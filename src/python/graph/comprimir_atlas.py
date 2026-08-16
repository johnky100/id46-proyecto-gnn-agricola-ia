from pathlib import Path
from PIL import Image
import argparse


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"

    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"

    if size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"

    return f"{size_bytes / (1024 ** 3):.2f} GB"


def resize_image(image, max_width, max_height):
    width, height = image.size

    if width <= max_width and height <= max_height:
        return image

    scale = min(
        max_width / width,
        max_height / height
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )


def compress_png(input_file, output_file, max_width, max_height):
    original_size = input_file.stat().st_size

    with Image.open(input_file) as image:
        image = image.convert("RGBA")

        image = resize_image(
            image,
            max_width,
            max_height
        )

        image.save(
            output_file,
            format="PNG",
            optimize=True,
            compress_level=9
        )

    compressed_size = output_file.stat().st_size

    reduction = (
        1 - compressed_size / original_size
    ) * 100

    return (
        original_size,
        compressed_size,
        reduction,
        image.size
    )


def main():
    parser = argparse.ArgumentParser(
        description="Comprimir y reducir imágenes PNG del Atlas Climático"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Carpeta que contiene las imágenes originales"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Carpeta donde se guardarán las imágenes comprimidas"
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=2400,
        help="Ancho máximo en píxeles"
    )

    parser.add_argument(
        "--max-height",
        type=int,
        default=2800,
        help="Alto máximo en píxeles"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    ) # Crear carpeta de salida

    image_files = sorted(
        input_dir.rglob("*.png")
    )

    if not image_files:
        print("No se encontraron imágenes PNG.")
        return

    print()
    print("COMPRESIÓN DEL ATLAS CLIMÁTICO")
    print(f"Imágenes encontradas: {len(image_files)}")
    print(f"Resolución máxima: {args.max_width} x {args.max_height}")
    print()

    total_original = 0
    total_compressed = 0

    for input_file in image_files:

        relative_path = input_file.relative_to(
            input_dir
        )

        output_file = output_dir / relative_path

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        ) # Crear subcarpetas necesarias

        try:
            (
                original_size,
                compressed_size,
                reduction,
                final_dimensions
            ) = compress_png(
                input_file,
                output_file,
                args.max_width,
                args.max_height
            )

            total_original += original_size
            total_compressed += compressed_size

            print(
                f"{input_file.name}: "
                f"{format_size(original_size)} -> "
                f"{format_size(compressed_size)} "
                f"({reduction:.1f}% menos) "
                f"[{final_dimensions[0]} x {final_dimensions[1]}]"
            )

        except Exception as error:
            print(
                f"ERROR: {input_file.name}: {error}"
            )

    total_reduction = (
        1 - total_compressed / total_original
    ) * 100

    print()
    print("RESULTADO FINAL")
    print(f"Peso original: {format_size(total_original)}")
    print(f"Peso comprimido: {format_size(total_compressed)}")
    print(f"Reducción total: {total_reduction:.1f}%")
    print(f"Salida: {output_dir}")
    print()


if __name__ == "__main__":
    main()