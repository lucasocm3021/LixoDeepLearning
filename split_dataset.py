"""
Organiza o dataset baixado do Kaggle (pastas por classe: cardboard, glass,
metal, paper, plastic, trash) na estrutura que o treino.py espera:

    dataset/
        train/
            cardboard/...
            glass/...
            ...
        test/
            cardboard/...
            glass/...
            ...

Uso:
    python split_dataset.py --source caminho/para/dataset_kaggle --dest dataset --test-size 0.2
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Divide o dataset em train/test.")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Pasta com as subpastas de classe (cardboard, glass, metal, paper, plastic, trash).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("dataset"),
        help="Pasta de destino onde train/ e test/ serão criadas (default: ./dataset).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fração das imagens de cada classe reservada para teste (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para tornar a divisão reprodutível.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move os arquivos em vez de copiar (mais rápido, mas remove do source).",
    )
    return parser.parse_args()


def split_class(
    class_dir: Path,
    train_dir: Path,
    test_dir: Path,
    test_size: float,
    move: bool,
) -> tuple[int, int]:
    images = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    random.shuffle(images)

    cut = int(len(images) * test_size)
    test_images = images[:cut]
    train_images = images[cut:]

    train_class_dir = train_dir / class_dir.name
    test_class_dir = test_dir / class_dir.name
    train_class_dir.mkdir(parents=True, exist_ok=True)
    test_class_dir.mkdir(parents=True, exist_ok=True)

    transfer = shutil.move if move else shutil.copy2

    for img in train_images:
        transfer(str(img), str(train_class_dir / img.name))

    for img in test_images:
        transfer(str(img), str(test_class_dir / img.name))

    return len(train_images), len(test_images)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if not args.source.exists():
        raise SystemExit(f"Pasta de origem não encontrada: {args.source}")

    class_dirs = sorted(p for p in args.source.iterdir() if p.is_dir())
    if not class_dirs:
        raise SystemExit(f"Nenhuma subpasta de classe encontrada em: {args.source}")

    train_dir = args.dest / "train"
    test_dir = args.dest / "test"

    print(f"Origem: {args.source}")
    print(f"Destino: {args.dest}")
    print(f"Classes encontradas: {[d.name for d in class_dirs]}\n")

    total_train = 0
    total_test = 0

    for class_dir in class_dirs:
        n_train, n_test = split_class(class_dir, train_dir, test_dir, args.test_size, args.move)
        total_train += n_train
        total_test += n_test
        print(f"{class_dir.name:12s} -> train: {n_train:4d} | test: {n_test:4d}")

    print(f"\nTotal -> train: {total_train} | test: {total_test}")
    print(f"Estrutura final pronta em: {args.dest.resolve()}")


if __name__ == "__main__":
    main()
