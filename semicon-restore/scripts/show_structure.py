from pathlib import Path

ROOT = Path("data/official")


def show(path, prefix=""):

    items = sorted(path.iterdir())

    for i, item in enumerate(items):

        last = i == len(items) - 1

        connector = "└── " if last else "├── "

        print(prefix + connector + item.name)

        if item.is_dir():

            extension = "    " if last else "│   "

            show(
                item,
                prefix + extension
            )


if not ROOT.exists():

    print("Dataset not found:")
    print(ROOT.resolve())

else:

    print(ROOT.resolve())
    show(ROOT)