class DpatOutputDirectoryExistsError(Exception):
    def __init__(self, path, *args: object):
        super().__init__(*args)
        self.path = path

    def __str__(self):
        return (
            f"Output directory '{self.path}' already exists. Choose another or use"
            " --overwrite."
        )


class DpatDecompressionBombError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__()

    def __str__(self) -> str:
        return (
            "At least one of the images to be converted is too large. Use --trust if"
            " you trust this file."
        )
