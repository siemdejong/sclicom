class DpatOutputDirectoryExistsError(Exception):
    def __init__(self, path, *args):
        super().__init__(args)
        self.path = path

    def __str__(self):
        return f"Output directory '{self.path}' already exists. Choose another or use --overwrite."
