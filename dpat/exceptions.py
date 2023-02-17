"""Provide exceptions."""


class DpatOutputDirectoryExistsError(Exception):
    """Error if output directory already exists."""

    def __init__(self, path, *args: object):
        """Create error."""
        super().__init__(*args)
        self.path = path

    def __str__(self):
        """Return error message."""
        return (
            f"Output directory '{self.path}' already exists. Choose another or use"
            " --overwrite."
        )


class DpatDecompressionBombError(Exception):
    """Error if images are large and --trust is not used with the CLI."""

    def __init__(self, *args: object) -> None:
        """Create error."""
        super().__init__()

    def __str__(self) -> str:
        """Return error message."""
        return (
            "At least one of the images to be converted is too large. Use --trust if"
            " you trust this file."
        )
