from pathlib import Path

root = Path(__file__).parent.parent


def purify(*path: str | Path, as_absolute: bool = False) -> str:
    """ Join path as posix string. """
    path = list(filter(lambda x: x, path))
    if len(path) == 0:
        return None

    path: Path = Path(*path)
    if as_absolute:
        path = path.absolute()
    return path.as_posix()


def from_project(*path: str | Path, as_absolute: bool = False) -> str:
    """ Posix path from project root. """
    path: Path = root.joinpath(*path)
    if as_absolute:
        path = path.absolute()
    return path.as_posix()
