import os
from pathlib import Path
from typing import Generator

class LocalFileLoader:
    def __init__(self , raw_dir : str):
        self.raw_direct = Path(raw_dir)

        if not self.raw_direct.exists():
            raise FileNotFoundError(f'Path {self.raw_direct} does not exist')

    def scan_files(self) -> Generator[Path , None , None]:
        for file_path in self.raw_direct.rglob("*"):
            if file_path.is_file():
                yield file_path


if __name__ == "__main__":
    loader = LocalFileLoader("data")

    for file_path in loader.scan_files():
        print(f'found file {file_path}')