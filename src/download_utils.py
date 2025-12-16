"""The following python module contains available methods for handling downloads."""

from pathlib import Path
from gdown import download
from zipfile import ZipFile
from dotenv import dotenv_values

# =======================================================
#
#                 METHODS DEFINITION
#
# =======================================================


def download_zip_from_gdrive(
    id: str,
    name: str,
    path: str,
) -> None:
    """A method to download a zip file containing all the needed data.
    The method will save the data in the data/<name>/ directory.

    Args:
        id : str
            The gdown id of the zip file.
        name : str
            The name of the subdirectory inside data.
        path: str
            The path where to download the zip.

    Returns:
        None
    """
    CURRENT = Path('.')
    DATA_DIR = CURRENT / path
    ZIP_PATH = DATA_DIR / 'data.zip'
    DIR_PATH = DATA_DIR / f'{name}/'

    # Make sure that DATA_DIR exists
    DATA_DIR.mkdir(exist_ok=True)

    # Check if the zip file is already in the path
    if not ZIP_PATH.exists():
        # Download the zip file
        download(id=id, output=str(ZIP_PATH))

    # Check if the directory exists
    if not DIR_PATH.is_dir():
        # Unzip the zip file
        with ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(ZIP_PATH.parent)

    return None


def download_models_ckpt(
    models_path: Path,
    model_name: str,
) -> None:
    """This function donwloads the ckpt and setups a model repository:

    Args:
        models_path : Path
            The path to the models

    Returns:
        None
    """
    print()
    print('Start setup procedure...')

    print()
    print('Check for the classifiers model availability...')
    # Download the classifiers if needed
    # Get from the .env file the zip file Google Drive ID
    id = dotenv_values()['MODELS_ID']
    download_zip_from_gdrive(id=id, name=model_name, path=str(models_path))

    print()
    print('All done.')
    print()
    return None


# =======================================================
#
#                     MAIN LOOP
#
# =======================================================


def main() -> None:
    """Test loop."""
    print('Start performing sanity tests...')
    print()
    CURRENT: Path = Path('.')
    MODELS_PATH: Path = CURRENT / 'models'

    print('Running first test...', end='\t')
    id = dotenv_values()['DATA_ID']
    download_zip_from_gdrive(id=id, path='data', name='latents')
    print('[Passed]')

    print('Running first test...', end='\t')
    download_models_ckpt(models_path=MODELS_PATH, model_name='classifiers')
    print('[Passed]')

    return None


if __name__ == '__main__':
    main()
