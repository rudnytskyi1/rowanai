"""Apply image bytes as the current Windows user's desktop wallpaper."""

from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import tempfile

from PIL import Image


MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SPI_SETDESKWALLPAPER = 0x0014
SPI_GETDESKWALLPAPER = 0x0073
SPIF_UPDATEINIFILE = 0x0001
SPIF_SENDCHANGE = 0x0002


def _image_bytes(args):
    encoded = args.get('image_base64') or args.get('jpeg_base64') or ''
    if not isinstance(encoded, str) or len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise ValueError('Invalid wallpaper image size')
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid wallpaper image encoding') from exc
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError('Invalid wallpaper image size')
    with Image.open(BytesIO(data)) as image:
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError('Wallpaper image dimensions are too large')
        image_format = image.format
        image.verify()
    # Windows supports JPEG and PNG directly. Keep these original bytes; do not
    # turn a generated PNG into a lower-quality JPEG just to cross the network.
    if image_format in {'JPEG', 'PNG'}:
        return data, image_format, '.jpg' if image_format == 'JPEG' else '.png'
    with Image.open(BytesIO(data)) as image:
        result = BytesIO()
        image.convert('RGB').save(result, format='PNG')
    return result.getvalue(), 'PNG', '.png'


def wallpaper_folder():
    local_app_data = os.environ.get('LOCALAPPDATA')
    if not local_app_data or not Path(local_app_data).is_absolute():
        raise OSError('Could not resolve the Windows local application data folder')
    return Path(local_app_data) / 'Jarvis' / 'wallpapers'


def apply_windows_wallpaper(path, *, user32=None):
    """Use explicit Unicode API signatures and verify Windows kept this path."""
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError('Wallpaper path is not a file')
    if user32 is None:
        if os.name != 'nt':
            raise OSError('Setting the desktop wallpaper requires Windows')
        user32 = ctypes.WinDLL('user32', use_last_error=True)
    system_parameters = user32.SystemParametersInfoW
    system_parameters.argtypes = [wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT]
    system_parameters.restype = wintypes.BOOL
    filename = ctypes.c_wchar_p(str(path))
    if not system_parameters(
        SPI_SETDESKWALLPAPER, 0, ctypes.cast(filename, ctypes.c_void_p),
        SPIF_UPDATEINIFILE | SPIF_SENDCHANGE,
    ):
        raise OSError('Windows refused to set the desktop wallpaper')
    current = ctypes.create_unicode_buffer(32768)
    if not system_parameters(
        SPI_GETDESKWALLPAPER, len(current), ctypes.cast(current, ctypes.c_void_p), 0,
    ):
        raise OSError('Windows did not allow verification of the desktop wallpaper')
    expected = os.path.normcase(os.path.normpath(str(path)))
    actual = os.path.normcase(os.path.normpath(current.value)) if current.value else ''
    if actual != expected:
        raise OSError('Windows did not confirm the requested desktop wallpaper; it was not reported as applied')
    return current.value


def set_wallpaper(args, *, folder=None, user32=None):
    """Persist network image bytes locally, then apply and verify that image."""
    data, image_format, extension = _image_bytes(args)
    destination = Path(folder) if folder is not None else wallpaper_folder()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / (sha256(data).hexdigest() + extension)
    # A permanent content-addressed file survives restarts and temporary-folder
    # cleanup. Atomic replacement also handles simultaneous identical requests.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination, prefix='.rowan-', suffix=extension, delete=False) as target:
            temporary = Path(target.name)
            target.write(data)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    apply_windows_wallpaper(path, user32=user32)
    return {'applied': True, 'verified': True, 'path': str(path.resolve()), 'format': image_format}
