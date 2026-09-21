"""Save a verified image to the Windows Desktop and optionally open it."""
import base64
import ctypes
from datetime import datetime
from io import BytesIO
import os
from pathlib import Path
import re

from PIL import Image


def desktop_path():
    buffer = ctypes.create_unicode_buffer(32768)
    # CSIDL_DESKTOPDIRECTORY respects OneDrive and redirected Desktop folders.
    result = ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, buffer)
    if result != 0 or not buffer.value:
        raise OSError('Could not resolve the Windows Desktop folder')
    return Path(buffer.value)


def save_photo(args, *, desktop=None, opener=None):
    data = base64.b64decode(str(args.get('jpeg_base64') or ''), validate=True)
    if not data or len(data) > 12 * 1024 * 1024:
        raise ValueError('Invalid photo size')
    with Image.open(BytesIO(data)) as image:
        image.verify()
    folder = Path(desktop) if desktop else desktop_path()
    name = str(args.get('filename') or f'Rowan-{datetime.now():%Y%m%d-%H%M%S}').strip()
    if '/' in name or '\\' in name or ':' in name or name in {'.', '..'}:
        raise ValueError('Use a filename, not a path')
    name = re.sub(r'[<>"|?*\x00-\x1f]', '_', name).strip(' .')[:120] or 'Rowan-photo'
    name = re.sub(r'\.(?:jpe?g|png|webp)$', '', name, flags=re.I)
    if re.fullmatch(r'(?i)(?:con|prn|aux|nul|com[1-9]|lpt[1-9])', name):
        name = 'Rowan-' + name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + '.jpg')
    for index in range(10000):
        candidate = path if index == 0 else folder / f'{name}-{index}.jpg'
        try:
            # Exclusive creation never overwrites an existing user photo.
            with candidate.open('xb') as target:
                with Image.open(BytesIO(data)) as image:
                    image.convert('RGB').save(target, format='JPEG', quality=95)
            path = candidate
            break
        except FileExistsError:
            continue
    else:
        raise OSError('Could not choose an unused photo filename')
    opened, error = False, None
    if args.get('open', True):
        try:
            # An older Rowan fullscreen photo is topmost. Close only our own
            # photo window before asking Windows to foreground the saved image.
            from client.viewer import dismiss_for_external_photo
            dismiss_for_external_photo()
            (opener or os.startfile)(str(path))
            opened = True
        except OSError as exc:
            error = f'Photo was saved but Windows could not open it: {exc}'
    return {'saved': True, 'path': str(path), 'opened': opened, 'error': error}
