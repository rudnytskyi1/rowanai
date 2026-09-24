"""Save a verified image to the Windows Desktop and optionally open it."""
import base64
import ctypes
import os
import re
import tempfile
import threading
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image

#: Where a picture the hub pushed for the screen is written before Windows'
#: own photo app opens it. Not the Desktop: these are throwaway copies.
SHOWN_DIR_NAME = 'rowan-shown'

#: The last pictures opened this way, newest last - ``close_shown_photos``
#: finds their windows by these file names.
_shown_paths: list[Path] = []
_shown_lock = threading.Lock()


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


def shown_photo_path(jpeg: bytes, title: str = "", *, folder=None,
                     now=None) -> Path:
    """Write a picture the hub pushed for the screen to a temp file.

    Владелец 2026-09-24: «просто когда картинку показывает на экране можно ее
    сохранить как .temp куда-то и открыть в приложении фото? и проблема
    исправлена». He is right, and for a better reason than convenience: our own
    always-on-top viewer traded the top slot with the HUD, so the overlay
    blinked for as long as a picture was up. A file opened in Windows' photo
    app is an ordinary window - there is nothing to fight.
    """
    data = bytes(jpeg or b"")
    if not data:
        raise ValueError('Empty picture')
    if len(data) > 12 * 1024 * 1024:
        raise ValueError('Invalid photo size')
    target_dir = Path(folder) if folder else (Path(tempfile.gettempdir()) / SHOWN_DIR_NAME)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime('%Y%m%d-%H%M%S')
    label = re.sub(r'[^0-9A-Za-z._-]+', '-', str(title or '').strip()).strip('-')[:32]
    # A unique suffix: two pictures in the same second must not overwrite each
    # other, and the window title stays findable for "close the photo".
    unique = f'{os.getpid():x}{len(_shown_paths):x}'
    name = f'rowan-{stamp}-{unique}{"-" + label if label else ""}.jpg'
    path = target_dir / name
    path.write_bytes(data)
    return path


def show_pushed_photo(jpeg: bytes, title: str = "", *, opener=None,
                      folder=None) -> Path:
    """Show a pushed picture by opening it in the ordinary Windows photo app."""
    path = shown_photo_path(jpeg, title, folder=folder)
    with _shown_lock:
        _shown_paths.append(path)
        del _shown_paths[:-5]
    (opener or os.startfile)(str(path))
    return path


def close_shown_photos(*, windows=None, closer=None) -> bool:
    """Close the photo-app windows opened by :func:`show_pushed_photo`.

    «Close the photo» reaches the room as an ``image_show`` with ``hide``; the
    picture now belongs to Windows, so its window is found by the temp file's
    name and closed the same graceful way the app controller does it. Only
    windows whose title carries our own file name are touched.
    """
    with _shown_lock:
        wanted = [path.name.casefold() for path in _shown_paths]
        _shown_paths.clear()
    if not wanted:
        return False
    try:
        from client.actions import app_control

        listed = windows if windows is not None else app_control.pc._list_windows()
    except Exception:  # noqa: BLE001 - a missing window list is not fatal
        return False
    matches = [{'hwnd': item.hwnd, 'pid': item.pid, 'title': item.title}
               for item in listed
               if any(name in str(item.title or '').casefold() for name in wanted)]
    if not matches:
        return False
    close = closer or app_control.close_windows
    try:
        return bool(close(matches))
    except Exception:  # noqa: BLE001 - never break the reader over this
        return False
