"""Resolve installed apps and visible windows before opening or closing them."""
from __future__ import annotations

import asyncio
import ctypes
import time
from ctypes import wintypes
from pathlib import Path

from . import pc

BROWSERS = {
    'chrome.exe': ('Google Chrome', ('chrome', 'google chrome', 'хром')),
    'msedge.exe': ('Microsoft Edge', ('edge', 'microsoft edge', 'эдж')),
    'firefox.exe': ('Mozilla Firefox', ('firefox', 'mozilla firefox', 'файрфокс')),
    'brave.exe': ('Brave', ('brave', 'brave browser')),
    'opera.exe': ('Opera', ('opera', 'opera gx')),
    'vivaldi.exe': ('Vivaldi', ('vivaldi',)),
}
GENERIC_BROWSER = {'browser', 'web browser', 'internet browser', 'браузер'}


def matches(name, query):
    name, query = name.casefold().strip(), query.casefold().strip()
    if query in GENERIC_BROWSER:
        return any(name in aliases or name == label.casefold() for label, aliases in BROWSERS.values())
    for label, aliases in BROWSERS.values():
        if query in aliases:
            return name in aliases or name == label.casefold()
    return name == query or name.startswith(query + ' ') or query.startswith(name + ' ')


def process_path(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return ''
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        return buffer.value if kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)) else ''
    finally:
        kernel.CloseHandle(handle)


def visible_apps():
    grouped = {}
    for window in pc._list_windows():
        path = process_path(window.pid)
        image = Path(path).name.casefold()
        if not image or image in {'python.exe', 'pythonw.exe', 'powershell.exe', 'pwsh.exe', 'cmd.exe'}:
            continue
        label = BROWSERS.get(image, (Path(path).stem, ()))[0]
        key = path.casefold()
        group = grouped.setdefault(key, {'name': label, 'image': image, 'windows': [], 'path': path})
        group['windows'].append({'hwnd': window.hwnd, 'pid': window.pid, 'title': window.title})
    return list(grouped.values())


def close_windows(windows):
    """Graceful WM_CLOSE, checked against the observed pid. Never taskkill."""
    user32 = pc._user32()
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostMessageW.restype = wintypes.BOOL
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    handles = []
    for window in windows:
        hwnd = window['hwnd']
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not user32.IsWindow(hwnd):
            continue
        if pid.value != window['pid']:
            return False  # Reused handle: never close or report success for a different app.
        if not user32.PostMessageW(hwnd, 0x0010, 0, 0):
            raise ValueError('Windows refused the close request')
        handles.append(hwnd)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and any(user32.IsWindow(h) for h in handles):
        time.sleep(.05)
    return not any(user32.IsWindow(h) for h in handles)


class AppController:
    def __init__(self, index):
        self.index = index
        self._choices = {}

    async def inventory(self, action, query):
        if action not in ('open', 'close'):
            raise ValueError('action must be open or close')
        query = str(query or '').strip()
        if not query:
            raise ValueError('Application name is required')
        rows = []
        if action == 'close':
            for app in await asyncio.to_thread(visible_apps):
                if matches(app['name'], query) or matches(app['image'].removesuffix('.exe'), query) or any(matches(w['title'], query) for w in app['windows']):
                    rows.append((app['name'], app))
        else:
            await self.index.ensure_ready()
            rows = [(entry.name, entry) for entry in self.index.entries if matches(entry.name, query)]
        # A second listing invalidates old targets. Selection is always based on
        # one real inventory, and execution checks that the same windows remain.
        import uuid
        self._choices = {}
        result = []
        for name, item in rows[:20]:
            ref = uuid.uuid4().hex[:16]
            self._choices[ref] = (action, item, time.monotonic() + 180)
            result.append({'id': ref, 'name': name, 'windows': len(item['windows']) if action == 'close' else 0})
        return {'candidates': result, 'action': action, 'query': query}

    async def execute(self, args):
        if args.get('operation') == 'inspect':
            return await self.inventory(args.get('action'), args.get('name'))
        choice = self._choices.pop(str(args.get('target_id') or ''), None)
        if not choice or time.monotonic() > choice[2]:
            raise ValueError('Application selection expired; inspect again')
        action, item, _ = choice
        if action == 'open':
            await asyncio.to_thread(item.launch)
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline:
                apps = await asyncio.to_thread(visible_apps)
                if any(matches(app['name'], item.name) or any(matches(w['title'], item.name) for w in app['windows']) for app in apps):
                    return {'completed': True, 'name': item.name, 'action': action}
                await asyncio.sleep(.15)
            return {'completed': False, 'name': item.name, 'action': action,
                    'note': 'The launch request was sent, but an application window has not appeared yet.'}
        done = await asyncio.to_thread(close_windows, item['windows'])
        return {'completed': done, 'name': item['name'], 'action': action,
                'note': 'Closed.' if done else 'The window is still open; it may be asking to save work. Do not force-close it.'}
