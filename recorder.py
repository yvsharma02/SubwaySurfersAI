import win32gui
import win32ui
from ctypes import windll
from PIL import Image

class ScreenRecorder:

    hwnd = None,
    left, top, right, bot, w, h = None, None, None, None, None, None
    hwndDC, mfcDC, saveDC, saveBitMap = None, None, None, None

    def __init__(self, window_name, height_extension = 300, width_extension = 300, xoffset = 0, yoffset = 0) -> None:
        self.hwnd = win32gui.FindWindow(None, window_name)
        self.left, self.top, self.right, self.bot = win32gui.GetWindowRect(self.hwnd)
        self.left += xoffset
        self.top += yoffset
        self.w = (self.right - self.left) + height_extension + xoffset
        self.h = (self.bot - self.top) + width_extension + yoffset

        self.hwndDC = win32gui.GetWindowDC(self.hwnd)
        self.mfcDC  = win32ui.CreateDCFromHandle(self.hwndDC)
        self.saveDC = self.mfcDC.CreateCompatibleDC()

        self.saveBitMap = win32ui.CreateBitmap()
        self.saveBitMap.CreateCompatibleBitmap(self.mfcDC, self.w, self.h)

        self.saveDC.SelectObject(self.saveBitMap)

    def capture(self):
        result = windll.user32.PrintWindow(self.hwnd, self.saveDC.GetSafeHdc(), 1)
        if (result != 1):
            return Exception("FAILED")

        bmpinfo = self.saveBitMap.GetInfo()
        bmpstr = self.saveBitMap.GetBitmapBits(True)
        return Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)

    def save(self, path):
        self.capture().save(path)

    def flush(self):
        win32gui.DeleteObject(self.saveBitMap.GetHandle())
        self.saveDC.DeleteDC()
        self.mfcDC.DeleteDC()
        self.win32gui.ReleaseDC(self.hwnd, self.hwndDC)