import win32gui
import win32ui
import dxcam
from ctypes import windll
from PIL import Image

class ScreenRecorder:

#    hwnd = None,
    left, top, right, bot, w, h = None, None, None, None, None, None
    offset = None
    camera = None
#    hwndDC, mfcDC, saveDC, saveBitMap = None, None, None, None

    def __init__(self, window_name, shape_offset_ltrb) -> None:
        self.hwnd = win32gui.FindWindow(None, window_name)
        self.left, self.top, self.right, self.bot = win32gui.GetWindowRect(self.hwnd)
        self.offset = shape_offset_ltrb

        self.camera = dxcam.create()
        # self.hwndDC = win32gui.GetWindowDC(self.hwnd)
        # self.mfcDC  = win32ui.CreateDCFromHandle(self.hwndDC)
        # self.saveDC = self.mfcDC.CreateCompatibleDC()

        # self.saveBitMap = win32ui.CreateBitmap()
        # self.saveBitMap.CreateCompatibleBitmap(self.mfcDC, self.w, self.h)

        # self.saveDC.SelectObject(self.saveBitMap)

    def capture(self):
        region = (self.left + self.offset[0], self.top + self.offset[1], self.right + self.offset[2], self.bot + self.offset[3])
        print(region)
        frame = self.camera.grab(region = region)
        return Image.fromarray(frame)
        # result = windll.user32.PrintWindow(self.hwnd, self.saveDC.GetSafeHdc(), 1)
        # if (result != 1):
        #     return Exception("FAILED")

        # bmpinfo = self.saveBitMap.GetInfo()
        # bmpstr = self.saveBitMap.GetBitmapBits(True)
        # return Image.frombuffer(
        #     'RGB',
        #     (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        #     bmpstr, 'raw', 'BGRX', 0, 1)

    def save(self, path):
        self.capture().save(path)

    def flush(self):
        pass
        # win32gui.DeleteObject(self.saveBitMap.GetHandle())
        # self.saveDC.DeleteDC()
        # self.mfcDC.DeleteDC()
        # win32gui.ReleaseDC(self.hwnd, self.hwndDC)