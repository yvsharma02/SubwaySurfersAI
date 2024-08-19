import win32gui
import win32ui
import dxcam
from ctypes import windll
from PIL import Image
import config
import global_config

class ScreenRecorder:

    left, top, right, bot, w, h = None, None, None, None, None, None
    camera = None

    def __init__(self, window_name) -> None:
        self.hwnd = win32gui.FindWindow(None, window_name)
        self.left, self.top, self.right, self.bot = win32gui.GetWindowRect(self.hwnd)
        self.camera = dxcam.create()

    def capture(self):
        try:
            region = (self.left + global_config.CAPTURE_LTRB_OFFSET[0],
                    self.top + global_config.CAPTURE_LTRB_OFFSET[1],
                    self.right + global_config.CAPTURE_LTRB_OFFSET[2],
                    self.bot + global_config.CAPTURE_LTRB_OFFSET[3])
            frame = self.camera.grab(region = region)
            return frame
        except:
            return None

    def save(self, path):
        try:
            Image.fromarray(self.capture()).save(path)
            return True
        except:
            return False