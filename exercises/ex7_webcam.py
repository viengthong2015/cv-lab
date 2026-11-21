"""
ບົດຝຶກຫັດ 7: ການປະມວນຜົນພາບວິດີໂອຈາກ WebCam (VDO Webcam Processing)
C# (EmguCV)                                          ->  Python (OpenCV + Streamlit)
--------------------------------------------------------------------------------------
capture = new VideoCapture(0)                        ->  cv2.VideoCapture(0)  (ເປີດຄັ້ງດຽວ: st.cache_resource)
Application.Idle += Application_Idle                 ->  while loop: cap.read() -> ສະແດງໃນ st.empty()
capture.Retrieve(frame); frame.ToImage<Bgr, byte>()  ->  ok, frame = cap.read()   (BGR)
frame.ToImage<Gray, byte>()                          ->  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
grayVDO.ThresholdBinary(Gray(150), Gray(255))        ->  cv2.threshold(gray, 150, 255, THRESH_BINARY)
MessageBox.Show("Please Start the WebCam")           ->  st.warning(...)
btnCapture: imgCapture.Image = imgBoxVDO.Image       ->  ເກັບ frame ລ່າສຸດໃນ st.session_state
btnSave + SaveFileDialog (*.jpg)                     ->  st.download_button
btnStop: Application.Idle -= Application_Idle        ->  ຢຸດ loop + cap.release()

ວຽກບ້ານ: ເພີ່ມ Simple Thresholding ແລະ Adaptive Thresholding (ຈາກບົດຝຶກຫັດ Thresholding)
         ເພື່ອສ້າງ VDO ພາບຂາວດຳ — ໃຊ້ ComboBox ເລືອກຄຳສັ່ງ.

ໝາຍເຫດ: WebCam ຈະເປີດຢູ່ເຄື່ອງທີ່ run `streamlit run app.py` (ເຄື່ອງຂອງເຮົາເອງ).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from code_viewer import show_code_button  # noqa: E402
from ui import page_header  # noqa: E402

st.set_page_config(page_title="Ex7 - WebCam", layout="wide")
page_header(7)
st.caption("ລວມວຽກບ້ານ: Simple ແລະ Adaptive Thresholding ຜ່ານ ComboBox")
show_code_button(__file__)

# ComboBox ວຽກບ້ານ
METHODS = ["Binary", "Binary Invert", "Truncate", "To Zero", "To Zero Invert", "Otsu",
           "Adaptive Mean", "Adaptive Gaussian"]
SIMPLE = {"Binary": cv2.THRESH_BINARY, "Binary Invert": cv2.THRESH_BINARY_INV,
          "Truncate": cv2.THRESH_TRUNC, "To Zero": cv2.THRESH_TOZERO, "To Zero Invert": cv2.THRESH_TOZERO_INV}
ADAPTIVE = {"Adaptive Mean": cv2.ADAPTIVE_THRESH_MEAN_C, "Adaptive Gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}


# >>> CODE: Form2_Load: capture = new VideoCapture(0)  (ເປີດກ້ອງຄັ້ງດຽວ ແລ້ວໃຊ້ຕໍ່)
@st.cache_resource(show_spinner="ກຳລັງເປີດ WebCam...")
def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
    if not cap.isOpened():  # ລອງ backend ເລີ່ມຕົ້ນ ຖ້າ DirectShow ເປີດບໍ່ໄດ້
        cap = cv2.VideoCapture(index)
    return cap


def close_camera(index: int) -> None:
    """btnStop: ປ່ອຍກ້ອງ (cap.release) ໃຫ້ໂປຣແກຣມອື່ນໃຊ້ໄດ້."""
    open_camera(index).release()
    open_camera.clear()
# <<< CODE


# >>> CODE: Application_Idle / Idle1 / Idle2 + ວຽກບ້ານ: ປະມວນຜົນແຕ່ລະ frame
def process_frame(frame, mode, method=None, thresh=127, max_value=255, block_size=11, c=2):
    """Color = ພາບສີ, Gray = ພາບເທົາ, Binary = ThresholdBinary(150, 255), Threshold = ວຽກບ້ານ."""
    if mode == "Color":
        return frame                                             # frame.ToImage<Bgr, byte>()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)               # frame.ToImage<Gray, byte>()
    if mode == "Gray":
        return gray
    if mode == "Binary":
        return cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]  # ThresholdBinary(Gray(150), Gray(255))

    # --- ວຽກບ້ານ: Simple Thresholding ---
    if method in SIMPLE:
        return cv2.threshold(gray, thresh, max_value, SIMPLE[method])[1]
    if method == "Otsu":
        return cv2.threshold(gray, 0, max_value, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # --- ວຽກບ້ານ: Adaptive Thresholding ---
    return cv2.adaptiveThreshold(gray, max_value, ADAPTIVE[method], cv2.THRESH_BINARY, block_size, c)
# <<< CODE


ss = st.session_state
ss.setdefault("turn_on", False)      # bool turn_on
ss.setdefault("vdo_mode", "Color")   # Color / Gray / Binary / Threshold
ss.setdefault("last_frame", None)    # ຮູບທີ່ສະແດງຢູ່ໃນ imgBoxVDO ຕອນນີ້
ss.setdefault("captured", None)      # imgCapture.Image


def need_camera(mode: str):
    """btnGray / btnBinary: ຖ້າຍັງບໍ່ Start -> MessageBox("Please Start the WebCam")."""
    if ss.turn_on:
        ss.vdo_mode = mode
    else:
        st.toast("Please Start the WebCam", icon="⚠️")


def on_method_change():
    if ss.ex7_method is not None:
        need_camera("Threshold")


# ---------- GroupBox1: VDO From WebCam ----------
with st.container(border=True):
    st.markdown("**VDO From WebCam**")
    b = st.columns([1, 1, 1, 1, 1, 1.4, 1.2])
    cam_index = b[6].number_input("ກ້ອງ (index)", min_value=0, max_value=9, value=0, step=1,
                                  disabled=ss.turn_on, label_visibility="collapsed",
                                  help="VideoCapture(index): 0 = ກ້ອງຫຼັກ, 1 = ກ້ອງທີ 2 ...")

    # >>> CODE: btnStart / btnStop / btnGray / btnBinary / btnCapture
    if b[0].button("Start", type="primary", width="stretch"):
        ss.turn_on = True
        ss.vdo_mode = "Color"
        ss.captured = None                                       # imgCapture.Image = null
    if b[1].button("Stop", width="stretch") and ss.turn_on:
        ss.turn_on = False                                       # Application.Idle -= Application_Idle
        close_camera(cam_index)
    if b[2].button("Gray", width="stretch"):
        need_camera("Gray")
    if b[3].button("Binary", width="stretch"):
        need_camera("Binary")
    if b[4].button("Capture", width="stretch") and ss.last_frame is not None:
        ss.captured = ss.last_frame.copy()                       # imgCapture.Image = imgBoxVDO.Image
    # <<< CODE

    # >>> CODE: btnSave_Click: ບັນທຶກຮູບ imgCapture ເປັນ .jpg
    b[5].download_button(
        "Save", data=cv2.imencode(".jpg", ss.captured)[1].tobytes() if ss.captured is not None else b"",
        file_name="capture.jpg", mime="image/jpeg", disabled=ss.captured is None, width="stretch",
        on_click=lambda: st.toast("Saving Complete"))
    # <<< CODE

# ---------- GroupBox2: ວຽກບ້ານ — Thresholding (ComboBox) ----------
with st.container(border=True):
    st.markdown("**ວຽກບ້ານ: VDO ພາບຂາວດຳ (Simple / Adaptive Thresholding)**")
    t = st.columns([1.6, 1.4, 1, 1, 1])
    # >>> CODE: ວຽກບ້ານ: ComboBox ເລືອກຄຳສັ່ງ Thresholding + ຄ່າ parameter
    method = t[0].selectbox("ComboBox: ຄຳສັ່ງ Thresholding", METHODS, index=None, key="ex7_method",
                            placeholder="ເລືອກຮູບແບບພາບຂາວດຳ", on_change=on_method_change)
    is_adaptive = method in ADAPTIVE
    thresh = t[1].slider("Threshold", 0, 255, 127, disabled=method is None or is_adaptive or method == "Otsu")
    max_value = t[2].number_input("maxValue", 0, 255, 255)
    block_size = t[3].number_input("Blocksize (ຄີກ ≥ 3)", 3, 255, 11, step=2, disabled=not is_adaptive)
    c_value = t[4].number_input("C (Param1)", -50, 50, 2, disabled=not is_adaptive)
    if block_size % 2 == 0:
        block_size += 1
        st.warning(f"Blocksize ຕ້ອງເປັນເລກຄີກ — ໃຊ້ {block_size}.")
    # <<< CODE

params = dict(method=method, thresh=thresh, max_value=int(max_value), block_size=int(block_size), c=int(c_value))
if ss.vdo_mode == "Threshold" and method is None:
    ss.vdo_mode = "Color"

# ---------- VDO Preview: imgBoxVDO | ImgCapture ----------
v1, v2 = st.columns(2)
v1.markdown(f"**imgBoxVDO** — ໂໝດ: `{ss.vdo_mode if ss.vdo_mode != 'Threshold' else method}`"
            + ("" if ss.turn_on else "  (ຢຸດ)"))
vdo_box = v1.empty()
v2.markdown("**ImgCapture**")
if ss.captured is not None:
    v2.image(ss.captured, channels="BGR" if ss.captured.ndim == 3 else "RGB", width="stretch")
else:
    v2.info("ກົດ **Capture** ເພື່ອຖ່າຍຮູບຈາກ VDO.")

if not ss.turn_on:
    if ss.last_frame is not None:  # Stop ແລ້ວ -> ຄ້າງ frame ສຸດທ້າຍໄວ້ (ຄື C#)
        vdo_box.image(ss.last_frame, channels="BGR" if ss.last_frame.ndim == 3 else "RGB", width="stretch")
    else:
        vdo_box.info("ກົດ **Start** ເພື່ອເປີດ WebCam.")
    st.stop()

# >>> CODE: Application_Idle: ອ່ານ frame ຈາກກ້ອງຕໍ່ເນື່ອງ ແລ້ວສະແດງ (loop ຢຸດເອງເມື່ອກົດປຸ່ມອື່ນ)
cap = open_camera(cam_index)
if not cap.isOpened():
    ss.turn_on = False
    open_camera.clear()
    vdo_box.error(f"ບໍ່ສາມາດເປີດ WebCam (index {cam_index}) — ກວດວ່າມີກ້ອງ ແລະ ບໍ່ມີໂປຣແກຣມອື່ນໃຊ້ຢູ່.")
    st.stop()

while ss.turn_on:
    ok, frame = cap.read()                                       # capture.Retrieve(frame)
    if not ok:
        vdo_box.error("ອ່ານ frame ຈາກ WebCam ບໍ່ໄດ້.")
        break
    out = process_frame(frame, ss.vdo_mode, **params)
    ss.last_frame = out
    vdo_box.image(out, channels="BGR" if out.ndim == 3 else "RGB", width="stretch", output_format="JPEG")
# <<< CODE
