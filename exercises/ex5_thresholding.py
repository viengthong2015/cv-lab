"""
ບົດຝຶກຫັດ 5: ຮັບຄ່າຂໍ້ມູນພາບຂາວດຳຈາກຮູບພາບສີ (Get Binary Image from Input Image)
C# (EmguCV)                                          ->  Python (OpenCV)
--------------------------------------------------------------------------------------
grayImage.ThresholdBinary(Gray(100), Gray(255))      ->  cv2.threshold(g, 100, 255, THRESH_BINARY)
grayImage.ThresholdBinaryInv(Gray(100), Gray(255))   ->  cv2.threshold(g, 100, 255, THRESH_BINARY_INV)
OT'Su                                                ->  cv2.threshold(g, 0, 255, THRESH_BINARY | THRESH_OTSU)
grayImage.ThresholdToZero(Gray(100))                 ->  cv2.threshold(g, 100, 255, THRESH_TOZERO)
ThresholdAdaptive(Gray(256), GaussianC, Binary, 9, Gray(0))
                                                     ->  cv2.adaptiveThreshold(g, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                               THRESH_BINARY, 9, 0)
ThresholdAdaptive(Gray(256), MeanC, Binary, 9, Gray(0))
                                                     ->  cv2.adaptiveThreshold(g, 255, ADAPTIVE_THRESH_MEAN_C, ...)

ວຽກບ້ານ 2:
  1. Fixed Thresholding: ປ່ຽນ maxValue ຜ່ານ TextBox
  2. Adaptive Thresholding: ປ່ຽນ blocksize ຜ່ານ TextBox, param1 (C) ຜ່ານ NumericUpDown
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from code_viewer import show_code_button  # noqa: E402

st.set_page_config(page_title="Ex5 - Thresholding", layout="wide")
st.title("ບົດຝຶກຫັດ 5: ຮັບຄ່າຂໍ້ມູນພາບຂາວດຳຈາກຮູບພາບສີ")
st.caption("Get Binary Image from Input Image")
show_code_button(__file__)

# ComboBox1 Items (Collection)
METHODS = ["Binary", "Binary Invert", "OT'Su", "Guassian", "Mean", "To Zero (ຕາມ code C# index 2)"]
FIXED = {"Binary", "Binary Invert", "OT'Su", "To Zero (ຕາມ code C# index 2)"}
ADAPTIVE = {"Guassian", "Mean"}


def read_bgr(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def parse_int(text: str, name: str, lo: int, hi: int, default: int) -> int:
    """ອ່ານຄ່າຈາກ TextBox ແລະ ກວດວ່າຖືກຕ້ອງ."""
    try:
        v = int(text)
    except ValueError:
        st.warning(f"{name} ຕ້ອງເປັນຕົວເລກ — ໃຊ້ຄ່າ {default} ແທນ.")
        return default
    if not lo <= v <= hi:
        st.warning(f"{name} ຕ້ອງຢູ່ລະຫວ່າງ {lo}–{hi} — ໃຊ້ຄ່າ {default} ແທນ.")
        return default
    return v


# >>> CODE: comboBox1_SelectedIndexChanged: Thresholding ແຕ່ລະແບບ (Fixed / Otsu / Adaptive)
def apply_threshold(gray, method, thresh, max_value, block_size, c):
    """ຄືກັບ comboBox1_SelectedIndexChanged ໃນ C#. ສົ່ງຄືນ (binary_image, ຄ່າ threshold ທີ່ໃຊ້)."""
    if method == "Binary":
        t, out = cv2.threshold(gray, thresh, max_value, cv2.THRESH_BINARY)
    elif method == "Binary Invert":
        t, out = cv2.threshold(gray, thresh, max_value, cv2.THRESH_BINARY_INV)
    elif method == "OT'Su":
        # Otsu ຊອກຫາຄ່າ threshold ທີ່ເໝາະສົມເອງ (ບໍ່ໃຊ້ thresh ທີ່ປ້ອນ)
        t, out = cv2.threshold(gray, 0, max_value, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    elif method.startswith("To Zero"):
        t, out = cv2.threshold(gray, thresh, max_value, cv2.THRESH_TOZERO)
    elif method == "Guassian":
        out = cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, block_size, c)
        t = None
    else:  # Mean
        out = cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, block_size, c)
        t = None
    return out, t
# <<< CODE


top_left, top_right = st.columns(2)
bottom_left, bottom_right = st.columns(2)

# ---------- ແຜງຄວບຄຸມ (ລຸ່ມຂວາ) ----------
with bottom_right:
    # btnOpen
    uploaded = st.file_uploader("Open", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])

    # Open ຮູບໃໝ່ -> imgBinary = null, ComboBox ກັບໄປ "ເລືອກຮູບແບບພາບຂາວດຳ"
    # (key ຂອງ ComboBox ປ່ຽນຕາມ file -> ຖືກ reset ອັດຕະໂນມັດ)
    file_key = uploaded.file_id if uploaded is not None else "none"

    with st.container(border=True):
        st.markdown("**Thresholding**")
        method = st.selectbox("ComboBox1", METHODS, index=None, key=f"method_{file_key}",
                              placeholder="ເລືອກຮູບແບບພາບຂາວດຳ", label_visibility="collapsed")

        # >>> CODE: ວຽກບ້ານ 2: TextBox (maxValue, Blocksize) + NumericUpDown (Param1)
        # --- ວຽກບ້ານ 2 ຂໍ້ 1: Gray MaxValue (TextBox) ---
        r1c1, r1c2 = st.columns(2)
        max_text = r1c1.text_input("Gray MaxValue:", value="255")
        thresh = r1c2.slider("Threshold (ຄ່າຕັດ)", 0, 255, 100,
                             disabled=method not in FIXED or method == "OT'Su",
                             help="C# ໃຊ້ຄ່າຄົງທີ່ 100. Otsu ຄຳນວນຄ່ານີ້ເອງ.")

        # --- ວຽກບ້ານ 2 ຂໍ້ 2: Blocksize (TextBox) + Param1 (NumericUpDown) ---
        r2c1, r2c2 = st.columns(2)
        block_text = r2c1.text_input("Blocksize:", value="9", disabled=method not in ADAPTIVE,
                                     help="ຕ້ອງເປັນເລກຄີກ ແລະ ≥ 3 (ເຊັ່ນ 3, 5, 9, 11, 15, 21...)")
        c_value = r2c2.number_input("Param1:", min_value=-100, max_value=100, value=0, step=1,
                                    disabled=method not in ADAPTIVE,
                                    help="ຄ່າຄົງທີ່ C ທີ່ຖືກລົບອອກຈາກຄ່າສະເລ່ຍຂອງ block")

    max_value = parse_int(max_text, "Gray MaxValue", 0, 255, 255)
    block_size = 9
    if method in ADAPTIVE:
        block_size = parse_int(block_text, "Blocksize", 3, 999, 9)
        if block_size % 2 == 0:
            block_size += 1
            st.warning(f"Blocksize ຕ້ອງເປັນເລກຄີກ — ປ່ຽນເປັນ {block_size}.")
    # <<< CODE

# ---------- imgInput / imgGray / imgBinary ----------
if uploaded is None:
    with top_left:
        st.info("ກົດ **Open** (ດ້ານລຸ່ມຂວາ) ເພື່ອເລືອກຮູບພາບ.")
    st.stop()

# >>> CODE: btnOpen_Click: ອ່ານຮູບ ແລະ ປ່ຽນເປັນຮູບເທົາ  (C#: oriImage.Convert<Gray, byte>())
ori_image = read_bgr(uploaded)
if ori_image is None:
    with top_left:
        st.error("ບໍ່ສາມາດອ່ານ file ນີ້ເປັນຮູບພາບໄດ້.")
    st.stop()
gray_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)  # oriImage.Convert<Gray, byte>()
# <<< CODE

with top_left:
    st.markdown("**imgInput**")
    st.image(ori_image, channels="BGR", width="stretch")
with top_right:
    st.markdown("**imgGray**")
    st.image(gray_image, width="stretch")

binary_image = None
with bottom_left:
    st.markdown("**imgBinary**")
    if method is None:
        st.info("ເລືອກຮູບແບບ Thresholding ຈາກ ComboBox.")
    else:
        binary_image, t = apply_threshold(gray_image, method, thresh, max_value, block_size, int(c_value))
        info = f"{method} | maxValue = {max_value}"
        if method in ADAPTIVE:
            info += f" | blocksize = {block_size} | C = {int(c_value)}"
        elif t is not None:
            info += f" | threshold = {t:.0f}"
        st.image(binary_image, width="stretch", caption=info)

# ---------- btnSave ----------
with bottom_right:
    s1, s2 = st.columns([1, 2])
    # >>> CODE: btnSave: ບັນທຶກຮູບຂາວດຳ (jpg / png / bmp)
    fmt = s1.selectbox("ຮູບແບບ file", ["png", "jpg", "bmp"], label_visibility="collapsed")
    data = cv2.imencode("." + fmt, binary_image)[1].tobytes() if binary_image is not None else b""
    s2.download_button("Save", data=data, file_name=f"binary_image.{fmt}",
                       mime=f"image/{'jpeg' if fmt == 'jpg' else fmt}",
                       disabled=binary_image is None, width="stretch",
                       on_click=lambda: st.toast("Saving Complete"))
    # <<< CODE
    st.caption("ໝາຍເຫດ: ປຸ່ມ **Exit** ຂອງ C# ບໍ່ຈຳເປັນໃນ web app — ປິດ tab browser ຫຼື ກົດ Ctrl+C ໃນ terminal.")

# ---------- ວຽກບ້ານ 2 ຂໍ້ 3 ແລະ 4 ----------
with st.expander("📝 ວຽກບ້ານ 2 — ຂໍ້ 3: ຄ່າທີ່ແນະນຳ  |  ຂໍ້ 4: ການນຳໃຊ້ Fixed ແລະ Adaptive Thresholding"):
    st.markdown("""
**ຂໍ້ 3 — ຄ່າທີ່ແນະນຳ (ຄ່າທີ່ດີທີ່ສຸດຂຶ້ນກັບຮູບ — ໃຫ້ທົດລອງກັບຮູບຂອງຕົນເອງ ແລ້ວ Save ຮູບແນບມາ):**

| ວິທີ | ຄ່າທີ່ຄວນເລີ່ມ | ຜົນຂອງການປ່ຽນຄ່າ |
|---|---|---|
| Binary / Binary Invert | threshold ≈ 100–150, maxValue = 255 | threshold ສູງ → ຂາວໜ້ອຍລົງ; maxValue ຕ່ຳ → ສ່ວນ "ຂາວ" ກາຍເປັນສີເທົາ |
| Otsu | maxValue = 255 | ຄຳນວນ threshold ເອງ — ດີເມື່ອ histogram ມີ 2 ຈອມ (ວັດຖຸ/ພື້ນຫຼັງ) |
| Adaptive Gaussian / Mean | blocksize ≈ 11–31, Param1 (C) ≈ 2–10 | blocksize ນ້ອຍ → ລາຍລະອຽດຫຼາຍ ແຕ່ noise ຫຼາຍ; C ໃຫຍ່ຂຶ້ນ → noise ພື້ນຫຼັງຫຼຸດລົງ |

ຂໍ້ສັງເກດ: ຄ່າ C# ຕົ້ນສະບັບ (blocksize = 9, C = 0) ມັກໃຫ້ຮູບທີ່ມີ noise ຫຼາຍ ເພາະພື້ນທີ່ລຽບ (ສີສະໝ່ຳສະເໝີ)
ຈະມີ pixel ປະມານເຄິ່ງໜຶ່ງສູງກວ່າຄ່າສະເລ່ຍ. ການເພີ່ມ C ເປັນ 5–10 ຈະຊ່ວຍໃຫ້ພື້ນຫຼັງສະອາດຂຶ້ນ.

**ຂໍ້ 4 — ຂອບເຂດການນຳໃຊ້:**

- **Fixed (Simple) Thresholding** — ໃຊ້ຄ່າ threshold ດຽວກັນທັງຮູບ: `pixel > T → maxValue, ນອກນັ້ນ → 0`.
  ເໝາະກັບຮູບທີ່ແສງສະໝ່ຳສະເໝີ ແລະ ວັດຖຸກັບພື້ນຫຼັງມີຄວາມແຕກຕ່າງຊັດເຈນ ເຊັ່ນ: ການກວດຊິ້ນສ່ວນໃນໂຮງງານ
  (ພື້ນຫຼັງຄວບຄຸມແສງໄດ້), ການແຍກວັດຖຸອອກຈາກພື້ນຫຼັງ, ການສ້າງ mask, ການນັບວັດຖຸ/ເຊວ ໃນຮູບທາງການແພດ.
  **Otsu** ເປັນ Fixed thresholding ທີ່ເລືອກ T ອັດຕະໂນມັດຈາກ histogram.
- **Adaptive Thresholding** — ຄຳນວນ threshold ແຍກກັນໃນແຕ່ລະບໍລິເວນ (block) ຈາກຄ່າສະເລ່ຍ (Mean) ຫຼື
  ຄ່າສະເລ່ຍແບບຖ່ວງນ້ຳໜັກ Gaussian ຂອງ pixel ອ້ອມຂ້າງ ລົບດ້ວຍ C. ເໝາະກັບຮູບທີ່ແສງບໍ່ສະໝ່ຳສະເໝີ (ມີເງົາ)
  ເຊັ່ນ: ສະແກນເອກະສານ/OCR, ອ່ານປ້າຍທະບຽນລົດ, ອ່ານ QR/Barcode, ຮູບຖ່າຍເອກະສານດ້ວຍໂທລະສັບ.
""")
