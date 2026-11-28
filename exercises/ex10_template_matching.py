"""
ບົດຝຶກຫັດ 10: ການຈັບຄູ່ແຜ່ນແມ່ແບບ (Template Matching)
C# (EmguCV)                                               ->  Python (OpenCV + Streamlit)
-------------------------------------------------------------------------------------------
MenuStrip: File (Open, Save, Exit), Matching              ->  st.popover("File") + ປຸ່ມ "Matching"
File > Open: tpImage = new Image<Bgr, byte>(file)         ->  st.file_uploader (Template)
imageBox2_Click: sImage = new Image<Bgr, byte>(file)      ->  st.file_uploader ຢູ່ເທິງ imgSource
CvInvoke.MatchTemplate(sImage, tpImage, imgOutput, TemplateMatchingType.Sqdiff)
                                                          ->  cv2.matchTemplate(src, tpl, cv2.TM_SQDIFF)
CvInvoke.MinMaxLoc(imgOutput, ref minVal, ref maxVal, ref minLoc, ref maxLoc)
                                                          ->  cv2.minMaxLoc(result)
Rectangle r = new Rectangle(minLoc, tpImage.Size)         ->  ມຸມຊ້າຍເທິງ = minLoc, ຂະໜາດ = ຂະໜາດ template
CvInvoke.Rectangle(sImage, r, new MCvScalar(0, 255, 0), 2) ->  cv2.rectangle(..., (0, 255, 0), 2)

ທາງເລືອກເພີ່ມ: imgSource ເປັນ VDO (file VDO ຫຼື WebCam) ແທນຮູບພາບ
capture = new VideoCapture(file) / VideoCapture(0)        ->  cv2.VideoCapture(path) / cv2.VideoCapture(0)
Application.Idle: capture.Retrieve(frame) + Matching      ->  while loop: cap.read() -> match_template ທຸກ frame
"""
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from code_viewer import show_code_button  # noqa: E402
from ui import page_header  # noqa: E402

st.set_page_config(page_title="Ex10 - Template Matching", layout="wide")
page_header(10)
show_code_button(__file__)

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "wmv", "webm", "m4v"]
# ແຫຼ່ງຂອງ imgSource: ຮູບພາບ (ຄື code C#) ຫຼື VDO
SRC_IMAGE, SRC_FILE, SRC_CAM = "ຮູບພາບ", "VDO (file)", "VDO (WebCam)"
# TemplateMatchingType: Sqdiff ຄື code C#. ແບບ Sqdiff = ຄ່ານ້ອຍສຸດຄືກົງກັນທີ່ສຸດ, ແບບອື່ນ = ຄ່າໃຫຍ່ສຸດ
METHODS = {
    "Sqdiff (ຕາມ code C#)": cv2.TM_SQDIFF,
    "SqdiffNormed": cv2.TM_SQDIFF_NORMED,
    "Ccorr": cv2.TM_CCORR,
    "CcorrNormed": cv2.TM_CCORR_NORMED,
    "Ccoeff": cv2.TM_CCOEFF,
    "CcoeffNormed": cv2.TM_CCOEFF_NORMED,
}
SQDIFF_METHODS = {cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED}


def read_bgr(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# >>> CODE: matchingToolStripMenuItem_Click: MatchTemplate + MinMaxLoc + ແຕ້ມຮູບສີ່ຫຼ່ຽມ
def match_template(s_image: np.ndarray, tp_image: np.ndarray, method: int):
    """ສົ່ງຄືນ (ຮູບທີ່ແຕ້ມກອບແລ້ວ, imgOutput ຂອງ MatchTemplate, ຕຳແໜ່ງທີ່ດີທີ່ສຸດ, ຄ່າຄະແນນ)."""
    img_output = cv2.matchTemplate(s_image, tp_image, method)          # CvInvoke.MatchTemplate(...)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img_output)     # CvInvoke.MinMaxLoc(...)
    if method in SQDIFF_METHODS:
        loc, score = min_loc, min_val    # Sqdiff: ຄວາມຕ່າງກຳລັງສອງ -> ນ້ອຍສຸດ = ກົງກັນທີ່ສຸດ (minLoc)
    else:
        loc, score = max_loc, max_val    # Ccorr / Ccoeff: ຄວາມຄ້າຍຄືກັນ -> ໃຫຍ່ສຸດ = ກົງກັນທີ່ສຸດ

    h, w = tp_image.shape[:2]                                          # tpImage.Size
    result = s_image.copy()  # C# ແຕ້ມໃສ່ sImage ໂດຍກົງ; ໃນນີ້ແຕ້ມໃສ່ສຳເນົາ ເພື່ອບໍ່ໃຫ້ກອບຊ້ອນກັນ
    cv2.rectangle(result, loc, (loc[0] + w - 1, loc[1] + h - 1), (0, 255, 0), 2)  # MCvScalar(0, 255, 0), 2
    return result, img_output, loc, score
# <<< CODE


def score_map(img_output: np.ndarray, method: int) -> np.ndarray:
    """ແປງ imgOutput (float) ເປັນຮູບ 0–255: ຈຸດສະຫວ່າງ = ກົງກັນຫຼາຍ (Sqdiff ກັບຄ່າກ່ອນ)."""
    m = cv2.normalize(img_output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return 255 - m if method in SQDIFF_METHODS else m


ss = st.session_state
ss.setdefault("ex10_tp_id", None)     # file_id ຂອງ template
ss.setdefault("ex10_src", None)       # (ແຫຼ່ງ, file_id / index ກ້ອງ) ຂອງ imgSource
ss.setdefault("ex10_matched", False)
ss.setdefault("ex10_play", False)     # VDO ກຳລັງຫຼິ້ນຢູ່ບໍ່
ss.setdefault("ex10_frame", None)     # frame ດິບລ່າສຸດຂອງ VDO (ຄ້າງໄວ້ເມື່ອກົດ Stop)
ss.setdefault("ex10_cap", (None, None))  # (source, cv2.VideoCapture) ທີ່ເປີດຢູ່
ss.setdefault("ex10_tmp", None)       # file ຊົ່ວຄາວຂອງ VDO ທີ່ upload


# >>> CODE: imgSource ເປັນ VDO: ເປີດ VideoCapture (file VDO ຫຼື WebCam)
def video_path(uploaded) -> str:
    """cv2.VideoCapture ອ່ານໄດ້ຈາກ path ເທົ່ານັ້ນ -> ຂຽນ file ທີ່ upload ລົງ file ຊົ່ວຄາວກ່ອນ."""
    path = Path(tempfile.gettempdir()) / f"cvlab_ex10_{uploaded.file_id}{Path(uploaded.name).suffix.lower()}"
    if not path.exists():
        path.write_bytes(uploaded.getvalue())
    return str(path)


def open_capture(source: str | int) -> cv2.VideoCapture:
    """source = path ຂອງ file VDO  ຫຼື  index ຂອງ WebCam (0 = ກ້ອງຫຼັກ)."""
    if isinstance(source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)            # new VideoCapture(0)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(source)                              # new VideoCapture(file)
# <<< CODE


def get_capture(source: str | int) -> cv2.VideoCapture:
    """ເປີດ VideoCapture ຄັ້ງດຽວ ແລ້ວໃຊ້ຕໍ່ (file VDO ຈຶ່ງຫຼິ້ນຕໍ່ຈາກ frame ເກົ່າເມື່ອກົດ Start ອີກ)."""
    opened, cap = ss.ex10_cap
    if opened == source and cap is not None and cap.isOpened():
        return cap
    release_capture()
    cap = open_capture(source)
    ss.ex10_cap = (source, cap)
    return cap


def release_capture() -> None:
    """ປ່ອຍ VDO / WebCam (cap.release) ໃຫ້ໂປຣແກຣມອື່ນໃຊ້ໄດ້."""
    cap = ss.ex10_cap[1]
    if cap is not None:
        cap.release()
    ss.ex10_cap = (None, None)


def check_source(key: tuple, tmp_path: str | None = None) -> None:
    """ປ່ຽນແຫຼ່ງ imgSource -> ປິດ VDO ເກົ່າ ແລະ ລ້າງຜົນເກົ່າ (ຄື imgSource.Image = null)."""
    if ss.ex10_src == key:
        return
    release_capture()
    if ss.ex10_tmp and ss.ex10_tmp != tmp_path:
        try:
            Path(ss.ex10_tmp).unlink(missing_ok=True)
        except OSError:
            pass
    ss.update(ex10_src=key, ex10_tmp=tmp_path, ex10_play=False, ex10_frame=None, ex10_matched=False)


def vdo_start():
    ss.ex10_play = True


def vdo_stop():
    """btnStop: ຢຸດ loop; WebCam ຕ້ອງປ່ອຍກ້ອງ, file VDO ເປີດຄ້າງໄວ້ເພື່ອຫຼິ້ນຕໍ່."""
    ss.ex10_play = False
    if ss.get("ex10_src_mode") == SRC_CAM:
        release_capture()


def on_matching():
    """ຮູບພາບ: ກົດ Matching = ຊອກຫາ 1 ຄັ້ງ.  VDO: ກົດເພື່ອ ເປີດ/ປິດ Matching ໃນທຸກ frame."""
    ss.ex10_matched = (not ss.ex10_matched) if ss.get("ex10_src_mode", SRC_IMAGE) != SRC_IMAGE else True


# ---------- MenuStrip: File | Matching | parameter ----------
m1, m2, m3, _ = st.columns([1, 1.2, 2.2, 3.6])
with m1.popover("File", width="stretch"):
    # >>> CODE: File > Open: ເລືອກຮູບ Template  (openToolStripMenuItem_Click)
    tp_file = st.file_uploader("Open (Template)", type=IMAGE_TYPES)
    # <<< CODE
    save_slot = st.container()
    st.caption("**Exit**: ໃນ web app ບໍ່ມີ this.Close() — ປິດ tab browser ຫຼື ກົດ Ctrl+C ໃນ terminal.")

with m3.popover("⚙️ ປັບຄ່າ parameter", width="stretch"):
    method_name = st.radio("TemplateMatchingType", list(METHODS), index=0)
    method = METHODS[method_name]

# ---------- imgTemplate | imgSource ----------
left, right = st.columns([1, 2.6])
with left:
    st.markdown("**imgTemplate**")
    tp_image = read_bgr(tp_file) if tp_file is not None else None
    if tp_image is not None:
        st.image(tp_image, channels="BGR", width="content", caption="Template")
    else:
        st.info("ເປີດເມນູ **File → Open** ເພື່ອເລືອກຮູບ Template.")

# ປ່ຽນ Template -> ລ້າງຜົນເກົ່າ
tp_id = tp_file.file_id if tp_file else None
if ss.ex10_tp_id != tp_id:
    ss.ex10_tp_id, ss.ex10_matched = tp_id, False

with right:
    st.markdown("**imgSource**")
    src_mode = st.radio("ແຫຼ່ງຂອງ imgSource", [SRC_IMAGE, SRC_FILE, SRC_CAM], horizontal=True,
                        key="ex10_src_mode", label_visibility="collapsed")
    is_vdo = src_mode != SRC_IMAGE
    source, loop = None, False
    if not is_vdo:
        # >>> CODE: imageBox2_Click: ເລືອກຮູບ Source
        s_file = st.file_uploader("ເລືອກຮູບ Source (ຄື imageBox2_Click)", type=IMAGE_TYPES,
                                  label_visibility="collapsed")
        # <<< CODE
        s_image = read_bgr(s_file) if s_file is not None else None
        check_source((src_mode, s_file.file_id if s_file else None))
    else:
        if src_mode == SRC_FILE:
            v_file = st.file_uploader("ເລືອກ file VDO", type=VIDEO_TYPES, label_visibility="collapsed")
            source = video_path(v_file) if v_file is not None else None
            check_source((src_mode, v_file.file_id if v_file else None), source)
            vc = st.columns([1, 1, 2.6])
            loop = vc[2].checkbox("ວົນຫຼິ້ນຄືນ (Loop)", value=True, key="ex10_loop")
        else:
            vc = st.columns([1, 1, 2.6])
            source = int(vc[2].number_input("ກ້ອງ (index)", min_value=0, max_value=9, value=0, step=1,
                                            disabled=ss.ex10_play, label_visibility="collapsed",
                                            help="VideoCapture(index): 0 = ກ້ອງຫຼັກ, 1 = ກ້ອງທີ 2 ..."))
            check_source((src_mode, source))
        vc[0].button("Start", type="primary", width="stretch", on_click=vdo_start,
                     disabled=source is None or ss.ex10_play)
        vc[1].button("Stop", width="stretch", on_click=vdo_stop, disabled=not ss.ex10_play)
        st.caption("ເຄັດລັບ: ກົດ **Stop** → **File → Save** ເພື່ອບັນທຶກ frame, ແລ້ວຕັດ (crop) ວັດຖຸອອກມາເປັນ "
                   "Template ໂດຍ**ບໍ່ຍໍ້/ຂະຫຍາຍ** — Template ຕ້ອງມີຂະໜາດເທົ່າກັບວັດຖຸໃນ VDO.")
        s_image = ss.ex10_frame  # Stop ແລ້ວ -> ໃຊ້ frame ທີ່ຄ້າງໄວ້ເປັນຮູບ Source
    source_box = st.container()

playing = is_vdo and ss.ex10_play
has_source = source is not None if is_vdo else s_image is not None
m2.button("✓ Matching" if is_vdo and ss.ex10_matched else "Matching", type="primary", width="stretch",
          disabled=tp_image is None or not has_source, on_click=on_matching,
          help="VDO: ກົດເພື່ອ ເປີດ/ປິດ Matching ໃນທຸກ frame" if is_vdo else None)

if msg := ss.pop("ex10_msg", None):  # ຂໍ້ຄວາມຈາກ loop VDO ຮອບກ່ອນ (ເຊັ່ນ VDO ຈົບແລ້ວ)
    getattr(source_box, msg[0])(msg[1])

shown = None if playing else s_image
if playing:
    pass  # VDO ກຳລັງຫຼິ້ນ -> loop ຢູ່ລຸ່ມສຸດເປັນຜູ້ສະແດງ frame
elif tp_image is not None and s_image is not None and ss.ex10_matched:
    th, tw = tp_image.shape[:2]
    sh, sw = s_image.shape[:2]
    if th > sh or tw > sw:
        source_box.error(f"Template ({tw}×{th}) ໃຫຍ່ກວ່າຮູບ Source ({sw}×{sh}) — ຕ້ອງເລືອກ Template ທີ່ນ້ອຍກວ່າ.")
    else:
        shown, img_output, loc, score = match_template(s_image, tp_image, method)
        if img_output.size <= 4:  # Template ໃຫຍ່ເທົ່າ (ຫຼືເກືອບເທົ່າ) Source -> ມີຕຳແໜ່ງໃຫ້ລອງພຽງ 1–4 ຈຸດ
            source_box.warning(
                f"Template ({tw}×{th}) ມີຂະໜາດເທົ່າກັບ (ຫຼືເກືອບເທົ່າ) ຮູບ Source ({sw}×{sh}) "
                f"→ ວາງ Template ໄດ້ພຽງ {img_output.shape[1]}×{img_output.shape[0]} ຕຳແໜ່ງ, ຜົນຈຶ່ງບໍ່ມີຄວາມໝາຍ. "
                "Template Matching ບໍ່ຮອງຮັບການປ່ຽນຂະໜາດ: ໃຫ້ຕັດ (crop) ສ່ວນນ້ອຍໆຈາກຮູບ Source "
                "ໂດຍ**ບໍ່ຍໍ້/ຂະຫຍາຍ**ຮູບ.")
        source_box.image(shown, channels="BGR", width="stretch",
                         caption=f"{method_name}: ກົງກັນທີ່ສຸດຢູ່ (x={loc[0]}, y={loc[1]}), "
                                 f"ຂະໜາດ {tw}×{th}, ຄ່າ = {score:.4g}")
        with right.expander("ເບິ່ງ imgOutput ຂອງ MatchTemplate (ແຜນທີ່ຄະແນນ)"):
            st.caption(f"ຂະໜາດ = (W − w + 1) × (H − h + 1) = {img_output.shape[1]}×{img_output.shape[0]}. "
                       "ແຕ່ລະຈຸດ = ຄະແນນເມື່ອວາງມຸມຊ້າຍເທິງຂອງ Template ຢູ່ຈຸດນັ້ນ. "
                       "ຈຸດທີ່ສະຫວ່າງທີ່ສຸດ = ຕຳແໜ່ງທີ່ກົງກັນທີ່ສຸດ.")
            if img_output.size <= 4:
                st.info("ແຜນທີ່ມີພຽງ 1–4 ຈຸດ (ທຸກຈຸດຄ່າເທົ່າກັນ ຫຼື ເກືອບເທົ່າກັນ) ຈຶ່ງສະແດງເປັນຮູບບໍ່ໄດ້ "
                        "— ຖ້າສະແດງຈະເປັນສີດຳລ້ວນ.")
            else:
                st.image(score_map(img_output, method), width="stretch")
elif s_image is not None:
    source_box.image(s_image, channels="BGR", width="stretch")
    if tp_image is not None:
        source_box.caption("ກົດ **Matching** ເພື່ອຊອກຫາ Template ໃນຮູບນີ້.")
elif src_mode == SRC_FILE:
    source_box.info("ເລືອກ file VDO ແລ້ວກົດ **Start** ເພື່ອຫຼິ້ນ VDO.")
elif src_mode == SRC_CAM:
    source_box.info("ກົດ **Start** ເພື່ອເປີດ WebCam.")
else:
    source_box.info("ເລືອກຮູບ Source ທີ່ຈະຊອກຫາ Template.")

# ---------- File > Save ----------
with save_slot:
    # >>> CODE: File > Save: ບັນທຶກຮູບ imgSource (ທີ່ແຕ້ມກອບແລ້ວ)
    fmt = st.selectbox("ຮູບແບບ file", ["jpg", "png", "bmp"])
    st.download_button("Save", data=cv2.imencode("." + fmt, shown)[1].tobytes() if shown is not None else b"",
                       file_name=f"template_matching.{fmt}", mime=f"image/{'jpeg' if fmt == 'jpg' else fmt}",
                       disabled=shown is None, width="stretch", on_click=lambda: st.toast("Saving Complete"))
    # <<< CODE
    if playing:
        st.caption("VDO ກຳລັງຫຼິ້ນ: ກົດ **Stop** ກ່ອນ ເພື່ອ Save frame ປັດຈຸບັນ.")

if not playing:
    st.stop()

# >>> CODE: imgSource ເປັນ VDO: ອ່ານ frame ຕໍ່ເນື່ອງ ແລ້ວ Matching ທຸກ frame (loop ຢຸດເອງເມື່ອກົດປຸ່ມອື່ນ)
cap = get_capture(source)
if not cap.isOpened():
    ss.ex10_play = False
    release_capture()
    ss.ex10_msg = ("error", "ເປີດ file VDO ບໍ່ໄດ້ — ລອງ file .mp4 ຫຼື .avi." if src_mode == SRC_FILE else
                   f"ບໍ່ສາມາດເປີດ WebCam (index {source}) — ກວດວ່າມີກ້ອງ ແລະ ບໍ່ມີໂປຣແກຣມອື່ນໃຊ້ຢູ່.")
    st.rerun()

is_file = src_mode == SRC_FILE
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1.0 / fps if is_file and 1 <= fps <= 240 else 0.0  # file VDO: ຫຼິ້ນບໍ່ໃຫ້ໄວກວ່າ FPS ຂອງ file
do_match = ss.ex10_matched and tp_image is not None
th, tw = tp_image.shape[:2] if tp_image is not None else (0, 0)
vdo_box = source_box.empty()
n_read, rate, t_last = 0, 0.0, time.perf_counter()

while ss.ex10_play:
    t_start = time.perf_counter()
    ok, frame = cap.read()                                       # capture.Retrieve(frame)
    if not ok:
        if is_file:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)                  # ກັບໄປ frame ທຳອິດ
            if loop and n_read > 0:
                n_read = 0
                continue
            ss.ex10_msg = ("info", "VDO ຈົບແລ້ວ — ກົດ **Start** ເພື່ອຫຼິ້ນອີກຄັ້ງ.")
        else:
            release_capture()
            ss.ex10_msg = ("error", "ອ່ານ frame ຈາກ WebCam ບໍ່ໄດ້.")
        ss.ex10_play = False
        st.rerun()
    n_read += 1
    ss.ex10_frame = frame                                        # ເກັບ frame ໄວ້ ເມື່ອກົດ Stop

    fh, fw = frame.shape[:2]
    if not do_match:
        out, caption = frame, ("ກົດ Matching ເພື່ອຊອກຫາ Template ໃນທຸກ frame." if tp_image is not None
                               else "ເປີດເມນູ File → Open ເພື່ອເລືອກຮູບ Template.")
    elif th > fh or tw > fw:
        out, caption = frame, f"Template ({tw}×{th}) ໃຫຍ່ກວ່າ frame VDO ({fw}×{fh}) — ຕ້ອງເລືອກ Template ທີ່ນ້ອຍກວ່າ."
    else:
        out, _, loc, score = match_template(frame, tp_image, method)   # MatchTemplate + MinMaxLoc + Rectangle
        caption = f"{method_name}: ກົງກັນທີ່ສຸດຢູ່ (x={loc[0]}, y={loc[1]}), ຄ່າ = {score:.4g}"

    now = time.perf_counter()
    rate = 1 / max(now - t_last, 1e-6) if rate == 0 else 0.9 * rate + 0.1 / max(now - t_last, 1e-6)
    t_last = now
    vdo_box.image(out, channels="BGR", width="stretch", output_format="JPEG",
                  caption=f"{caption}  ·  {rate:.1f} fps")
    wait = frame_time - (time.perf_counter() - t_start)
    if wait > 0:
        time.sleep(wait)
# <<< CODE
