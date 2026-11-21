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
"""
import sys
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
ss.setdefault("ex10_ids", (None, None))  # (file_id ຂອງ template, file_id ຂອງ source)
ss.setdefault("ex10_matched", False)

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

with right:
    st.markdown("**imgSource**")
    # >>> CODE: imageBox2_Click: ເລືອກຮູບ Source
    s_file = st.file_uploader("ເລືອກຮູບ Source (ຄື imageBox2_Click)", type=IMAGE_TYPES,
                              label_visibility="collapsed")
    # <<< CODE
    s_image = read_bgr(s_file) if s_file is not None else None
    source_box = st.container()

# ປ່ຽນຮູບໃດຮູບໜຶ່ງ -> ລ້າງຜົນເກົ່າ (ຄື imgSource.Image = null)
ids = (tp_file.file_id if tp_file else None, s_file.file_id if s_file else None)
if ss.ex10_ids != ids:
    ss.ex10_ids, ss.ex10_matched = ids, False

ready = tp_image is not None and s_image is not None
if m2.button("Matching", type="primary", width="stretch", disabled=not ready):
    ss.ex10_matched = True

shown = s_image
if ready and ss.ex10_matched:
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
