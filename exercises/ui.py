"""
ສ່ວນ UI ທີ່ໃຊ້ຮ່ວມກັນທຸກໜ້າ.

ແນວຄິດການອອກແບບ: ແຜ່ນທຽບສີ ColorChecker (24 patches) ທີ່ໃຊ້ calibrate ສີຂອງກ້ອງໃນວຽກ Computer Vision.
  - ແຖບ patch 24 ສີ ຢູ່ເທິງສຸດຂອງທຸກໜ້າ (ຈຸດເດັ່ນອັນດຽວຂອງການອອກແບບ)
  - ແຕ່ລະບົດຝຶກຫັດມີສີ patch ຂອງຕົນເອງ (ໃຊ້ໃນຫົວໜ້າ ແລະ ໜ້າຫຼັກ)
  - ຮູບພາບວາງເທິງພື້ນສີເທົາກາງ (18% gray) ຄືໂປຣແກຣມແຕ່ງຮູບ ເພື່ອບໍ່ໃຫ້ສີອ້ອມຂ້າງຫຼອກຕາ
"""
from html import escape

import streamlit as st

# ColorChecker Classic: 24 patches (sRGB), ລຳດັບແຖວຕໍ່ແຖວ
COLORCHECKER = [
    "#735244", "#C29682", "#627A9D", "#576C43", "#8580B1", "#67BDAA",
    "#D67E2C", "#505BA6", "#C15A63", "#5E3C6C", "#9DBC40", "#E0A32E",
    "#383D96", "#469449", "#AF363C", "#E7C71F", "#BB5695", "#0885A1",
    "#F3F3F2", "#C8C8C8", "#A0A0A0", "#7A7A79", "#555555", "#343434",
]

# ຂໍ້ມູນບົດຝຶກຫັດ (ໃຊ້ທັງຫົວໜ້າ ແລະ ໜ້າຫຼັກ). patch = ສີປະຈຳບົດ, fg = ສີຕົວອັກສອນເທິງ patch
EXERCISES = {
    2: dict(title="ທົດລອງ Load ຮູບພາບ", short="ໂຫຼດຮູບພາບ", en="Load Image", en_short="Load Image",
            path="ex2_load_image", patch="#627A9D", fg="#FFFFFF"),
    3: dict(title="ປ່ຽນຮູບສີເປັນຮູບສີເທົາ", short="ຮູບສີເທົາ", en="Convert to Grayscale", en_short="Grayscale",
            path="ex3_convert_gray", patch="#7A7A79", fg="#FFFFFF"),
    4: dict(title="ທົດລອງນຳໃຊ້ ImageBox ແລະ HistogramBox", short="Histogram", en="Histogram and Pixel Table", en_short="Pixel Table",
            path="ex4_histogram", patch="#D67E2C", fg="#1E1E1E"),
    5: dict(title="ຮັບຄ່າຂໍ້ມູນພາບຂາວດຳຈາກຮູບພາບສີ", short="ພາບຂາວດຳ", en="Thresholding", en_short="Thresholding",
            path="ex5_thresholding", patch="#343434", fg="#FFFFFF"),
    6: dict(title="ການປັບຄ່າຟິກເຊວຂອງຮູບພາບ ແລະ ການຕອງຮູບພາບ", short="ປັບຄ່າ ແລະ ຕອງຮູບ", en_short="Equalization, Filtering",
            en="Histogram Equalization and Image Filtering",
            path="ex6_equalization_filtering", patch="#67BDAA", fg="#1E1E1E"),
    7: dict(title="ການປະມວນຜົນພາບວິດີໂອຈາກ WebCam", short="ວິດີໂອຈາກ WebCam", en="VDO Webcam Processing", en_short="WebCam",
            path="ex7_webcam", patch="#AF363C", fg="#FFFFFF"),
    8: dict(title="ປະມວນຜົນພາບດ້ວຍ Morphology", short="Morphology", en="Morphological Processing",
            en_short="Dilation, Erosion", path="ex8_morphology", patch="#576C43", fg="#FFFFFF"),
    9: dict(title="ການກວດຈັບຂອບ", short="ກວດຈັບຂອບ", en="Edge Detection", en_short="Edge Detection",
            path="ex9_edge_detection", patch="#BB5695", fg="#FFFFFF"),
    10: dict(title="ການຈັບຄູ່ແຜ່ນແມ່ແບບ", short="ຈັບຄູ່ແມ່ແບບ", en="Template Matching", en_short="Template Matching",
             path="ex10_template_matching", patch="#383D96", fg="#FFFFFF"),
    11: dict(title="ການຈັບຄູ່ຈຸດພິເສດດ້ວຍ ORB ແລະ SIFT", short="ຈັບຄູ່ຈຸດພິເສດ", en="ORB and SIFT Feature Matching",
             en_short="ORB, SIFT", path="ex11_feature_matching", patch="#5E3C6C", fg="#FFFFFF"),
}

_CSS = """
<style>
  /* ພື້ນທີ່ເນື້ອຫາ: ຫຼຸດຊ່ອງວ່າງດ້ານເທິງ ແຕ່ໃຫ້ແຖບສີຢູ່ລຸ່ມແຖບເຄື່ອງມື (Deploy) ຂອງ Streamlit (ສູງ 60px) */
  [data-testid="stMainBlockContainer"] { padding-top: 4.5rem; }

  /* ແຖບ 24 patch ເທິງຂອບດຳ ຄືແຜ່ນ ColorChecker ແທ້ (patch ສີຂາວຈຶ່ງບໍ່ກືນກັບພື້ນຫຼັງ) */
  .cv-strip { display: grid; grid-template-columns: repeat(24, 1fr); gap: 2px; height: 12px;
              padding: 2px; background: #262626; border-radius: 3px; margin-bottom: 1.5rem; }
  .cv-strip span { border-radius: 1px; }

  .cv-head { display: flex; align-items: center; gap: 1rem; }
  .cv-num { flex: none; width: 3.5rem; height: 3.5rem; border-radius: 0.3rem;
            display: grid; place-items: center; background: var(--patch); color: var(--fg);
            font-size: 1.75rem; font-weight: 600; font-variant-numeric: tabular-nums; }
  .cv-head .cv-title { margin: 0; padding: 0; font-size: 2.25rem; font-weight: 600;
                       line-height: 1.35; color: #343434; }
  .cv-head .cv-sub { margin: 0.15rem 0 0; font-size: 1rem; color: #5A5A59; }

  /* ຮູບພາບ: ວາງເທິງພື້ນສີເທົາກາງ (Neutral 5 = 18% gray) */
  [data-testid="stImage"] img { background: #7A7A79; padding: 6px; border-radius: 0.3rem; }

  @media (max-width: 640px) {
    .cv-num { width: 2.75rem; height: 2.75rem; font-size: 1.35rem; }
    .cv-head .cv-title { font-size: 1.6rem; }
  }
</style>
"""


def strip_html() -> str:
    cells = "".join(f'<span style="background:{c}"></span>' for c in COLORCHECKER)
    return f'<div class="cv-strip" aria-hidden="true">{cells}</div>'


def page_header(number: int | None = None, title: str | None = None, subtitle: str | None = None) -> None:
    """ຫົວໜ້າຂອງທຸກໜ້າ: ແຖບ ColorChecker + ເລກບົດຝຶກຫັດ (ສີ patch ຂອງບົດ) + ຫົວຂໍ້."""
    ex = EXERCISES.get(number, {})
    title = escape(title or ex.get("title", ""))
    subtitle = escape(subtitle or ex.get("en", ""))
    badge = ""
    if number is not None:
        badge = (f'<div class="cv-num" style="--patch:{ex["patch"]};--fg:{ex["fg"]}" '
                 f'role="img" aria-label="ບົດຝຶກຫັດ {number}">{number}</div>')
    st.html(f"""{_CSS}{strip_html()}
<header class="cv-head">{badge}
  <div><h1 class="cv-title">{title}</h1><p class="cv-sub">{subtitle}</p></div>
</header>""")
