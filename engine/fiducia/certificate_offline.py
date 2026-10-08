"""Reading a calibration certificate without a network or an account.

The model-based reader handles any layout. This one handles the common case
well enough to save the typing, using only what is already on the computer:

1. **The text.** A PDF made on a computer carries its text, with positions,
   and that is read directly. A scan -- a scanned PDF page, a PNG or a JPEG --
   goes through the text recognition built into Windows 10 and 11.

2. **Rows.** Words are grouped into rows by their height on the page, so a
   table comes out as a table however the PDF or the recogniser ordered it.

3. **Rules.** Certificates follow a small number of conventions: a labelled
   focal length, labelled principal points, a fiducial table and a distortion
   table under recognisable headings. Columns are matched by position under
   their heading, so "mean" means the column beneath the word "mean".

Every value carries the row it was read from, exactly as the model reader's
do, and the same arithmetic checks run afterwards. Anything not found is
listed for the operator to type in; nothing is guessed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Callable, Optional

__all__ = ["available", "read", "OfflineReadError"]


class OfflineReadError(RuntimeError):
    """A message for the operator, as-is."""


# -- what is on this computer ----------------------------------------------


def _pdf_support() -> bool:
    try:
        import pypdfium2  # noqa: F401
        return True
    except ImportError:
        return False


def _ocr_engine():
    """Windows' own text recogniser, or None where there is none."""
    try:
        from winrt.windows.globalization import Language
        from winrt.windows.media.ocr import OcrEngine
    except ImportError:
        return None
    for tag in ("en-GB", "en-US"):
        try:
            if OcrEngine.is_language_supported(Language(tag)):
                return OcrEngine.try_create_from_language(Language(tag))
        except Exception:
            continue
    try:
        return OcrEngine.try_create_from_user_profile_languages()
    except Exception:
        return None


def available() -> dict:
    """Whether offline reading can run here, and what it can read."""
    pdf = _pdf_support()
    ocr = _ocr_engine() is not None
    if pdf and ocr:
        reason = ""
    elif not ocr:
        reason = ("Windows text recognition is unavailable, so only PDFs made on a "
                  "computer can be read offline. Scans need an English language "
                  "pack with optical character recognition, added in Windows "
                  "Settings under Time & language.")
    else:
        reason = "The pypdfium2 package is not installed, so PDFs cannot be read."
    return {"installed": pdf or ocr, "pdf": pdf, "ocr": ocr, "reason": reason}


# -- words and rows ----------------------------------------------------------


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return max(1e-6, self.y1 - self.y0)


@dataclass
class Number:
    value: float
    cx: float
    raw: str
    corrected: bool = False


@dataclass
class Row:
    words: list
    page: int
    numbers: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

    @property
    def lower(self) -> str:
        return self.text.lower()


def _words_from_pdf_text(page, index: int) -> list[Word]:
    """Words and their boxes from a PDF page's own text layer."""
    textpage = page.get_textpage()
    height = page.get_height()
    words: list[Word] = []
    current: list = []

    def flush():
        if current:
            text = "".join(ch for ch, _ in current).strip()
            if text:
                words.append(Word(
                    text,
                    min(box[0] for _, box in current), min(box[1] for _, box in current),
                    max(box[2] for _, box in current), max(box[3] for _, box in current),
                    index,
                ))
            current.clear()

    for i in range(textpage.count_chars()):
        ch = textpage.get_text_range(i, 1)
        if not ch or ch.isspace():
            flush()
            continue
        # Loose boxes span the font's full height, so a full stop or a minus
        # sign is as tall as the digits beside it and stays in the same word.
        left, bottom, right, top = textpage.get_charbox(i, loose=True)
        box = (left, height - top, right, height - bottom)   # y down, like a scan
        if current:
            previous = current[-1][1]
            gap = box[0] - previous[2]
            char_height = max(1e-6, previous[3] - previous[1])
            # A wide gap or a jump in line starts a new word, even without a space.
            if gap > char_height * 0.5 or abs(box[1] - previous[1]) > char_height * 0.5:
                flush()
        current.append((ch, box))
    flush()
    textpage.close()
    return words


def _recognise(image, index: int, engine, fit: bool = True) -> list[Word]:
    """Words and their boxes from a picture, through Windows text recognition."""
    import asyncio

    from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import DataWriter

    gray = image.convert("L")
    # The recogniser does best with text 20-40 pixels tall. Scans at 150 dpi
    # or below are enlarged; anything past its size limit is reduced.
    scale = 1.0
    if fit and max(gray.size) < 2400:
        scale = 2400 / max(gray.size)
    limit = OcrEngine.max_image_dimension
    if max(gray.size) * scale > limit:
        scale = limit / max(gray.size)
    if abs(scale - 1.0) > 1e-3:
        from PIL import Image
        gray = gray.resize((round(gray.width * scale), round(gray.height * scale)),
                           Image.Resampling.LANCZOS)

    writer = DataWriter()
    writer.write_bytes(gray.tobytes())
    bitmap = SoftwareBitmap.create_copy_from_buffer(
        writer.detach_buffer(), BitmapPixelFormat.GRAY8, gray.width, gray.height,
    )

    async def run():
        return await engine.recognize_async(bitmap)

    result = asyncio.run(run())
    words: list[Word] = []
    for line in result.lines:
        for word in line.words:
            rect = word.bounding_rect
            words.append(Word(
                word.text,
                rect.x / scale, rect.y / scale,
                (rect.x + rect.width) / scale, (rect.y + rect.height) / scale,
                index,
            ))
    return words


# Windows recognition reads running text well but skips a token standing alone
# in white space when it is short -- a single digit, or "0,9" -- which is most
# of a calibration table. So after the page is read, every word-sized mark it
# did not account for is cut out and read again on a sheet of its own, between
# two printed words that give the recogniser the context it wants.
_SHEET_LINES = 4           # taller sheets make the recogniser skip lines again
_SHEET_HEIGHT = 36          # every mark is scaled to this height on the sheet


def _marks(gray, text_height: float):
    """Boxes around word-sized marks on the page, by morphology."""
    import cv2
    import numpy as np

    pixels = cv2.medianBlur(np.asarray(gray), 3)
    _, ink = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    reach = max(3, int(text_height * 0.55))
    joined = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (reach, 3)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    boxes = []
    for x, y, w, h, area in stats[1:]:
        if text_height * 0.45 <= h <= text_height * 2.2 and w >= 3 and w < gray.width * 0.8:
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def _covered(box, words: list[Word], share: float = 0.3) -> bool:
    """Whether words already read cover this much of the box between them."""
    x, y, w, h = box
    area = 0.0
    for word in words:
        dx = min(x + w, word.x1) - max(x, word.x0)
        dy = min(y + h, word.y1) - max(y, word.y0)
        if dx > 0 and dy > 0:
            area += dx * dy
    return area > share * w * h


def _reread(gray, boxes, index: int, engine) -> list[Word]:
    """Read marks one per line on printed sheets, and place the words back."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("arial.ttf", int(_SHEET_HEIGHT * 1.3))
    except OSError:
        font = ImageFont.load_default()
    line = int(_SHEET_HEIGHT * 2.6)
    left = int(_SHEET_HEIGHT * 5)
    found: list[Word] = []

    for start in range(0, len(boxes), _SHEET_LINES):
        batch = boxes[start:start + _SHEET_LINES]
        placed = []
        widest = 0
        crops = []
        for x, y, w, h in batch:
            scale = _SHEET_HEIGHT / h
            crop = gray.crop((max(0, x - 2), max(0, y - 2), x + w + 2, y + h + 2))
            crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                               Image.Resampling.LANCZOS)
            crops.append((crop, scale))
            widest = max(widest, crop.width)
        sheet = Image.new("L", (left + widest + _SHEET_HEIGHT * 5, line * len(batch) + line), 250)
        draw = ImageDraw.Draw(sheet)
        for i, ((crop, scale), box) in enumerate(zip(crops, batch)):
            middle = i * line + line // 2
            draw.text((_SHEET_HEIGHT // 2, middle), "Value", fill=20, font=font, anchor="lm")
            sheet.paste(crop, (left, middle - crop.height // 2))
            draw.text((left + crop.width + _SHEET_HEIGHT, middle), "mm", fill=20, font=font, anchor="lm")
            placed.append((middle - line // 2, left, left + crop.width, scale, box))

        for word in _recognise(sheet, index, engine, fit=False):
            slot = int(word.cy // line)
            if slot >= len(placed):
                continue
            _, x0, x1, scale, (bx, by, bw, bh) = placed[slot]
            if word.x1 <= x0 + 2 or word.x0 >= x1 - 2:
                continue          # one of the printed words
            found.append(Word(
                word.text,
                bx + (max(word.x0, x0) - x0) / scale, by,
                bx + (min(word.x1, x1) - x0) / scale, by + bh,
                index,
            ))
    return found


def _recognise_page(image, index: int, engine) -> list[Word]:
    """Read a page, then read again whatever the first pass skipped."""
    gray = image.convert("L")
    words = _recognise(gray, index, engine)
    heights = [w.height for w in words if any(ch.isalnum() for ch in w.text)]
    text_height = median(heights) if heights else max(12.0, gray.height / 120)
    missed = [box for box in _marks(gray, text_height) if not _covered(box, words)]
    if missed:
        again = _reread(gray, missed, index, engine)
        words += [w for w in again
                  if not _covered((w.x0, w.y0, w.x1 - w.x0, w.y1 - w.y0), words, 0.2)]
    return words


def _rows(words: list[Word]) -> list[Row]:
    """Group words into rows by the height of their centres on the page."""
    rows: list[Row] = []
    for page in sorted({word.page for word in words}):
        on_page = sorted((w for w in words if w.page == page), key=lambda w: w.cy)
        if not on_page:
            continue
        tolerance = median(w.height for w in on_page) * 0.55
        group: list[Word] = []
        centre = None
        for word in on_page:
            if group and abs(word.cy - centre) > tolerance:
                rows.append(Row(sorted(group, key=lambda w: w.x0), page))
                group = []
            group.append(word)
            centre = sum(w.cy for w in group) / len(group)
        if group:
            rows.append(Row(sorted(group, key=lambda w: w.x0), page))
    for row in rows:
        row.numbers = _numbers(row.words)
    return rows


# -- numbers -----------------------------------------------------------------

_MINUS = str.maketrans({"−": "-", "–": "-", "—": "-", "‐": "-"})
_NUMBER = re.compile(r"[-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][-+]?\d+)?")
# Letters a recogniser commonly returns for digits.
_LOOKALIKE = str.maketrans({"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1",
                            "S": "5", "B": "8"})
_UNIT = re.compile(r"(mm|µm|μm|um|microns?|°|deg)$", re.IGNORECASE)


def _parse_number(token: str, allow_fixes: bool) -> Optional[tuple[float, bool]]:
    """A whole word as a number, or None if the word is not one.

    Only a word that is a number counts -- "RC30" is not thirty -- after a
    leading label such as "x=" and a trailing unit are taken off.
    """
    text = token.translate(_MINUS).strip("()[]{};:,")
    text = re.split(r"[=:]", text)[-1]
    text = _UNIT.sub("", text).strip()
    if not text:
        return None
    corrected = False
    if not _NUMBER.fullmatch(text) and allow_fixes:
        digits = sum(ch.isdigit() for ch in text)
        if digits and digits >= 0.6 * len(re.sub(r"[-+.,]", "", text)):
            fixed = text.translate(_LOOKALIKE)
            if _NUMBER.fullmatch(fixed):
                text, corrected = fixed, True
    if not _NUMBER.fullmatch(text):
        return None
    # A decimal comma, as continental certificates print it.
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        return float(text.replace(",", "")), corrected
    except ValueError:
        return None


_ALLOW_FIXES = {"on": False}


def _numbers(words: list[Word]) -> list[Number]:
    out = []
    previous = ""
    for word in words:
        text = word.text
        # "0.000 mm ± 0.002mm": the figure after ± is a tolerance, not a value.
        tolerance = previous in ("±", "+/-", "+-") or text.startswith(("±", "+/-"))
        previous = text
        if tolerance:
            continue
        parsed = _parse_number(text, _ALLOW_FIXES["on"])
        if parsed is not None:
            out.append(Number(parsed[0], word.cx, text, parsed[1]))
    return out


# -- the rules ---------------------------------------------------------------


class _Reader:
    def __init__(self, rows: list[Row], recognised: bool):
        self.rows = rows
        self.recognised = recognised
        self.corrections = False

    def confidence(self, numbers: list[Number], strong: bool = True) -> str:
        if any(n.corrected for n in numbers):
            self.corrections = True
            return "low"
        if self.recognised or not strong:
            return "medium"
        return "high"

    def reading(self, value: float, row: Row, numbers: list[Number], strong=True) -> dict:
        return {"value": value, "sourceText": row.text[:160],
                "confidence": self.confidence(numbers, strong)}

    def find(self, pattern: str, start: int = 0, exclude: Optional[str] = None):
        regex = re.compile(pattern, re.IGNORECASE)
        skip = re.compile(exclude, re.IGNORECASE) if exclude else None
        for index in range(start, len(self.rows)):
            text = self.rows[index].text
            if regex.search(text) and not (skip and skip.search(text)):
                yield index, self.rows[index]

    # -- focal length ---------------------------------------------------

    def focal_length(self) -> Optional[dict]:
        for pattern, strong in (
            (r"calibrated\s+(focal\s+length|principal\s+distance)|\bCFL\b", True),
            (r"principal\s+distance|focal\s+length|\bc\s*=", False),
        ):
            for index, row in self.find(pattern, exclude=r"nominal"):
                for candidate in (row, *self.rows[index + 1:index + 3]):
                    values = [n for n in candidate.numbers if 5 <= abs(n.value) <= 1000]
                    if values:
                        return self.reading(abs(values[0].value), candidate, values[:1], strong)
        return None

    # -- principal point --------------------------------------------------

    def point(self, pattern: str, exclude: Optional[str] = None):
        """An x, y pair from a labelled row, or the rows just after it."""
        for index, row in self.find(pattern, exclude=exclude):
            micro = bool(re.search(r"µm|μm|\bum\b|micron", row.lower))
            found: list[Number] = []
            source = row
            for candidate in (row, *self.rows[index + 1:index + 3]):
                # The y value may sit on the next row under its own label,
                # as in "X_ppa 0.000 mm" then "(Level 2) Y_ppa 0.000 mm".
                y_row = re.search(r"(^|\s|\()y(_?pp[aso]?|[0p])?\b", candidate.lower)
                if candidate is not row and re.search(r"[a-z]{4,}", candidate.lower) \
                        and not y_row:
                    break
                numbers = candidate.numbers
                if candidate is not row and y_row:
                    # Only figures after the y label: "(Level 2) Y_ppa 0.000" is not 2.
                    label = next((w for w in candidate.words
                                  if re.fullmatch(r"\(?y(_?pp[aso]?|[0p])?\)?[:=]?", w.text.lower())),
                                 None)
                    if label is not None:
                        numbers = [n for n in numbers if n.cx > label.cx]
                found += [n for n in numbers if abs(n.value) < (500 if micro else 5)
                          and "(" not in n.raw and ")" not in n.raw]
                if len(found) >= 2:
                    break
            if not found:
                continue
            factor = 0.001 if micro else 1.0
            x = self.reading(found[0].value * factor, source, found[:1])
            y = self.reading(found[1].value * factor, source, found[1:2]) if len(found) > 1 else None
            return x, y
        return None, None

    # -- fiducials ------------------------------------------------------

    def fiducials(self) -> list[dict]:
        heading = next(self.find(r"fiducial"), None)
        if heading is None:
            return []
        start = heading[0]
        window = self.rows[start:start + 40]

        marks: list[tuple[str, float, float, Row, list]] = []

        # Transposed: one row of x values and one of y values.
        xs = next((r for r in window if re.match(r"\s*x\b", r.lower) and len(r.numbers) >= 4), None)
        ys = next((r for r in window if re.match(r"\s*y\b", r.lower) and len(r.numbers) >= 4), None)
        if xs and not ys:
            # A lone "y" is easily lost in recognition; take the next row of as many figures.
            after = window[window.index(xs) + 1:window.index(xs) + 3]
            ys = next((r for r in after if len(r.numbers) == len(xs.numbers)), None)
        if xs and ys and len(xs.numbers) == len(ys.numbers):
            labels = next((r.numbers for r in window
                           if len(r.numbers) == len(xs.numbers) and r not in (xs, ys)
                           and all(float(n.value).is_integer() for n in r.numbers)), None)
            for i, (nx, ny) in enumerate(zip(xs.numbers, ys.numbers)):
                label = str(int(labels[i].value)) if labels else str(i + 1)
                marks.append((label, nx.value, ny.value, xs, [nx, ny]))
        else:
            for row in window[1:]:
                if marks and re.search(r"distortion|principal|focal", row.lower):
                    break
                numbers = row.numbers
                if len(numbers) < 2:
                    continue
                x, y = numbers[-2], numbers[-1]
                if not (max(abs(x.value), abs(y.value)) >= 10
                        and abs(x.value) <= 200 and abs(y.value) <= 200):
                    continue
                label_words = [w.text for w in row.words if w.cx < x.cx - 1e-6]
                label = " ".join(label_words).strip() or str(len(marks) + 1)
                marks.append((label, x.value, y.value, row, [x, y]))
                if len(marks) >= 8:
                    break

        if not marks:
            return []
        extent = max(max(abs(m[1]), abs(m[2])) for m in marks)
        near = extent * 0.25
        out, taken = [], set()
        for label, x, y, row, numbers in marks:
            horizontal = "middle" if abs(x) < near else ("right" if x > 0 else "left")
            vertical = "middle" if abs(y) < near else ("top" if y > 0 else "bottom")
            if horizontal == "middle" and vertical == "middle":
                continue
            if vertical == "middle":
                slot = f"{horizontal}_middle"
            elif horizontal == "middle":
                slot = f"{vertical}_middle"
            else:
                slot = f"{vertical}_{horizontal}"
            if slot in taken:
                continue
            taken.add(slot)
            self.confidence(numbers)
            out.append({"slot": slot, "x": x, "y": y, "label": label[:40]})
        return out

    # -- distortion -----------------------------------------------------

    def distortion(self, focal: Optional[float]) -> tuple[list[dict], str, list[str]]:
        notes: list[str] = []
        heading = next(self.find(r"radial\s+distortion"), None) or next(self.find(r"distortion"), None)
        if heading is None:
            return [], "", notes
        start = heading[0]

        # The header is everything between the heading and the first row of
        # figures; the table runs until two rows in a row carry none.
        header_words: list[Word] = []
        header_text = ""
        table: list[Row] = []
        misses = 0
        for row in self.rows[start + 1:start + 60]:
            numeric = row.numbers and len(row.numbers) >= len(row.words) * 0.5
            # A row of nothing but figures belongs to the table even when the
            # recogniser kept only its first column; it is reported below.
            if numeric and (table or len(row.numbers) >= 2
                            or len(row.numbers) == len(row.words)):
                table.append(row)
                misses = 0
            elif table:
                misses += 1
                if misses >= 2:
                    break
            else:
                header_words += row.words
                header_text += " " + row.lower
        if len(table) < 2:
            return [], "", notes

        def column(pattern: str) -> Optional[float]:
            regex = re.compile(pattern, re.IGNORECASE)
            hits = [w.cx for w in header_words if regex.search(w.text)]
            return hits[-1] if hits else None

        mean_x = column(r"^(mean|average|avg)")
        radius_x = column(r"^(radi(?!al)|distance|r$|r\(|height)")
        angle_x = column(r"^(angle|deg|field|°)")
        key_x = radius_x if radius_x is not None else angle_x
        from_angle = radius_x is None and angle_x is not None and bool(focal)

        def key_of(row: Row) -> Number:
            if key_x is None:
                return row.numbers[0]
            return min(row.numbers, key=lambda n: abs(n.cx - key_x))

        # The distortion columns, found from where the figures actually sit,
        # so a value the recogniser missed leaves a gap instead of shifting
        # its neighbours into the wrong column.
        height = median(w.height for row in table for w in row.words)
        centres = sorted(n.cx for row in table for n in row.numbers if n is not key_of(row))
        columns: list[list[float]] = []
        for x in centres:
            if columns and x - columns[-1][-1] < height * 2.5:
                columns[-1].append(x)
            else:
                columns.append([x])
        columns = [sum(c) / len(c) for c in columns if len(c) >= max(2, len(table) // 3)]
        if not columns:
            return [], "", notes
        if mean_x is not None:
            columns = [min(columns, key=lambda c: abs(c - mean_x))]

        micro = bool(re.search(r"µm|μm|\bum\b|micron", header_text))
        rows_out: list[dict] = []
        incomplete: list[str] = []
        previous = -1.0
        for row in table:
            key = key_of(row)
            radius = (focal * math.tan(math.radians(key.value)) if from_angle else key.value)
            if not (0 <= radius <= 250) or radius <= previous:
                continue
            values: dict[int, Number] = {}
            for number in row.numbers:
                if number is key:
                    continue
                slot = min(range(len(columns)), key=lambda i: abs(columns[i] - number.cx))
                if abs(columns[slot] - number.cx) < height * 3:
                    values.setdefault(slot, number)
            if not values:
                incomplete.append(f"{key.value:g}")
                continue
            chosen = list(values.values())
            value = sum(n.value for n in chosen) / len(chosen)
            confidence = self.confidence([key, *chosen])
            if len(values) < len(columns):
                incomplete.append(f"{key.value:g}")
                confidence = "low"
            rows_out.append({"radiusMm": round(radius, 4), "distortionUm": value,
                             "confidence": confidence})
            previous = radius

        if not rows_out:
            return [], "", notes

        units = "micrometres"
        if not micro and max(abs(r["distortionUm"]) for r in rows_out) < 0.3:
            for r in rows_out:
                r["distortionUm"] = r["distortionUm"] * 1000.0
            units = "millimetres"
        for r in rows_out:
            r.pop("confidence", None)
        if from_angle:
            notes.append("Radial distances were computed from the tabulated field angles "
                         "and the focal length.")
        if mean_x is None and len(columns) > 1:
            notes.append(f"No mean column was found, so each distortion value is the "
                         f"average of {len(columns)} columns. Check that is how the table "
                         "is meant to be read.")
        if incomplete:
            label = "angles" if from_angle else "radii"
            notes.append(f"Some distortion figures could not be read, at {label} "
                         f"{', '.join(incomplete)}. Those rows are left out or averaged "
                         "from fewer columns; check them against the page.")
        return rows_out, units, notes

    # -- descriptive fields ---------------------------------------------

    def text_after(self, pattern: str, value: str) -> str:
        for _, row in self.find(pattern):
            match = re.search(value, row.text, re.IGNORECASE)
            if match:
                text = match.group(match.lastindex or 0)
                # Stop where the next label on the same line begins.
                text = re.split(r"\s+(?:lens\s+|camera\s+)?(?:serial|s/n|date)\b", text,
                                flags=re.IGNORECASE)[0]
                return text.strip(" :.-")[:60]
        return ""

    def camera_name(self) -> str:
        # A labelled "Camera: UltraCam Eagle, S/N ..." line is the best source.
        for row in self.rows[:80]:
            match = re.match(r"\s*camera(?:\s+type)?\s*:\s*(.+)$", row.text, re.IGNORECASE)
            if match:
                name = re.split(r"\s*,|\s+(?:serial|s/n|no\.|lens)\b", match.group(1),
                                flags=re.IGNORECASE)[0].strip()
                if name:
                    return name[:60]
        maker = (r"\b(Wild\s*RC\s*-?\d+\w*|Leica\s*RC\s*-?\d+\w*|RC\s*-?\d{2}\b|Zeiss\s*RMK[\w /-]*|"
                 r"RMK\s*[\w/-]+|UltraCam\s*\w*|DMC\s*\w*|Vexcel\s*\w*|Z/I\s*\w*)")
        for row in self.rows[:80]:
            match = re.search(maker, row.text, re.IGNORECASE)
            if match:
                name = re.split(r"\s+(?:serial|s/n|no\.|lens)\b", match.group(1),
                                flags=re.IGNORECASE)[0]
                return name.strip()[:60]
        return ""


def _collect_words(path: Path, progress) -> tuple[list[Word], bool, int]:
    """All words in the document, and whether recognition was needed."""
    suffix = path.suffix.lower()
    engine = None

    def need_engine():
        nonlocal engine
        if engine is None:
            engine = _ocr_engine()
            if engine is None:
                raise OfflineReadError(
                    "This certificate is a scan, and Windows text recognition is not "
                    "available on this computer. Add the English language pack with "
                    "optical character recognition in Windows Settings, or choose "
                    "Claude or ChatGPT in Settings."
                )
        return engine

    if suffix == ".pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise OfflineReadError("Reading PDFs offline needs the pypdfium2 package.") from exc
        try:
            document = pdfium.PdfDocument(str(path))
        except Exception as exc:
            raise OfflineReadError(f"{path.name} could not be opened as a PDF: {exc}") from exc
        pages = min(len(document), 12)
        words: list[Word] = []
        recognised = False
        for index in range(pages):
            page = document[index]
            found = _words_from_pdf_text(page, index)
            # A page whose text layer holds almost no digits is a scan.
            if sum(ch.isdigit() for w in found for ch in w.text) < 20:
                if progress:
                    progress(0.2 + 0.5 * index / pages, f"Recognising text on page {index + 1}")
                image = page.render(scale=300 / 72).to_pil()
                found = _recognise_page(image, index, need_engine())
                recognised = True
            words += found
            page.close()
        document.close()
        return words, recognised, pages

    from PIL import Image, ImageOps
    try:
        image = ImageOps.exif_transpose(Image.open(path))
    except Exception as exc:
        raise OfflineReadError(f"{path.name} could not be opened as an image: {exc}") from exc
    if progress:
        progress(0.3, "Recognising text")
    return _recognise_page(image, 0, need_engine()), True, 1


def read(path: Path, progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """Read a certificate into the same shape the model readers return."""
    words, recognised, pages = _collect_words(path, progress)
    if not words:
        raise OfflineReadError(
            f"No text was found in {path.name}. If it is a photograph of the page, "
            "try a flatter, sharper scan, or read it with Claude or ChatGPT."
        )

    if progress:
        progress(0.75, "Finding the calibration values")

    _ALLOW_FIXES["on"] = recognised
    try:
        rows = _rows(words)
    finally:
        _ALLOW_FIXES["on"] = False
    reader = _Reader(rows, recognised)

    focal = reader.focal_length()
    ppa_x, ppa_y = reader.point(r"\bPPA\b|autocollimation")
    pps_x, pps_y = reader.point(r"\bPPS\b|point\s+of\s+(best\s+)?symmetry")
    ppo_x, ppo_y = reader.point(r"\bPPO\b|principal\s+point(\s+offset)?\b|\bx[0p]\b",
                                exclude=r"autocollimation|symmetry|\bPPA\b|\bPPS\b")
    fiducials = reader.fiducials()
    table, units, table_notes = reader.distortion(focal["value"] if focal else None)

    level = "medium" if recognised else "high"
    pitch = None
    for _, row in reader.find(r"pixel\s+(size|pitch)"):
        # Read from the text, since "5.200µm*5.200µm" is one word.
        match = re.search(r"pixel\s+(?:size|pitch)\D{0,12}?(\d+(?:[.,]\d+)?)\s*(µm|μm|um|mm)?",
                          row.text, re.IGNORECASE)
        if match:
            value = float(match.group(1).replace(",", "."))
            in_mm = (match.group(2) or "").lower() == "mm" or value <= 0.5
            pitch = {"value": value if in_mm else value / 1000, "sourceText": row.text[:160],
                     "confidence": level}
            break
    columns = rows_count = None
    for _, row in reader.find(r"pixel|sensor|image\s+size|format|array"):
        match = re.search(r"(\d{3,5})\s*[x×X*]\s*(\d{3,5})", row.text)
        if match:
            columns = {"value": float(match.group(1)), "sourceText": row.text[:160],
                       "confidence": level}
            rows_count = {"value": float(match.group(2)), "sourceText": row.text[:160],
                          "confidence": level}
            break
    if columns is None:
        # Digital frame cameras quote the format per axis: "cross track
        # 104.052mm 20010pixel" is the image width, "long track ... 13080pixel"
        # its height (the flight direction runs down the image).
        cross = next((r for _, r in reader.find(r"cross[\s-]*track.*?\d{3,5}\s*pix")), None)
        along = next((r for _, r in reader.find(r"long[\s-]*track.*?\d{3,5}\s*pix")), None)
        if cross is not None and along is not None:
            count = lambda r: float(re.search(r"(\d{3,5})\s*pix", r.text, re.IGNORECASE).group(1))
            columns = {"value": count(cross), "sourceText": cross.text[:160], "confidence": level}
            rows_count = {"value": count(along), "sourceText": along.text[:160], "confidence": level}

    has_fiducial_text = any("fiducial" in row.lower for row in rows)
    kind = ("film" if fiducials or has_fiducial_text
            else "digital" if pitch else "unknown")
    if ppo_x or ppo_y:
        mode = "ppo"
    elif ppa_x or pps_x:
        mode = "ppa_pps"
    else:
        mode = "none"

    # Digital cameras are often delivered with distortion already removed
    # from the images, and say so instead of giving a table.
    removed = next((r for _, r in reader.find(
        r"distortion.*(less\s+than|below|negligible|removed|corrected)")), None)
    if removed is not None and len(table) < 3:
        table, units = [], ""
        table_notes = [f"The certificate states distortion is already removed "
                       f"(“{removed.text[:90]}”), so the distortion "
                       "coefficients are left at zero."]

    unreadable = []
    if not focal:
        unreadable.append("focal length")
    if mode == "none":
        unreadable.append("principal point")
    if kind != "digital" and not fiducials:
        unreadable.append("fiducial marks")
    if not table and removed is None:
        unreadable.append("radial distortion table")
    if kind == "digital" and columns is not None:
        table_notes.append(
            f"The sensor is read as {columns['value']:.0f} columns by "
            f"{rows_count['value']:.0f} rows, as the certificate states it. Photos "
            "delivered rotated have these the other way round; Fiducia checks this "
            "against your images.")

    notes = [
        "Read offline from the PDF's own text." if not recognised else
        "Read offline with Windows text recognition. Recognition misreads digits "
        "more often than Claude or ChatGPT; check every value against the page.",
        *table_notes,
    ]
    if reader.corrections:
        notes.append("Some characters were read as letters and taken to be digits "
                     "(O as 0, l as 1). Those values are marked low confidence.")

    # The principal point per delivered rotation, as UltraCam certificates
    # tabulate it: "Level 3 90 0.000 0.000" (degrees clockwise, then x, y mm).
    rotation_table = []
    seen_rotations = set()
    for row in rows:
        match = re.search(r"level\s*3\s+(0|90|180|270)\s+(-?\d+(?:[.,]\d+)?)\s+(-?\d+(?:[.,]\d+)?)",
                          row.text, re.IGNORECASE)
        if match and int(match.group(1)) not in seen_rotations:
            seen_rotations.add(int(match.group(1)))
            rotation_table.append({
                "rotationDeg": int(match.group(1)),
                "x": float(match.group(2).replace(",", ".")),
                "y": float(match.group(3).replace(",", ".")),
            })

    date = reader.text_after(r"date\s+of\s+calibration|calibration\s+date|date",
                             r"(\d{1,2}[./ -]\w{2,9}[./ -]\d{2,4}|\d{4}-\d{2}-\d{2}"
                             r"|[A-Za-z]{3,9}[-./ ]\d{1,2}[-./, ]+\d{4})")
    data = {
        "cameraName": reader.camera_name(),
        "lensType": reader.text_after(r"\blens\b(?!\s+(distortion|resolving))",
                                      r"lens(?:\s+type)?\s*[:.]?\s*(.+)$"),
        "serialNumber": reader.text_after(r"serial", r"serial\s*(?:no\.?|number|#)?\s*[:.]?\s*([A-Z0-9-]*\d[A-Z0-9-]*)"),
        "calibrationDate": date,
        "cameraKind": kind,
        "focalLength": focal,
        "principalPointMode": mode,
        "ppoX": ppo_x, "ppoY": ppo_y,
        "ppaX": ppa_x, "ppaY": ppa_y,
        "ppsX": pps_x, "ppsY": pps_y,
        "fiducials": fiducials,
        "rotationTable": rotation_table,
        "distortionTable": table,
        "distortionUnitsAsPrinted": units,
        "radialCoefficients": [],
        "decenteringP1": None, "decenteringP2": None,
        "pixelPitchMm": pitch, "columns": columns, "rows": rows_count,
        "notes": " ".join(notes),
        "unreadable": unreadable,
    }
    usage = {
        "method": "text recognition" if recognised else "PDF text",
        "pages": pages,
        "rows": len(rows),
    }
    return data, usage
