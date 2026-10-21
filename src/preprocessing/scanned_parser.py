from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf  

from src.preprocessing.tree_buillder import NaryTree, Node
from src.preprocessing.chunker import chunk_document_tree

logger = logging.getLogger(__name__)

_CAPTION_RE = re.compile(
    r"^\s*(Table|Figure|Fig\.?|Appendix)\s+[A-Z0-9]+\s*[:.\-\u2013]",
    re.IGNORECASE,
)

_MIN_OCR_CHARS = 20



def _save_image(image_data: bytes, output_dir: Path) -> str:
    """Persist *image_data* to *output_dir* and return the absolute path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    img_hash = hashlib.md5(image_data).hexdigest()[:12]
    ext = "png"
    if image_data[:3] == b"\xff\xd8\xff":
        ext = "jpg"
    elif image_data[:4] == b"\x89PNG":
        ext = "png"

    file_name = f"img_{img_hash}.{ext}"
    file_path = output_dir / file_name
    if not file_path.exists():
        file_path.write_bytes(image_data)
        logger.debug("Saved image → %s", file_path)
    return str(file_path.resolve())


def _html_table_to_markdown(html: str) -> str:
    """Convert an HTML table string to Markdown (identical to native_parser)."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")
        if not rows:
            return html

        md_rows: List[str] = []
        for i, tr in enumerate(rows):
            cells = [
                td.get_text(separator=" ", strip=True)
                for td in tr.find_all(["th", "td"])
            ]
            md_rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")

        return "\n".join(md_rows)

    except ImportError:
        logger.warning("BeautifulSoup not installed – returning raw HTML for table.")
        return html


def _ocr_pil_image(pil_image: Any) -> Tuple[str, str]:
    """
    Run Tesseract OCR on a PIL Image once, producing both plain text and hOCR XML.

    Returns
    -------
    (text, hocr) : Tuple[str, str]
    """
    text = ""
    hocr = ""

    try:
        import pytesseract

        raw_text = pytesseract.image_to_string(pil_image, lang="eng")
        if isinstance(raw_text, str):
            text = raw_text
        elif isinstance(raw_text, bytes):
            text = raw_text.decode("utf-8", errors="replace")
        elif raw_text:
            text = str(raw_text)

        try:
            hocr_out = pytesseract.image_to_pdf_or_hocr(
                pil_image, extension="hocr", lang="eng"
            )
            if isinstance(hocr_out, bytes):
                hocr = hocr_out.decode("utf-8", errors="replace")
            elif isinstance(hocr_out, str):
                hocr = hocr_out
        except Exception as exc:
            logger.debug("hOCR extraction skipped: %s", exc)

    except ImportError:
        logger.warning("pytesseract or Pillow not installed; skipping OCR.")
    except Exception as exc:
        logger.debug("OCR processing failed: %s", exc)

    return text, hocr


def _parse_hocr_font_sizes(hocr: str) -> List[float]:
    """
    Extract font-size values from hOCR ``x_fsize`` hints.

    Tesseract embeds ``x_fsize <N>`` inside ``title`` attributes of
    ``<span class="ocr_word">`` elements.
    """
    sizes: List[float] = []
    if not hocr:
        return sizes

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(hocr, "html.parser")
        for span in soup.find_all("span", class_="ocr_word"):
            title = span.get("title", "")
            if isinstance(title, str):
                m = re.search(r"x_fsize\s+(\d+)", title)
                if m:
                    sizes.append(float(m.group(1)))
    except Exception as exc:
        logger.debug("Failed parsing hOCR font sizes: %s", exc)

    return sizes


def _looks_like_heading(line: str) -> bool:
    """
    Simple heuristic: short line that ends without sentence punctuation
    and has title-case or uppercase words is likely a heading.
    """
    if len(line) > 100:
        return False
    stripped = line.rstrip()
    if stripped and stripped[-1] in (".", "?", "!", ",", ";", ":"):
        return False
    words = stripped.split()
    if not words:
        return False
    title_case_count = sum(1 for w in words if w and w[0].isupper())
    return title_case_count / len(words) >= 0.6


def _classify_scanned_elements(
    page_text: str,
    page_num: int,
    hocr_sizes: List[float],
) -> List[Dict[str, Any]]:
    
    from collections import Counter

    if hocr_sizes:
        size_counter: Counter = Counter(round(s, 0) for s in hocr_sizes)
        body_size = size_counter.most_common(1)[0][0]
        heading_candidates = sorted(
            {s for s in size_counter if s > body_size + 2.0},
            reverse=True,
        )
        heading_level_map = {sz: min(i + 1, 3) for i, sz in enumerate(heading_candidates)}
    else:
        body_size = 10.0
        heading_level_map = {}

    elements: List[Dict[str, Any]] = []
    lines = page_text.splitlines()

    hocr_idx = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        words = line.split()
        word_count = len(words)

        if hocr_sizes and hocr_idx < len(hocr_sizes):
            line_word_sizes = hocr_sizes[hocr_idx : hocr_idx + word_count]
            line_size = max(line_word_sizes) if line_word_sizes else body_size
        else:
            line_size = body_size

        hocr_idx = min(hocr_idx + word_count, len(hocr_sizes))

        if _CAPTION_RE.match(line):
            node_type = "Paragraph"
            level = 4

        elif line_size in heading_level_map:
            node_type = "Title"
            level = heading_level_map[line_size]

        elif len(line) < 80 and (line.isupper() or _looks_like_heading(line)):
            node_type = "Title"
            level = 3  # Default sub-heading level when font size map is absent

        elif re.match(r"^[\-\*\•\u2022]\s+|^\d+[\.\)]\s+|^[a-z]\)\s+", line):
            node_type = "ListItem"
            level = 5

        else:
            node_type = "Paragraph"
            level = 4

        elements.append(
            {
                "text": line,
                "type": node_type,
                "level": level,
                "metadata": {"page": page_num},
            }
        )

    return elements


def _detect_tables_via_unstructured(
    page_image_bytes: bytes,
    page_num: int,
) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    try:
        import io
        from unstructured.partition.image import partition_image
        from unstructured.documents.elements import Table

        elements = partition_image(
            file=io.BytesIO(page_image_bytes),
            strategy="hi_res",
        )
        for el in elements:
            if isinstance(el, Table):
                raw_html = getattr(el.metadata, "text_as_html", "") or ""
                md = _html_table_to_markdown(raw_html) if raw_html else str(el)
                tables.append(
                    {
                        "page": page_num,
                        "text": md,
                        "table_markdown": md,
                        "table_html": raw_html,
                    }
                )
    except ImportError:
        logger.debug(
            "unstructured[local-inference] not installed; table detection skipped."
        )
    except Exception as exc:
        logger.debug("Table detection via unstructured failed on page %d: %s", page_num, exc)

    return tables


def _extract_scanned_page(
    page: Any,
    page_num: int,
    output_dir: Path,
    dpi: int = 300,
    run_table_detection: bool = True,
    save_page_renders: bool = False,
) -> Tuple[str, List[float], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract all content from a single *scanned* PDF page.
    Renders page pixmap once for optimal performance.
    """
    import io
    from PIL import Image

    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
    page_image_bytes: bytes = pix.tobytes("png")
    pil_image = Image.open(io.BytesIO(page_image_bytes))

    page_img_path = None
    if save_page_renders:
        page_img_path = _save_image(page_image_bytes, output_dir / "page_renders")

    page_text, hocr = _ocr_pil_image(pil_image)

    if not page_text.strip():
        page_text = page.get_text("text") or ""

    hocr_sizes = _parse_hocr_font_sizes(hocr)

    tables: List[Dict[str, Any]] = []
    if run_table_detection:
        tables = _detect_tables_via_unstructured(page_image_bytes, page_num)

    images: List[Dict[str, Any]] = []
    try:
        doc = page.parent
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_path = _save_image(img_bytes, output_dir / "embedded_images")
                images.append({"page": page_num, "image_path": img_path})
            except Exception as exc:
                logger.debug(
                    "Could not extract image xref=%d on page %d: %s",
                    xref, page_num, exc,
                )
    except Exception as exc:
        logger.debug("Image listing skipped on page %d: %s", page_num, exc)

    if page_img_path:
        images.insert(0, {"page": page_num, "image_path": page_img_path, "is_page_render": True})

    return page_text, hocr_sizes, tables, images



def _link_captions(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    i = 0
    while i < len(elements):
        el = elements[i]

        if el["type"] == "Paragraph" and _CAPTION_RE.match(el["text"]):
            if result and result[-1]["type"] in ("Table", "Image"):
                result[-1].setdefault("metadata", {})["caption"] = el["text"]
                i += 1
                continue

            if i + 1 < len(elements) and elements[i + 1]["type"] in ("Table", "Image"):
                elements[i + 1].setdefault("metadata", {})["caption"] = el["text"]
                i += 1
                continue

        result.append(el)
        i += 1

    return result



class ScannedDocumentParser:

    def __init__(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        dpi: int = 300,
        run_table_detection: bool = True,
        save_page_renders: bool = False,
    ) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = (
                self.file_path.parent
                / "extracted_assets"
                / self.file_path.stem
            )

        self.dpi = dpi
        self.run_table_detection = run_table_detection
        self.save_page_renders = save_page_renders

    def _build_page_elements(
        self,
        page_text: str,
        page_num: int,
        hocr_sizes: List[float],
        tables: List[Dict[str, Any]],
        images: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        text_elements = _classify_scanned_elements(page_text, page_num, hocr_sizes)

        table_elements: List[Dict[str, Any]] = [
            {
                "text": t["text"],
                "type": "Table",
                "level": 4,
                "metadata": {
                    "table_markdown": t.get("table_markdown", ""),
                    "table_html": t.get("table_html", ""),
                    "page": t["page"],
                },
            }
            for t in tables
        ]

        image_elements: List[Dict[str, Any]] = [
            {
                "text": f"[Image: {Path(img['image_path']).name}]",
                "type": "Image",
                "level": 4,
                "metadata": {
                    "image_path": img["image_path"],
                    "page": img["page"],
                },
            }
            for img in images
            if not img.get("is_page_render", False)
        ]

        return text_elements + table_elements + image_elements

    def parse_to_tree(self) -> NaryTree:
        logger.info("OCR-parsing scanned document: %s", self.file_path)

        doc = pymupdf.open(str(self.file_path))
        num_pages = len(doc)
        all_elements: List[Dict[str, Any]] = []
        total_tables = 0
        total_images = 0

        for page_num in range(num_pages):
            page = doc[page_num]

            page_text, hocr_sizes, tables, images = _extract_scanned_page(
                page,
                page_num + 1,
                self.output_dir,
                dpi=self.dpi,
                run_table_detection=self.run_table_detection,
                save_page_renders=self.save_page_renders,
            )

            if len(page_text.strip()) < _MIN_OCR_CHARS:
                logger.debug("Page %d appears blank after OCR – skipping.", page_num + 1)
                image_elements: List[Dict[str, Any]] = [
                    {
                        "text": f"[Image: {Path(img['image_path']).name}]",
                        "type": "Image",
                        "level": 4,
                        "metadata": {
                            "image_path": img["image_path"],
                            "page": img["page"],
                        },
                    }
                    for img in images
                    if not img.get("is_page_render", False)
                ]
                all_elements.extend(image_elements)
                total_images += len(image_elements)
                continue

            page_elements = self._build_page_elements(
                page_text, page_num + 1, hocr_sizes, tables, images
            )
            all_elements.extend(page_elements)
            total_tables += len(tables)
            total_images += len([img for img in images if not img.get("is_page_render", False)])

        doc.close()

        logger.info(
            "Extracted %d text elements, %d tables, %d images across %d pages.",
            len(all_elements) - total_tables - total_images,
            total_tables,
            total_images,
            num_pages,
        )

        all_elements = _link_captions(all_elements)
        logger.info("Total elements for tree insertion: %d", len(all_elements))

        doc_tree = NaryTree(text=self.file_path.name, node_type="DocumentRoot")
        stack: List[Node] = [doc_tree.root]

        for el in all_elements:
            el_level: int = el["level"]

            while len(stack) > 1 and stack[-1].level >= el_level:
                stack.pop()

            parent_node = stack[-1]

            new_node = doc_tree.insert(
                text=el["text"],
                node_type=el["type"],
                level=el_level,
                parent=parent_node,
                metadata=el["metadata"] if el["metadata"] else None,
            )

            stack.append(new_node)

        return doc_tree

    def parse_and_save_json(self, json_output_path: Optional[str] = None) -> str:
        tree = self.parse_to_tree()

        if json_output_path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.output_dir / f"{self.file_path.stem}_tree.json"
        else:
            out_path = Path(json_output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text(tree.to_json(), encoding="utf-8")
        logger.info("Tree JSON saved → %s", out_path)

        chunks_path = out_path.parent / f"{self.file_path.stem}_chunks.json"
        chunk_document_tree(tree, doc_name=self.file_path.name, output_path=chunks_path)
        logger.info("Chunks JSON saved → %s", chunks_path)

        return str(out_path.resolve())


def run_scanned_parser(
    file_path: str,
    output_dir: Optional[str] = None,
    dpi: int = 300,
    run_table_detection: bool = True,
    save_page_renders: bool = False,
) -> NaryTree:
    parser = ScannedDocumentParser(
        file_path,
        output_dir=output_dir,
        dpi=dpi,
        run_table_detection=run_table_detection,
        save_page_renders=save_page_renders,
    )
    return parser.parse_to_tree()


if __name__ == "__main__":
    import sys

    sample = "data/scanned_sample.pdf"
    if not Path(sample).exists():
        print(f"[!] Place a scanned PDF at '{sample}' to run the smoke test.")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    print(f"[*] Processing: {sample}")
    parser = ScannedDocumentParser(sample, dpi=300)
    out = parser.parse_and_save_json()
    print(f"[✓] JSON written to: {out}")
