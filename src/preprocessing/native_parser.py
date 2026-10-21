from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf 

from unstructured.partition.text import partition_text
from unstructured.documents.elements import (
    Title,
    NarrativeText,
    ListItem,
    Header,
    Footer,
)

from src.preprocessing.tree_buillder import NaryTree, Node
from src.preprocessing.chunker import chunk_document_tree

logger = logging.getLogger(__name__)

#match table captions
_CAPTION_RE = re.compile(
    r"^\s*(Table|Figure|Fig\.?|Appendix)\s+[A-Z0-9]+\s*[:\.\-–]",
    re.IGNORECASE,
)

# Structural heading keywords
_HEADING_KEYWORDS_RE = re.compile(
    r"^(abstract|introduction|conclusion|concluding remarks|results|discussion|references|related work|methodology|experimental setup)\b",
    re.IGNORECASE,
)

# Numbered or Roman-numeral section headings (case-sensitive on prefix)
_HEADING_SECTIONS_RE = re.compile(
    r"^([IVXLCDM]+\.?\s+[A-Z]|[A-Z]\.\s+[A-Z]|\d+(\.\d+)*\.?\s+[A-Z])",
)

# Text that should NEVER be treated as a section heading (emails, affiliations, arxiv tags, urls)
_NON_HEADING_RE = re.compile(
    r"(@|\bemail\b|^arxiv:\d+\.\d+|https?://|^\s*(fig(\.|ure)?|table)\s+\d+)",
    re.IGNORECASE,
)

_AFFILIATION_RE = re.compile(
    r"\b(university|institute|department|laboratory|lab|college|school|faculty)\b",
    re.IGNORECASE,
)



def _html_table_to_markdown(html: str) -> str:
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


def _save_image(image_data: bytes, output_dir: Path) -> str:
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


def _compute_body_font_size(doc: Any) -> float:
    size_counter: Counter = Counter()
    for page in doc:
        try:
            raw_dict = page.get_text("dict")
            for block in raw_dict.get("blocks", []):
                if block.get("type") != 0:   # 0 = text block
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sz = round(span.get("size", 0), 1)
                        text = span.get("text", "")
                        if sz > 0 and text.strip():
                            size_counter[sz] += len(text)
        except Exception:
            pass

    if size_counter:
        return size_counter.most_common(1)[0][0]
    return 10.0


def _build_heading_level_map(doc: Any, body_font_size: float) -> Dict[float, int]:
    """
    Build a ``{font_size: heading_level}`` mapping.

    Only font sizes **noticeably larger** than *body_font_size* (> 0.5 pt) are
    treated as heading candidates.  They are ranked in descending order:
    largest → level 1, next → level 2, next → level 3 (cap).

    Returns an empty dict if no heading sizes are detected.
    """
    heading_sizes: set = set()
    for page in doc:
        try:
            raw_dict = page.get_text("dict")
            for block in raw_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sz = round(span.get("size", 0), 1)
                        if sz > body_font_size + 0.5 and span.get("text", "").strip():
                            heading_sizes.add(sz)
        except Exception:
            pass

    sorted_sizes = sorted(heading_sizes, reverse=True)
    return {sz: min(i + 1, 3) for i, sz in enumerate(sorted_sizes)}


def _pymupdf_extract_page(
    page: Any,
    page_num: int,
    output_dir: Path,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    text_blocks: List[Dict[str, Any]] = []
    page_text_parts: List[str] = []

    try:
        raw_dict = page.get_text("dict")
        for block in raw_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            block_parts: List[str] = []
            block_sizes: List[float] = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if span_text:
                        block_parts.append(span_text)
                        block_sizes.append(span.get("size", 0.0))

            if block_parts:
                block_text = " ".join(block_parts)
                avg_size = sum(block_sizes) / len(block_sizes) if block_sizes else 0.0
                block_bbox = [round(c, 2) for c in block.get("bbox", ())] if block.get("bbox") else None
                text_blocks.append(
                    {
                        "text": block_text,
                        "font_size": round(avg_size, 1),
                        "page": page_num,
                        "bbox": block_bbox,
                    }
                )
                page_text_parts.append(block_text)
    except Exception as exc:
        logger.debug("Font-size extraction failed on page %d: %s", page_num, exc)
        fallback = page.get_text("text") or ""
        page_text_parts = [fallback]

    page_text = "\n".join(page_text_parts)

    tables: List[Dict[str, Any]] = []
    try:
        table_finder = page.find_tables()
        for tbl in table_finder.tables:
            html = (
                tbl.to_pandas().to_html(index=False, border=0)
                if hasattr(tbl, "to_pandas")
                else ""
            )
            markdown = _html_table_to_markdown(html) if html else str(tbl.extract())
            tbl_bbox = [round(c, 2) for c in tbl.bbox] if hasattr(tbl, "bbox") and tbl.bbox else None
            tables.append(
                {
                    "page": page_num,
                    "text": markdown,
                    "table_markdown": markdown,
                    "table_html": html,
                    "bbox": tbl_bbox,
                }
            )
    except Exception as exc:
        logger.debug("Table extraction skipped on page %d: %s", page_num, exc)

    images: List[Dict[str, Any]] = []
    try:
        doc = page.parent
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_path = _save_image(img_bytes, output_dir)
                img_rects = page.get_image_rects(xref)
                img_bbox = [round(c, 2) for c in img_rects[0]] if img_rects else None
                images.append({"page": page_num, "image_path": img_path, "bbox": img_bbox})
            except Exception as exc:
                logger.debug(
                    "Could not extract image xref=%d on page %d: %s",
                    xref, page_num, exc,
                )
    except Exception as exc:
        logger.debug("Image listing skipped on page %d: %s", page_num, exc)

    return page_text, text_blocks, tables, images


def _link_captions(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    i = 0
    while i < len(elements):
        el = elements[i]

        if el["type"] == "Paragraph" and _CAPTION_RE.match(el["text"]):
            if result and result[-1]["type"] in ("Table", "Image"):
                result[-1]["metadata"]["caption"] = el["text"]
                i += 1
                continue

            if i + 1 < len(elements) and elements[i + 1]["type"] in ("Table", "Image"):
                elements[i + 1]["metadata"]["caption"] = el["text"]
                i += 1
                continue

        result.append(el)
        i += 1

    return result


class NativeDocumentParser:
    def __init__(self, file_path: str, output_dir: Optional[str] = None) -> None:
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

    def _classify_page_elements(
        self,
        page_text: str,
        page_num: int,
        text_blocks: List[Dict[str, Any]],
        body_font_size: float,
        heading_level_map: Dict[float, int],
    ) -> List[Dict[str, Any]]:
        if not page_text.strip():
            return []

        block_info_map: Dict[str, Dict[str, Any]] = {
            b["text"][:60]: {
                "font_size": b["font_size"],
                "bbox": b.get("bbox"),
            }
            for b in text_blocks
        }

        processed: List[Dict[str, Any]] = []
        raw_elements = partition_text(text=page_text)

        for el in raw_elements:
            if isinstance(el, (Header, Footer)):
                continue

            text_content = str(el).strip()
            if not text_content:
                continue

            matched_size: Optional[float] = None
            matched_bbox: Optional[List[float]] = None
            prefix = text_content[:60]
            for block_prefix, b_info in block_info_map.items():
                if prefix.startswith(block_prefix[:30]) or block_prefix.startswith(prefix[:30]):
                    matched_size = b_info["font_size"]
                    matched_bbox = b_info.get("bbox")
                    break

            metadata: Dict[str, Any] = {
                "page": page_num,
                "char_count": len(text_content),
            }
            if matched_bbox:
                metadata["bbox"] = matched_bbox

            if isinstance(el, Title):
                clean_t = text_content.strip()
                is_excluded = bool(_NON_HEADING_RE.search(clean_t) or _AFFILIATION_RE.search(clean_t))
                is_large_font = matched_size is not None and matched_size in heading_level_map
                is_tiny_font = matched_size is not None and matched_size < (body_font_size - 1.0)
                matches_heading_pattern = bool(_HEADING_KEYWORDS_RE.match(clean_t) or _HEADING_SECTIONS_RE.match(clean_t))

                # Must not be excluded, must not be tiny font (diagram/subscript), and must either be large font or section pattern
                if not is_excluded and not is_tiny_font and (is_large_font or matches_heading_pattern):
                    if is_large_font:
                        level = heading_level_map[matched_size]
                    elif matches_heading_pattern:
                        # Major section (e.g. "Abstract", "II. ...") gets level 2; subsections get level 3
                        if _HEADING_KEYWORDS_RE.match(clean_t) or re.match(r"^[IVXLCDM]+\.", clean_t):
                            level = 2
                        else:
                            level = 3
                    else:
                        level = 2
                    node_type = "Title"
                else:
                    # Demote false positive titles to Paragraph so they do not break the tree hierarchy
                    node_type = "Paragraph"
                    level = 4

            elif isinstance(el, NarrativeText):
                node_type = "Paragraph"
                level = 4

            elif isinstance(el, ListItem):
                node_type = "ListItem"
                level = 5

            else:
                node_type = "Paragraph"
                level = 4

            processed.append(
                {
                    "text": text_content,
                    "type": node_type,
                    "level": level,
                    "metadata": metadata,
                }
            )

        return processed

    def parse_to_tree(self) -> NaryTree:
        logger.info("Parsing document: %s", self.file_path)

        doc = pymupdf.open(str(self.file_path))

        body_font_size = _compute_body_font_size(doc)
        heading_level_map = _build_heading_level_map(doc, body_font_size)
        logger.info(
            "Body font size: %.1f pt | Heading level map: %s",
            body_font_size, heading_level_map,
        )

        all_elements: List[Dict[str, Any]] = []
        total_tables = 0
        total_images = 0
        num_pages = len(doc)

        for page_num in range(num_pages):
            page = doc[page_num]

            page_text, text_blocks, tables, images = _pymupdf_extract_page(
                page,
                page_num + 1,
                self.output_dir,
            )

            text_elements = self._classify_page_elements(
                page_text,
                page_num + 1,
                text_blocks,
                body_font_size,
                heading_level_map,
            )

            table_elements: List[Dict[str, Any]] = [
                {
                    "text": t["text"],
                    "type": "Table",
                    "level": 4,
                    "metadata": {
                        "table_markdown": t.get("table_markdown", ""),
                        "table_html": t.get("table_html", ""),
                        "page": t["page"],
                        "bbox": t.get("bbox"),
                        "char_count": len(t["text"]),
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
                        "bbox": img.get("bbox"),
                    },
                }
                for img in images
            ]

            all_elements.extend(text_elements + table_elements + image_elements)
            total_tables += len(tables)
            total_images += len(images)

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



def run_native_parser(file_path: str, output_dir: Optional[str] = None) -> NaryTree:
    parser = NativeDocumentParser(file_path, output_dir=output_dir)
    return parser.parse_to_tree()


if __name__ == "__main__":
    import sys

    sample = "data/sub1/doc_8_2404.18884v2.pdf"
    if not Path(sample).exists():
        print(f"[!] Place a native PDF at '{sample}' to run the smoke test.")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    print(f"[*] Processing: {sample}")
    parser = NativeDocumentParser(sample)
    out = parser.parse_and_save_json()
    print(f"[✓] JSON written to: {out}")