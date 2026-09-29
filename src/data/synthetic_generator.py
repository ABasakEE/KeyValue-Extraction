"""Synthetic Form Generator with AcroForm Layouts and Augraphy Degradation.

Generates programmatically synthesized form documents across diverse template families
with exact ground-truth word bounding boxes, entity labels (question, answer, header, other),
and key-value linking relations (question -> answer).

Includes advanced, irregular tabular structures:
- Dynamic column count shifts per row (e.g. 5-col items -> 1-col subheaders -> 2-col summary rows)
- Spanned cells (section dividers, multi-column subheaders, merged total rows)
- Multi-line cell wrapping (variable row heights where description takes 2-3 lines while numbers take 1)
- Sparse/blank cells (missing values, unassigned codes)
- Bordered grids vs. borderless whitespace-aligned tables
- Explicit tabular key-value linking (column header -> cell, summary key -> summary value)

Applies Augraphy degradation (paper texture, scan noise, lighting gradients, dirty rollers,
bleed-through) to produce realistic scanned form images.

Output format is 100% compatible with FUNSD raw dataset schema, allowing direct loading
via src.data.funsd_loader without modifications.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

# Attempt to import augraphy components
try:
    from augraphy import (
        AugraphyPipeline,
        BadPhotoCopy,
        BleedThrough,
        Brightness,
        DirtyRollers,
        LightingGradient,
        LowLightNoise,
        SubtleNoise,
    )
    HAVE_AUGRAPHY = True
except ImportError:
    HAVE_AUGRAPHY = False


# Canvas geometry (Letter / A4 at ~150-200 DPI)
PAGE_WIDTH = 1654
PAGE_HEIGHT = 2338


def get_font(size: int = 24, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a system TrueType font with graceful fallback to PIL default."""
    font_candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_augraphy_pipeline() -> Any:
    """Construct an Augraphy pipeline simulating physical scanner artifacts."""
    if not HAVE_AUGRAPHY:
        return None

    ink_phase = [
        LowLightNoise(num_photons_range=(50, 100), alpha_range=(0.8, 1.0), p=0.5),
        SubtleNoise(subtle_range=(5, 12), p=0.5),
    ]
    paper_phase = [
        Brightness(brightness_range=(0.92, 1.05), p=0.7),
        LightingGradient(
            light_position=None,
            direction=None,
            max_brightness=255,
            min_brightness=0,
            mode="linear",
            transparency=0.85,
            p=0.4,
        ),
        BleedThrough(alpha=0.08, p=0.3),
    ]
    post_phase = [
        DirtyRollers(line_width_range=(2, 6), p=0.3),
        BadPhotoCopy(noise_iteration=(1, 2), noise_size=(1, 2), noise_value=(0, 48), p=0.4),
    ]

    return AugraphyPipeline(ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase)


@dataclass
class FormField:
    key: str
    value: str
    layout: str = "horizontal"  # "horizontal", "stacked", "boxed"


@dataclass
class TableCell:
    text: str
    col_span: int = 1
    label: str = "answer"  # "header", "question", "answer", "other"
    is_key_for_next: bool = False  # If True, links to the next cell in this row as (key -> value)
    wrap: bool = False  # If True, text with \n wraps across multiple lines inside the cell


@dataclass
class TableRow:
    cells: list[TableCell]
    is_subheader: bool = False  # True if row is a section divider spanning columns
    is_summary: bool = False    # True if row is a summary / subtotal row
    bg_color: str | None = None  # Background fill (e.g. "#EEEEEE")


@dataclass
class FormTable:
    headers: list[str]
    col_widths_pct: list[float]  # Relative column width percentages, sum to 1.0
    rows: list[TableRow]
    border_style: str = "grid"  # "grid", "borderless", "horizontal_only"
    link_column_headers: bool = True  # If True, links column headers to data cells


@dataclass
class FormDocument:
    template_name: str
    header_title: str
    fields: list[FormField]
    table: FormTable | None = None
    table_headers: list[str] = field(default_factory=list)
    table_rows: list[list[str]] = field(default_factory=list)
    footer_text: str = ""


class SyntheticFormRenderer:
    """Renders structured synthetic form documents to images and FUNSD annotations."""

    def __init__(self, fake: Faker, use_augraphy: bool = True):
        self.fake = fake
        self.use_augraphy = use_augraphy and HAVE_AUGRAPHY
        self.pipeline = build_augraphy_pipeline() if self.use_augraphy else None

        self.font_title = get_font(size=36, bold=True)
        self.font_header = get_font(size=26, bold=True)
        self.font_key = get_font(size=22, bold=True)
        self.font_val = get_font(size=22, bold=False)
        self.font_small = get_font(size=18, bold=False)

    def _render_text_with_boxes(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        font: Any,
        fill: str = "#000000",
    ) -> tuple[list[dict], list[int], int, int]:
        """Draw text and record individual word bounding boxes and overall span box."""
        words = text.split()
        if not words:
            return [], [x, y, x, y], x, y

        word_records = []
        cur_x = x

        space_bbox = draw.textbbox((0, 0), " ", font=font)
        space_width = max(6, space_bbox[2] - space_bbox[0])

        span_x0, span_y0, span_x1, span_y1 = cur_x, y, cur_x, y

        for i, word in enumerate(words):
            w_bbox = draw.textbbox((cur_x, y), word, font=font)
            draw.text((cur_x, y), word, font=font, fill=fill)

            w_x0 = max(0, min(PAGE_WIDTH, int(w_bbox[0])))
            w_y0 = max(0, min(PAGE_HEIGHT, int(w_bbox[1])))
            w_x1 = max(0, min(PAGE_WIDTH, int(w_bbox[2])))
            w_y1 = max(0, min(PAGE_HEIGHT, int(w_bbox[3])))

            word_records.append({
                "text": word,
                "box": [w_x0, w_y0, w_x1, w_y1],
            })

            if i == 0:
                span_x0, span_y0, span_x1, span_y1 = w_x0, w_y0, w_x1, w_y1
            else:
                span_x0 = min(span_x0, w_x0)
                span_y0 = min(span_y0, w_y0)
                span_x1 = max(span_x1, w_x1)
                span_y1 = max(span_y1, w_y1)

            cur_x = w_x1 + space_width

        overall_box = [span_x0, span_y0, span_x1, span_y1]
        next_x = cur_x
        next_y = span_y1
        return word_records, overall_box, next_x, next_y

    def _render_table(
        self,
        draw: ImageDraw.ImageDraw,
        table: FormTable,
        cur_y: int,
        form_entities: list[dict],
        entity_id_counter: int,
    ) -> tuple[int, int]:
        """Render advanced tables with variable column counts, cell wrapping, and spans."""
        tbl_x0 = 100
        tbl_x1 = PAGE_WIDTH - 100
        tbl_w = tbl_x1 - tbl_x0
        n_cols = len(table.headers)

        if table.col_widths_pct and len(table.col_widths_pct) == n_cols:
            col_w = [int(p * tbl_w) for p in table.col_widths_pct]
        else:
            col_w = [tbl_w // max(1, n_cols)] * n_cols

        # Adjust last column width for rounding
        col_w[-1] = tbl_w - sum(col_w[:-1])
        col_x = [tbl_x0 + sum(col_w[:c]) for c in range(n_cols)]

        col_header_ids: dict[int, int] = {}

        # 1. Table Header Row
        if table.headers:
            header_h = 42
            if table.border_style == "grid":
                draw.rectangle([tbl_x0, cur_y, tbl_x1, cur_y + header_h], fill="#EDEDED", outline="#333333", width=2)
            elif table.border_style in ("horizontal_only", "borderless"):
                draw.line([(tbl_x0, cur_y), (tbl_x1, cur_y)], fill="#333333", width=2)
                draw.line([(tbl_x0, cur_y + header_h), (tbl_x1, cur_y + header_h)], fill="#333333", width=2)

            for c, h_text in enumerate(table.headers):
                cell_x = col_x[c] + 8
                h_words, h_box, _, _ = self._render_text_with_boxes(
                    draw, h_text, cell_x, cur_y + 8, self.font_key, fill="#111111"
                )
                h_id = entity_id_counter
                entity_id_counter += 1
                col_header_ids[c] = h_id

                form_entities.append({
                    "id": h_id,
                    "text": h_text,
                    "box": h_box,
                    "label": "header",
                    "words": h_words,
                    "linking": [],
                })

            cur_y += header_h

        # 2. Table Rows
        for r_idx, row in enumerate(table.rows):
            if cur_y >= PAGE_HEIGHT - 220:
                break

            # Calculate dynamic row height based on cell wrapping
            row_h = 36
            for cell in row.cells:
                if cell.wrap and "\n" in cell.text:
                    n_lines = len(cell.text.split("\n"))
                    row_h = max(row_h, n_lines * 28 + 12)

            # Draw background
            if row.bg_color:
                draw.rectangle([tbl_x0, cur_y, tbl_x1, cur_y + row_h], fill=row.bg_color)
            elif row.is_subheader:
                draw.rectangle([tbl_x0, cur_y, tbl_x1, cur_y + row_h], fill="#F4F4F4")
            elif table.border_style == "grid" and r_idx % 2 == 1:
                draw.rectangle([tbl_x0, cur_y, tbl_x1, cur_y + row_h], fill="#FAFAFA")

            # Draw grid borders
            if table.border_style == "grid":
                draw.rectangle([tbl_x0, cur_y, tbl_x1, cur_y + row_h], outline="#888888", width=1)
            elif table.border_style == "horizontal_only":
                draw.line([(tbl_x0, cur_y + row_h), (tbl_x1, cur_y + row_h)], fill="#CCCCCC", width=1)

            cur_col = 0
            prev_key_id: int | None = None

            for cell in row.cells:
                if cur_col >= n_cols:
                    break

                span = min(cell.col_span, n_cols - cur_col)
                cell_x = col_x[cur_col] + 8
                start_col = cur_col
                cur_col += span

                # Draw vertical column divider in grid mode
                if table.border_style == "grid" and cur_col < n_cols:
                    draw.line([(col_x[cur_col], cur_y), (col_x[cur_col], cur_y + row_h)], fill="#888888", width=1)

                cell_text = cell.text.strip()
                if not cell_text:
                    continue  # Sparse / blank cell

                this_id = entity_id_counter
                entity_id_counter += 1

                # Render text (wrapped or single line)
                if cell.wrap and "\n" in cell_text:
                    lines = cell_text.split("\n")
                    all_words = []
                    u_x0, u_y0, u_x1, u_y1 = None, None, None, None
                    line_y = cur_y + 6
                    for line_str in lines:
                        l_words, l_box, _, next_ly = self._render_text_with_boxes(
                            draw, line_str, cell_x, line_y, self.font_val, fill="#000000"
                        )
                        all_words.extend(l_words)
                        if u_x0 is None:
                            u_x0, u_y0, u_x1, u_y1 = l_box
                        else:
                            u_x0 = min(u_x0, l_box[0])
                            u_y0 = min(u_y0, l_box[1])
                            u_x1 = max(u_x1, l_box[2])
                            u_y1 = max(u_y1, l_box[3])
                        line_y = next_ly + 6
                    c_words = all_words
                    c_box = [u_x0 or cell_x, u_y0 or cur_y, u_x1 or cell_x, u_y1 or (cur_y + row_h)]
                else:
                    font = self.font_key if (row.is_subheader or cell.is_key_for_next) else self.font_val
                    fill = "#111111" if (row.is_subheader or cell.is_key_for_next) else "#000000"
                    c_words, c_box, _, _ = self._render_text_with_boxes(
                        draw, cell_text, cell_x, cur_y + 6, font, fill=fill
                    )

                # Determine entity label
                if row.is_subheader:
                    lbl = "header"
                elif cell.is_key_for_next:
                    lbl = "question"
                else:
                    lbl = cell.label

                ent: dict[str, Any] = {
                    "id": this_id,
                    "text": cell_text,
                    "box": c_box,
                    "label": lbl,
                    "words": c_words,
                    "linking": [],
                }

                # Key-value linking logic
                if cell.is_key_for_next:
                    prev_key_id = this_id
                elif prev_key_id is not None:
                    # Link previous key in row to this value (e.g. Subtotal -> $1,200)
                    ent["linking"].append([prev_key_id, this_id])
                    for f_ent in form_entities:
                        if f_ent["id"] == prev_key_id:
                            f_ent["linking"].append([prev_key_id, this_id])
                            break
                    prev_key_id = None
                elif table.link_column_headers and not row.is_subheader and not row.is_summary:
                    # Link column header to this data cell
                    if start_col in col_header_ids:
                        h_id = col_header_ids[start_col]
                        ent["linking"].append([h_id, this_id])
                        for f_ent in form_entities:
                            if f_ent["id"] == h_id:
                                f_ent["linking"].append([h_id, this_id])
                                break

                form_entities.append(ent)

            cur_y += row_h

        # Final bottom line for borderless / horizontal-only tables
        if table.border_style in ("horizontal_only", "borderless"):
            draw.line([(tbl_x0, cur_y), (tbl_x1, cur_y)], fill="#333333", width=2)

        cur_y += 45
        return cur_y, entity_id_counter

    def render(self, doc: FormDocument) -> tuple[Image.Image, dict]:
        """Render a FormDocument into a PIL Image and a FUNSD-compatible annotation dict."""
        canvas = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), color="#FFFFFF")
        draw = ImageDraw.Draw(canvas)

        form_entities: list[dict] = []
        entity_id_counter = 0

        # 1. Document Title / Header
        cur_y = 120
        title_bbox = draw.textbbox((0, 0), doc.header_title, font=self.font_title)
        title_w = title_bbox[2] - title_bbox[0]
        title_x = max(100, (PAGE_WIDTH - title_w) // 2)

        draw.line([(100, cur_y - 20), (PAGE_WIDTH - 100, cur_y - 20)], fill="#333333", width=3)

        w_records, span_box, _, next_y = self._render_text_with_boxes(
            draw, doc.header_title, title_x, cur_y, self.font_title
        )
        form_entities.append({
            "id": entity_id_counter,
            "text": doc.header_title,
            "box": span_box,
            "label": "header",
            "words": w_records,
            "linking": [],
        })
        entity_id_counter += 1

        cur_y = next_y + 40
        draw.line([(100, cur_y), (PAGE_WIDTH - 100, cur_y)], fill="#666666", width=2)
        cur_y += 60

        # 2. Key-Value Fields Rendering
        if doc.fields:
            use_two_columns = len(doc.fields) >= 8
            col_width = (PAGE_WIDTH - 260) // 2 if use_two_columns else (PAGE_WIDTH - 200)

            left_margin = 110
            right_col_x = left_margin + col_width + 40
            col_y = [cur_y, cur_y]

            for i, field_item in enumerate(doc.fields):
                col_idx = (i % 2) if use_two_columns else 0
                start_x = right_col_x if col_idx == 1 else left_margin
                y_pos = col_y[col_idx]

                key_text = field_item.key.strip()
                if not key_text.endswith(":"):
                    key_text += ":"
                val_text = field_item.value.strip()

                key_id = entity_id_counter
                val_id = entity_id_counter + 1
                entity_id_counter += 2

                if field_item.layout == "stacked":
                    k_words, k_box, _, next_k_y = self._render_text_with_boxes(
                        draw, key_text, start_x, y_pos, self.font_key, fill="#222222"
                    )
                    v_words, v_box, _, next_v_y = self._render_text_with_boxes(
                        draw, val_text, start_x + 10, next_k_y + 8, self.font_val, fill="#000000"
                    )
                    col_y[col_idx] = next_v_y + 35
                elif field_item.layout == "boxed":
                    k_words, k_box, next_k_x, next_k_y = self._render_text_with_boxes(
                        draw, key_text, start_x, y_pos, self.font_key, fill="#222222"
                    )
                    box_x0 = next_k_x + 15
                    box_y0 = y_pos - 4
                    box_w = max(220, col_width - (box_x0 - start_x))
                    box_h = 38
                    draw.rectangle(
                        [box_x0, box_y0, box_x0 + box_w, box_y0 + box_h],
                        outline="#555555",
                        width=2,
                    )
                    v_words, v_box, _, next_v_y = self._render_text_with_boxes(
                        draw, val_text, box_x0 + 10, box_y0 + 5, self.font_val, fill="#000000"
                    )
                    col_y[col_idx] = max(next_k_y, box_y0 + box_h) + 30
                else:
                    k_words, k_box, next_k_x, next_k_y = self._render_text_with_boxes(
                        draw, key_text, start_x, y_pos, self.font_key, fill="#222222"
                    )
                    val_start_x = max(next_k_x + 15, start_x + int(col_width * 0.42))
                    v_words, v_box, _, next_v_y = self._render_text_with_boxes(
                        draw, val_text, val_start_x, y_pos, self.font_val, fill="#000000"
                    )
                    col_y[col_idx] = max(next_k_y, next_v_y) + 32

                form_entities.append({
                    "id": key_id,
                    "text": key_text,
                    "box": k_box,
                    "label": "question",
                    "words": k_words,
                    "linking": [[key_id, val_id]],
                })
                form_entities.append({
                    "id": val_id,
                    "text": val_text,
                    "box": v_box,
                    "label": "answer",
                    "words": v_words,
                    "linking": [[key_id, val_id]],
                })

            cur_y = max(col_y[0], col_y[1]) + 40

        # 3. Tabular Section (Advanced FormTable or legacy table_headers/rows)
        if doc.table is not None:
            cur_y, entity_id_counter = self._render_table(
                draw, doc.table, cur_y, form_entities, entity_id_counter
            )
        elif doc.table_headers and doc.table_rows and cur_y < PAGE_HEIGHT - 450:
            # Wrap simple string rows into a FormTable
            t_rows = [
                TableRow([TableCell(c, col_span=1, label="answer") for c in r])
                for r in doc.table_rows
            ]
            f_table = FormTable(
                headers=doc.table_headers,
                col_widths_pct=[1.0 / len(doc.table_headers)] * len(doc.table_headers),
                rows=t_rows,
                border_style="grid",
                link_column_headers=True,
            )
            cur_y, entity_id_counter = self._render_table(
                draw, f_table, cur_y, form_entities, entity_id_counter
            )

        # 4. Footer / Other text
        if doc.footer_text:
            foot_y = min(PAGE_HEIGHT - 80, cur_y + 30)
            f_words, f_box, _, _ = self._render_text_with_boxes(
                draw, doc.footer_text, 100, foot_y, self.font_small, fill="#777777"
            )
            form_entities.append({
                "id": entity_id_counter,
                "text": doc.footer_text,
                "box": f_box,
                "label": "other",
                "words": f_words,
                "linking": [],
            })

        # Apply Augraphy physical scanning degradation
        final_image = canvas
        if self.use_augraphy and self.pipeline is not None:
            try:
                np_img = np.array(canvas)
                augmented = self.pipeline(np_img)
                final_image = Image.fromarray(augmented)
            except Exception:
                final_image = canvas

        annotation = {"form": form_entities}
        return final_image, annotation


# ---------------------------------------------------------------------------
# Template Generators
# ---------------------------------------------------------------------------

def generate_invoice_document(fake: Faker, template_idx: int) -> FormDocument:
    """Generate an Invoice with irregular table rows and summary spans."""
    layouts = ["horizontal", "stacked", "boxed"]
    layout = layouts[template_idx % len(layouts)]

    titles = ["TAX INVOICE", "COMMERCIAL INVOICE", "BILLING STATEMENT", "PAYMENT INVOICE"]
    title = f"{fake.company().upper()} - {titles[template_idx % len(titles)]}"

    fields = [
        FormField("Invoice Number", f"INV-{fake.numerify(text='#####')}", layout),
        FormField("Invoice Date", fake.date_this_year().strftime("%d-%b-%Y"), layout),
        FormField("Due Date", fake.date_this_year().strftime("%d-%b-%Y"), layout),
        FormField("Client Name", fake.name(), layout),
        FormField("Billing Address", fake.street_address(), layout),
        FormField("Payment Status", random.choice(["PENDING", "DUE UPON RECEIPT", "NET 30"]), layout),
    ]

    # Irregular 5-column table with multi-line description and spanned summary rows
    headers = ["Item", "Description", "Qty", "Rate", "Total"]
    col_widths = [0.10, 0.42, 0.12, 0.16, 0.20]

    rows = [
        TableRow([
            TableCell("1", col_span=1),
            TableCell("Technical Consulting Services\nCloud Architecture & Security Audit", col_span=1, wrap=True),
            TableCell("40", col_span=1),
            TableCell("$125.00", col_span=1),
            TableCell("$5,000.00", col_span=1),
        ]),
        TableRow([
            TableCell("2", col_span=1),
            TableCell("Dedicated Server Deployment & Setup", col_span=1),
            TableCell("2", col_span=1),
            TableCell("$750.00", col_span=1),
            TableCell("$1,500.00", col_span=1),
        ]),
        # Spanned Subheader Row (changing col count from 5 to 1)
        TableRow([
            TableCell("-- SOFTWARE SUBSCRIPTIONS & THIRD-PARTY LICENSES --", col_span=5, label="header")
        ], is_subheader=True),
        TableRow([
            TableCell("3", col_span=1),
            TableCell("Database Enterprise License (Annual)", col_span=1),
            TableCell("1", col_span=1),
            TableCell("$2,400.00", col_span=1),
            TableCell("$2,400.00", col_span=1),
        ]),
        # Spanned Summary Rows (changing col count from 5 to 2, with key-value link)
        TableRow([
            TableCell("Subtotal Amount:", col_span=4, is_key_for_next=True),
            TableCell("$8,900.00", col_span=1, label="answer"),
        ], is_summary=True),
        TableRow([
            TableCell("Tax Assessment (8.25%):", col_span=4, is_key_for_next=True),
            TableCell("$734.25", col_span=1, label="answer"),
        ], is_summary=True),
        TableRow([
            TableCell("Total Balance Due:", col_span=4, is_key_for_next=True),
            TableCell("$9,634.25", col_span=1, label="answer"),
        ], is_summary=True, bg_color="#EDEDED"),
    ]

    table = FormTable(headers=headers, col_widths_pct=col_widths, rows=rows, border_style="grid")
    footer = f"Thank you for your business. Remit payment to {fake.company()} - Routing: {fake.numerify(text='#########')}"
    return FormDocument(f"invoice_{template_idx}", title, fields, table=table, footer_text=footer)


def generate_itemized_financial_statement(fake: Faker, template_idx: int) -> FormDocument:
    """Generate a multi-tier financial operations statement with irregular subheaders and totals."""
    title = f"{fake.company().upper()} - CONSOLIDATED STATEMENT OF OPERATIONS"

    fields = [
        FormField("Fiscal Period", "Q3 Ended September 30, 2024", "horizontal"),
        FormField("Reporting Currency", "United States Dollars (USD)", "horizontal"),
        FormField("Audit Status", "Audited Financial Report", "horizontal"),
        FormField("Entity Identifier", f"CORP-{fake.numerify(text='######')}", "horizontal"),
    ]

    headers = ["Operating Account Category", "Account Code", "Prior Quarter", "Current Quarter"]
    col_widths = [0.46, 0.16, 0.19, 0.19]

    rev1 = random.randint(250, 450) * 1000
    rev2 = random.randint(150, 300) * 1000
    tot_rev = rev1 + rev2

    exp1 = random.randint(100, 220) * 1000
    exp2 = random.randint(50, 120) * 1000
    tot_exp = exp1 + exp2
    ebitda = tot_rev - tot_exp

    rows = [
        # Section 1 Subheader (span 4, col count = 1)
        TableRow([
            TableCell(">>> SECTION I: OPERATING REVENUES & GROSS RECEIPTS <<<", col_span=4, label="header")
        ], is_subheader=True),
        TableRow([
            TableCell("Enterprise Software Subscription Fees\nMulti-Year Recurring SaaS Agreements", col_span=1, wrap=True),
            TableCell("REV-101", col_span=1),
            TableCell(f"${rev1 - 25000:,.2f}", col_span=1),
            TableCell(f"${rev1:,.2f}", col_span=1),
        ]),
        TableRow([
            TableCell("Professional Consulting & Solution Delivery", col_span=1),
            TableCell("REV-204", col_span=1),
            TableCell(f"${rev2 - 15000:,.2f}", col_span=1),
            TableCell(f"${rev2:,.2f}", col_span=1),
        ]),
        # Spanned Summary (span 3 + 1, col count = 2)
        TableRow([
            TableCell("Total Operating Revenues (Gross):", col_span=3, is_key_for_next=True),
            TableCell(f"${tot_rev:,.2f}", col_span=1, label="answer"),
        ], is_summary=True, bg_color="#F2F2F2"),
        # Section 2 Subheader (span 4, col count = 1)
        TableRow([
            TableCell(">>> SECTION II: OPERATING & ADMINISTRATIVE EXPENDITURES <<<", col_span=4, label="header")
        ], is_subheader=True),
        TableRow([
            TableCell("Personnel Salaries, Wages & Executive Benefits", col_span=1),
            TableCell("EXP-301", col_span=1),
            TableCell(f"${exp1 - 10000:,.2f}", col_span=1),
            TableCell(f"${exp1:,.2f}", col_span=1),
        ]),
        TableRow([
            TableCell("Cloud Computing Infrastructure & Network Transit\nAmazon AWS / Google GCP Clusters", col_span=1, wrap=True),
            TableCell("EXP-412", col_span=1),
            TableCell(f"${exp2 - 8000:,.2f}", col_span=1),
            TableCell(f"${exp2:,.2f}", col_span=1),
        ]),
        # Spanned Summary (span 3 + 1, col count = 2)
        TableRow([
            TableCell("Total Administrative Expenditures:", col_span=3, is_key_for_next=True),
            TableCell(f"${tot_exp:,.2f}", col_span=1, label="answer"),
        ], is_summary=True, bg_color="#F2F2F2"),
        # Grand Total (span 3 + 1)
        TableRow([
            TableCell("Net Operating Profit (EBITDA):", col_span=3, is_key_for_next=True),
            TableCell(f"${ebitda:,.2f}", col_span=1, label="answer"),
        ], is_summary=True, bg_color="#E0E0E0"),
    ]

    table = FormTable(headers=headers, col_widths_pct=col_widths, rows=rows, border_style="grid")
    footer = "CERTIFIED FINANCIAL DISCLOSURE: Prepared in accordance with Generally Accepted Accounting Principles (US GAAP)."
    return FormDocument(f"fin_statement_{template_idx}", title, fields, table=table, footer_text=footer)


def generate_procurement_order_with_wrapped_table(fake: Faker, template_idx: int) -> FormDocument:
    """Generate a Purchase Order with multi-line wrapped descriptions and sparse cells."""
    title = f"{fake.company().upper()} - PURCHASE ORDER REQUISITION"

    fields = [
        FormField("Purchase Order No", f"PO-{fake.numerify(text='#####')}", "boxed"),
        FormField("Requisitioner Name", fake.name(), "boxed"),
        FormField("Vendor Name", f"{fake.company()} Technologies", "boxed"),
        FormField("Authorized Department", "Information Technology & Infrastructure", "boxed"),
        FormField("Delivery Date", fake.date_this_month().strftime("%d-%b-%Y"), "boxed"),
        FormField("Payment Terms", "Net 45 Days", "boxed"),
    ]

    headers = ["Item", "Item Specifications & Description", "Qty", "Unit Price", "Extended Total"]
    col_widths = [0.08, 0.44, 0.10, 0.18, 0.20]

    rows = [
        TableRow([
            TableCell("1", col_span=1),
            TableCell("Dell PowerEdge R750 Rack Server\nDual Intel Xeon Gold 6330 CPUs\n128GB DDR4 RAM, 2x 960GB NVMe SSD", col_span=1, wrap=True),
            TableCell("2", col_span=1),
            TableCell("$4,250.00", col_span=1),
            TableCell("$8,500.00", col_span=1),
        ]),
        TableRow([
            TableCell("2", col_span=1),
            TableCell("Cisco Catalyst 9300 48-Port Switch\nNetwork Advantage Software Licensing Tier", col_span=1, wrap=True),
            TableCell("1", col_span=1),
            TableCell("$3,100.00", col_span=1),
            TableCell("$3,100.00", col_span=1),
        ]),
        # Subheader row (span 5, col count = 1)
        TableRow([
            TableCell("-- PROFESSIONAL LOGISTICS & DEPLOYMENT CHARGES --", col_span=5, label="header")
        ], is_subheader=True),
        # Sparse row (Unit Price is blank)
        TableRow([
            TableCell("3", col_span=1),
            TableCell("On-site Data Center Rack Mounting & Cabling", col_span=1),
            TableCell("1", col_span=1),
            TableCell("", col_span=1),  # Sparse blank cell
            TableCell("$1,200.00", col_span=1),
        ]),
        # Spanned Summary Rows (span 4 + 1, col count = 2)
        TableRow([
            TableCell("Requisition Subtotal:", col_span=4, is_key_for_next=True),
            TableCell("$12,800.00", col_span=1, label="answer"),
        ], is_summary=True),
        TableRow([
            TableCell("Freight & Express Handling:", col_span=4, is_key_for_next=True),
            TableCell("$450.00", col_span=1, label="answer"),
        ], is_summary=True),
        TableRow([
            TableCell("Authorized Requisition Total:", col_span=4, is_key_for_next=True),
            TableCell("$13,250.00", col_span=1, label="answer"),
        ], is_summary=True, bg_color="#E8E8E8"),
    ]

    table = FormTable(headers=headers, col_widths_pct=col_widths, rows=rows, border_style="grid")
    footer = "OFFICIAL PROCUREMENT RECORD: All deliveries must reference the PO Number on outer packaging."
    return FormDocument(f"procure_{template_idx}", title, fields, table=table, footer_text=footer)


def generate_shipping_manifest(fake: Faker, template_idx: int) -> FormDocument:
    """Generate an international shipping manifest with borderless whitespace-separated columns."""
    carrier = f"{fake.company().upper()} GLOBAL LOGISTICS"
    title = f"{carrier} - CARGO CONSIGNMENT & WAYBILL MANIFEST"

    fields = [
        FormField("Waybill Number", f"WB-{fake.numerify(text='#######')}", "horizontal"),
        FormField("Carrier Vessel", f"M/V {fake.first_name().upper()} EXPLORER", "horizontal"),
        FormField("Port of Loading", f"{fake.city()} Seaport", "horizontal"),
        FormField("Port of Discharge", "Port of Rotterdam, Netherlands", "horizontal"),
        FormField("Customs Broker", f"{fake.last_name()} Clearance Agency LLC", "horizontal"),
        FormField("Departure Date", fake.date_this_year().strftime("%d-%b-%Y"), "horizontal"),
    ]

    headers = ["Consignment ID", "Commodity Description & Marks", "Packages", "Gross Weight", "Declared Value"]
    col_widths = [0.18, 0.40, 0.12, 0.15, 0.15]

    rows = [
        # Subheader row (span 5, col count = 1)
        TableRow([
            TableCell(">> CONTAINER ID: MSCU-9941820 / 40FT TEMPERATURE CONTROLLED <<", col_span=5, label="header")
        ], is_subheader=True),
        TableRow([
            TableCell("CON-8801", col_span=1),
            TableCell("High Precision Optical Sensors\nFragile Laboratory Glass Assemblies", col_span=1, wrap=True),
            TableCell("120 Cartons", col_span=1),
            TableCell("420 KG", col_span=1),
            TableCell("$38,500.00", col_span=1),
        ]),
        TableRow([
            TableCell("CON-8802", col_span=1),
            TableCell("Industrial Electric Servomotors", col_span=1),
            TableCell("45 Crates", col_span=1),
            TableCell("850 KG", col_span=1),
            TableCell("$21,000.00", col_span=1),
        ]),
        # Subheader row (span 5, col count = 1)
        TableRow([
            TableCell(">> CONTAINER ID: TGHU-3301984 / 20FT STANDARD DRY VAN <<", col_span=5, label="header")
        ], is_subheader=True),
        # Sparse row (Declared Value is blank)
        TableRow([
            TableCell("CON-9914", col_span=1),
            TableCell("Marketing Brochures & Documentation", col_span=1),
            TableCell("30 Boxes", col_span=1),
            TableCell("110 KG", col_span=1),
            TableCell("", col_span=1),  # Sparse blank cell
        ]),
        # Spanned Summary Rows (span 3 + 2, and span 4 + 1)
        TableRow([
            TableCell("Total Consignment Pieces & Weight:", col_span=3, is_key_for_next=True),
            TableCell("195 Packages / 1,380 KG", col_span=2, label="answer"),
        ], is_summary=True),
        TableRow([
            TableCell("Total Declared Customs Value:", col_span=4, is_key_for_next=True),
            TableCell("$59,500.00 USD", col_span=1, label="answer"),
        ], is_summary=True),
    ]

    # Borderless layout with horizontal separator lines
    table = FormTable(headers=headers, col_widths_pct=col_widths, rows=rows, border_style="horizontal_only")
    footer = "INTERNATIONAL CARGO DECLARATION: Certified compliant with International Maritime Organization (IMO) conventions."
    return FormDocument(f"manifest_{template_idx}", title, fields, table=table, footer_text=footer)


def generate_medical_intake_document(fake: Faker, template_idx: int) -> FormDocument:
    """Generate a Patient Medical Intake & Insurance Registration document."""
    layouts = ["stacked", "horizontal", "boxed"]
    layout = layouts[template_idx % len(layouts)]

    title = f"{fake.city().upper()} REGIONAL HOSPITAL - PATIENT INTAKE FORM"

    fields = [
        FormField("Patient Legal Name", fake.name(), layout),
        FormField("Date of Birth", fake.date_of_birth(minimum_age=18, maximum_age=85).strftime("%m/%d/%Y"), layout),
        FormField("Gender Identity", random.choice(["Male", "Female", "Non-Binary", "Other"]), layout),
        FormField("Social Security No", fake.ssn(), layout),
        FormField("Residential Address", fake.street_address(), layout),
        FormField("Contact Phone", fake.phone_number()[:14], layout),
        FormField("Emergency Contact", fake.name(), layout),
        FormField("Emergency Relation", random.choice(["Spouse", "Parent", "Sibling", "Guardian"]), layout),
        FormField("Insurance Carrier", f"{fake.company()} Health Plan", layout),
        FormField("Policy Member ID", f"MEM-{fake.numerify(text='########')}", layout),
        FormField("Group Number", f"GRP-{fake.numerify(text='#####')}", layout),
        FormField("Primary Physician", f"Dr. {fake.last_name()}, MD", layout),
    ]

    footer = "CONFIDENTIAL MEDICAL RECORD: Protected under Health Insurance Portability and Accountability Act (HIPAA)."
    return FormDocument(f"intake_{template_idx}", title, fields, footer_text=footer)


def generate_adbuy_contract_document(fake: Faker, template_idx: int) -> FormDocument:
    """Generate an FCC Broadcast Political Advertising Contract Form (VRDU DeepForm style)."""
    layouts = ["boxed", "horizontal", "stacked"]
    layout = layouts[template_idx % len(layouts)]

    station = f"W{fake.lexify(text='???').upper()}-TV"
    title = f"{station} BROADCAST ADVERTISING ORDER CONTRACT"

    fields = [
        FormField("Contract Number", fake.numerify(text='######'), layout),
        FormField("Advertiser Name", fake.company(), layout),
        FormField("Political Candidate", fake.name(), layout),
        FormField("Election Office", random.choice(["US Senate", "State Governor", "Congressional District 04"]), layout),
        FormField("Agency Representation", f"{fake.last_name()} Media Partners LLC", layout),
        FormField("Flight Start Date", fake.date_this_month().strftime("%m/%d/%Y"), layout),
        FormField("Flight End Date", fake.date_this_year().strftime("%m/%d/%Y"), layout),
        FormField("Broadcast Station", station, layout),
        FormField("DMA Market Code", fake.city(), layout),
        FormField("Gross Contract Total", f"${fake.numerify(text='#####.00')}", layout),
        FormField("Agency Commission", "15.00%", layout),
        FormField("Net Payable Amount", f"${fake.numerify(text='#####.00')}", layout),
    ]

    table_headers = ["Program", "Daypart", "Spots", "Rate", "Subtotal"]
    table_rows = [
        ["Evening News 6PM", "M-F 18:00", "5", "$1,200.00", "$6,000.00"],
        ["Late Night Edition", "M-F 23:30", "4", "$850.00", "$3,400.00"],
        ["Sunday Morning Show", "Sun 09:00", "2", "$2,100.00", "$4,200.00"],
    ]

    footer = "FCC PUBLIC INSPECTION FILE: Certified compliance under Communications Act Section 315/312."
    return FormDocument(f"adbuy_{template_idx}", title, fields, table_headers=table_headers, table_rows=table_rows, footer_text=footer)


def generate_fara_registration_document(fake: Faker, template_idx: int) -> FormDocument:
    """Generate a Foreign Agents Registration Act Disclosure Statement (VRDU FARA style)."""
    layouts = ["horizontal", "boxed", "stacked"]
    layout = layouts[template_idx % len(layouts)]

    title = "UNITED STATES DEPARTMENT OF JUSTICE - FARA REGISTRATION STATEMENT"

    fields = [
        FormField("Registration Number", fake.numerify(text='####'), layout),
        FormField("Registrant Legal Entity", fake.company(), layout),
        FormField("Principal Office Address", fake.street_address(), layout),
        FormField("Authorized Official", fake.name(), layout),
        FormField("Official Title", random.choice(["Managing Partner", "Senior Counsel", "Executive Director"]), layout),
        FormField("Foreign Principal Name", f"Government of {fake.country()}", layout),
        FormField("Foreign Principal Address", fake.city(), layout),
        FormField("Nature of Services", random.choice(["Public Relations Representation", "Government Affairs Counseling", "Economic Trade Promotion"]), layout),
        FormField("Contract Execution Date", fake.date_this_year().strftime("%B %d, %Y"), layout),
        FormField("Term Duration", "12 Calendar Months", layout),
        FormField("Retainer Compensation", f"${fake.numerify(text='######.00')} USD", layout),
        FormField("Filing Year", "2024", layout),
    ]

    footer = "Pursuant to Foreign Agents Registration Act of 1938, as amended (22 U.S.C. § 611 et seq.)."
    return FormDocument(f"fara_{template_idx}", title, fields, footer_text=footer)


GENERATOR_REGISTRY = [
    generate_invoice_document,
    generate_itemized_financial_statement,
    generate_procurement_order_with_wrapped_table,
    generate_shipping_manifest,
    generate_medical_intake_document,
    generate_adbuy_contract_document,
    generate_fara_registration_document,
]


def generate_synthetic_dataset(
    output_dir: str | Path,
    n_train: int = 100,
    n_test: int = 30,
    seed: int = 42,
    use_augraphy: bool = True,
) -> dict:
    """Generate complete synthetic dataset split into training_data and testing_data."""
    output_path = Path(output_dir)
    random.seed(seed)
    np.random.seed(seed)
    fake = Faker()
    fake.seed_instance(seed)

    renderer = SyntheticFormRenderer(fake=fake, use_augraphy=use_augraphy)

    summary: dict[str, Any] = {"train_count": 0, "test_count": 0, "templates": {}}

    for split_name, count in [("training_data", n_train), ("testing_data", n_test)]:
        img_dir = output_path / split_name / "images"
        ann_dir = output_path / split_name / "annotations"
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        for doc_id in range(count):
            gen_func = GENERATOR_REGISTRY[doc_id % len(GENERATOR_REGISTRY)]
            template_variant = doc_id // len(GENERATOR_REGISTRY)
            doc = gen_func(fake, template_variant)

            template_key = doc.template_name
            summary["templates"][template_key] = summary["templates"].get(template_key, 0) + 1

            img, ann = renderer.render(doc)

            file_stem = f"syn_{doc.template_name}_{doc_id:04d}"
            img_file = img_dir / f"{file_stem}.png"
            ann_file = ann_dir / f"{file_stem}.json"

            img.save(img_file, format="PNG")
            ann_file.write_text(json.dumps(ann, indent=2), encoding="utf-8")

        summary[f"{split_name}_generated"] = count

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="data/synthetic/dataset", help="Output root directory")
    parser.add_argument("--n_train", type=int, default=100, help="Number of training documents")
    parser.add_argument("--n_test", type=int, default=30, help="Number of testing documents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no_augraphy", action="store_true", help="Disable Augraphy physical degradation")
    args = parser.parse_args()

    print(f"Generating synthetic form dataset into {args.output_dir}...")
    summary = generate_synthetic_dataset(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        use_augraphy=not args.no_augraphy,
    )
    print(f"Dataset generated successfully:\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
