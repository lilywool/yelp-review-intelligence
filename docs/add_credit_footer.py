"""One-off script: stamps a credit-line footer onto every page of the PDF.

Run:
    .venv/Scripts/python.exe docs/add_credit_footer.py
"""

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

PDF_PATH = Path(__file__).resolve().parent / "Yelp-Review-Intelligence-NLP-Pipeline-v3.pdf"
CREDIT_TEXT = "Lillian Wool, MSBA"


def make_overlay(width: float, height: float) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=(width, height))
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor("#6f6a64"))
    c.drawRightString(width - 18, 12, CREDIT_TEXT)
    c.save()
    buffer.seek(0)
    return buffer.read()


def main() -> None:
    reader = PdfReader(PDF_PATH)
    writer = PdfWriter()

    for page in reader.pages:
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        overlay_reader = PdfReader(BytesIO(make_overlay(width, height)))
        page.merge_page(overlay_reader.pages[0])
        writer.add_page(page)

    with open(PDF_PATH, "wb") as f:
        writer.write(f)
    print(f"Stamped {len(reader.pages)} pages with credit footer: {PDF_PATH}")


if __name__ == "__main__":
    main()
