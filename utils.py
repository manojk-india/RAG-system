from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import HTMLConverter



def extract_text_from_pdf(pdf_path: str, pages: list[int] | None = None) -> str:
        result = ""
        with open(pdf_path, "rb") as f:
            if pages is None:
                for page_num, page in enumerate(PDFPage.get_pages(f), start=1):
                    page_text = extract_text(pdf_path, page_numbers=[page_num - 1])
                    result += f"\n\n### Page {page_num}\n\n" + page_text.strip()
            else:
                for page_num, page in enumerate(PDFPage.get_pages(f), start=1):
                    if page_num in pages:
                        page_text = extract_text(pdf_path, page_numbers=[page_num - 1])
                        result += f"\n\n### Page {page_num}\n\n" + page_text.strip()

        return result
