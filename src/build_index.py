import re
from pathlib import Path
import fitz
import easyocr
import numpy as np

class PdfScanner:
    """

    """
    def __init__(self, folder):
        """

        :param path:
        """
        self.folder = Path(folder).resolve()

    def scan(self):
        """

        :param folder:
        :return:
        """
        paths = []

        for path in self.folder.rglob("*.pdf"):
            if path.name.startswith("~") or path.name.startswith("."):
                continue

            paths.append(str(path))
        return paths

# pdf = PdfScanner("../data")
# print(pdf.scan())


class PdfParser:
    """

    """
    def __init__(self, use_ocr=True):
        """

        :param use_ocr:
        """
        self.use_ocr = use_ocr
        self.reader = None
        if self.use_ocr:
            self.reader = easyocr.Reader(["ru", "en"])

    def parse_file(self, file_path):
        """

        :param file_path:
        :return:
        """
        content = ""
        meta = {}

        with fitz.open(file_path) as document:
            meta = document.metadata
            for page_num, page in enumerate(document):
                page_text = page.get_text().strip()
                if page_text.strip():
                    content += f"--- Страница {page_num + 1} ---\n{page_text}\n\n"
        if len(content.strip()) > 50:
            return {
                "text": content,
                "meta": meta,
                "source": "text_layer"
            }

        if self.use_ocr:
            return {
                "text": self._parse_with_ocr(file_path),
                "meta": meta,
                "source": "ocr"
            }

        return {"text": content, "meta": meta, "source": "text_layer"}

    def _parse_with_ocr(self, file_path):
        """

        :param file_path:
        :return:
        """
        content = ""
        with fitz.open(file_path) as document:
            meta = document.metadata
            for page_num in range(len(document)):
                page = document.load_page(page_num)

                pix = page.get_pixmap(dpi=300)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

                result = self.reader.readtext(img, detail=0)
                page_text = " ".join(result)

                content += f"--- Страница {page_num + 1} (OCR) ---\n{page_text}\n\n"

            return content

parser = PdfParser()
print(parser.parse_file(r"C:\Users\user\Desktop\LLMApplication\data\52.pdf"))

class TextChunker:
    """

    """
    def __init__(self, size=800, overlap=150):
        """

        :param size:
        :param overlap:
        """
        self.size = size
        self.overlap = overlap

    def split(self, text):
        """

        :param text:
        :return:
        """
        text = re.sub(r"[ \t]+", " ", text).strip()

        paragraphs = [p for p in text.split("\n") if p.strip()]

        chunks = []
        current_chunk = ""

        for p in paragraphs:
            if len(p) > self.size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = current_chunk[-self.overlap:]

                start = 0
                while start < len(p):
                    space = self.size - len(current_chunk) - 1
                    space = max(1, space)

                    part = p[start:start+space]

                    if current_chunk:
                        current_chunk = current_chunk + " " + part
                    else:
                        current_chunk = part

                    if start + space < len(p):
                        chunks.append(current_chunk.strip())
                        current_chunk = current_chunk[-self.overlap:]
                        start += len(part)
                    else:
                        start = len(p)

                continue

            if len(current_chunk) + len(p) > self.size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = current_chunk[-self.overlap:] + " " + p
            else:
                if current_chunk:
                    current_chunk = current_chunk + " " + p
                else:
                    current_chunk = p

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks



text = """
Первый короткий абзац. Он точно влезет в лимит.

Второй абзац специально сделан подлиннее, чтобы проверить, как код отработает превышение лимита в восемьсот символов или сколько мы там задали в настройках нашего класса TextChunker.

Третий абзац для финала.
"""

chunker = TextChunker(size=100, overlap=20)

print(chunker.split(text))
