import re
from pathlib import Path
import fitz
import easyocr
import numpy as np
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.models import PointStruct
from typing import *
import hashlib
import time
from vector_store import VectorStore

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
            ocr_text = self._parse_with_ocr(file_path)
            meta = self._extract_meta_from_text(ocr_text, meta)
            return {
                "text": ocr_text,
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
            for page_num in range(len(document)):
                page = document.load_page(page_num)

                pix = page.get_pixmap(dpi=300)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

                result = self.reader.readtext(img, detail=0)
                page_text = " ".join(result)

                content += f"--- Страница {page_num + 1} (OCR) ---\n{page_text}\n\n"

                del pix, img, result

            return content

    def _extract_meta_from_text(self, text: str, existing_meta: dict) -> dict:
        meta = existing_meta.copy() if existing_meta else {}
        preview = text[:2000]

        author_patterns = [
            r"(?:Автор[ы]?|Автор:|Авторы:?)\s*[:\-]?\s*([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.[А-ЯЁ]\.?)",
            r"(?:Author[s]?:?)\s*[:\-]?\s*([A-Z][a-z]+\s+[A-Z]\.[A-Z]\.?)",
            r"^([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)",
        ]
        for pattern in author_patterns:
            match = re.search(pattern, preview, re.MULTILINE)
            if match and not meta.get("author"):
                meta["author"] = match.group(1).strip()
                break

        year_pattern = r"\b(19\d{2}|20[0-2]\d)\b"
        year_matches = re.findall(year_pattern, preview)
        if year_matches and not meta.get("year"):
            meta["year"] = year_matches[0]

        doi_pattern = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
        doi_match = re.search(doi_pattern, preview, re.IGNORECASE)
        if doi_match and not meta.get("doi"):
            meta["doi"] = doi_match.group(0)

        journal_pattern = r"(?:журнал|вестник|издание|conference|journal)\s+[:\-]?\s*([А-ЯЁA-Z][^.\n]{10,100})"
        journal_match = re.search(journal_pattern, preview, re.IGNORECASE)
        if journal_match and not meta.get("journal"):
            meta["journal"] = journal_match.group(1).strip()

        return meta


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


class LocalEmbedder:
    """

    """
    def __init__(self, model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """

        :param model:
        """
        self.model = SentenceTransformer(model)

    def encode(self, texts):
        """

        :param texts:
        :return:
        """
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings.tolist()

class QdrantUploader:
    """

    """

    def __init__(self, collection_name: str, url: str = "http://localhost:6333"):
        """
        Конструктор класса.
        """
        self.store = VectorStore(collection_name=collection_name, url=url)
        self.collection_name = collection_name
        self._collection_created = False

    def upload(self, vectors, chunks, metadata_list):
        """"""
        if not self._collection_created and vectors:
            self.store.create_collection(vector_size=len(vectors[0]))
            self._collection_created = True

        ids = []
        payloads = []
        for text, meta in zip(chunks, metadata_list):
            uid = hashlib.md5((text + meta.get("source_path", "")).encode()).hexdigest()
            ids.append(uid)
            payloads.append({"text": text, **meta})

        self.store.upload_vectors(vectors=vectors, payloads=payloads, ids=ids)
        return True

class ExecutionReporter:
    def __init__(self):
        self.start_time = time.time()
        self.total_files = 0
        self.success_count = 0
        self.errors = []
        self.total_chunks = 0

    def log_success(self):
        self.success_count += 1

    def log_error(self, file, reason):
        self.errors.append((file, reason))

    def log_chunks(self, count):
        self.total_chunks += count

    def print_summary(self):
        duration = time.time() - self.start_time
        print("\n" + "=" * 50)
        print(f"{'ИТОГОВЫЙ ОТЧЕТ ИНДЕКСАЦИИ':^50}")
        print("=" * 50)
        print(f"Время работы:       {duration:.2f} сек.")
        print(f"Файлов обработано:  {self.total_files}")
        print(f"Успешно:            {self.success_count}")
        print(f"Ошибок:             {len(self.errors)}")
        print(f"Создано векторов:   {self.total_chunks}")

        if self.errors:
            print("-" * 50)
            print("СПИСОК ПРОБЛЕМНЫХ ФАЙЛОВ:")
            for file, reason in self.errors:
                print(f"  - {file}: {reason}")
        print("=" * 50 + "\n")


class IndexBuilder:
    def __init__(self, scanner, parser, chunker, embedder, uploader):
        self.scanner = scanner
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.uploader = uploader
        self.reporter = ExecutionReporter()

    def run(self):
        files = self.scanner.scan()
        self.reporter.total_files = len(files)

        for file_path in files:
            try:
                result = self.parser.parse_file(file_path)
                text = result["text"]
                meta = result["meta"]

                meta["source_path"] = file_path

                chunks = self.chunker.split(text)

                self.reporter.log_chunks(len(chunks))

                vectors = self.embedder.encode(chunks)

                metadata_list = [meta.copy() for _ in chunks]

                self.uploader.upload(vectors, chunks, metadata_list)

                self.reporter.log_success()

            except Exception as e:
                self.reporter.log_error(file_path, str(e))

        self.reporter.print_summary()


if __name__ == "__main__":
    scanner = PdfScanner(folder="../data")
    parser = PdfParser(use_ocr=True)
    chunker = TextChunker(size=800, overlap=150)
    embedder = LocalEmbedder()
    uploader = QdrantUploader(collection_name="papers_index")

    builder = IndexBuilder(scanner, parser, chunker, embedder, uploader)
    builder.run()