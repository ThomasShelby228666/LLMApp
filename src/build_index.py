import re
import fitz
import easyocr
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
import time
from vector_store import VectorStore
from typing import *
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config import DATA_FOLDER, EMBEDDING_MODEL, QDRANT_COLLECTION, QDRANT_URL, DEVICE

class PdfScanner:
    """
    Сканер PDF-файлов в директории.
    """
    def __init__(self, folder: str) -> None:
        """
        Инициализация сканера.
        """
        self.folder = Path(folder).resolve()

    def scan(self) -> List[str]:
        """
        Рекурсивный поиск всех PDF-файлов в директории.
        """
        paths = []

        for path in self.folder.rglob("*.pdf"):
            if path.name.startswith("~") or path.name.startswith("."):
                continue

            paths.append(str(path))
        return paths

class PdfParser:
    """
    Парсер PDF-файлов с поддержкой OCR.
    """
    def __init__(self, use_ocr: bool = True) -> None:
        """
        Инициализация парсера.
        """
        self.use_ocr = use_ocr
        self.reader = None
        if self.use_ocr:
            self.reader = easyocr.Reader(["ru", "en"], gpu=(DEVICE == "cuda"))

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Извлечение текста и метаданных из PDF-файла.
        """
        content = ""

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

    def _parse_with_ocr(self, file_path: str) -> str:
        """
        Извлечение текста с помощью OCR.
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

    def _extract_meta_from_text(self, text: str, existing_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлечение метаданных из текста с помощью регулярных выражений.
        """
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

        meta.update({
            "authors": meta.get("author", ""),
            "title": meta.get("title"),
            "section": "Abstract",
            "page": 1,
        })

        return meta


class TextChunker:
    """
    Разбиение текста на чанки (сегменты) для векторного индексирования.
    """
    def __init__(self, size: int = 800, overlap: int = 150) -> None:
        """
        Инициализация чанкера.
        """
        self.size = size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        """
        Разделение текста на чанки с учетом абзацев.
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
    Локальный эмбеддер для преобразования текста в векторные представления.
    """
    def __init__(self, model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        """
        Инициализация эмбеддера.
        """
        self.model = SentenceTransformer(model, device=DEVICE)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Преобразование текстов в векторные эмбеддинги.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=DEVICE
        )

        return embeddings.tolist()

class QdrantUploader:
    """
    Загрузчик векторов в Qdrant коллекцию.
    """

    def __init__(self, collection_name: str, url: str = "http://localhost:6333") -> None:
        """
        Инициализация загрузчика.
        """
        self.store = VectorStore(collection_name=collection_name, url=url)
        self.collection_name = collection_name
        self._collection_created = False

    def upload(self, vectors: List[List[float]], chunks: List[str], metadata_list: List[Dict[str, Any]]) -> bool:
        """
        Загрузка векторов с метаданными в Qdrant.
        """
        if not self._collection_created and vectors:
            self.store.create_collection(vector_size=len(vectors[0]))
            self._collection_created = True

        ids = []
        payloads = []
        for text, meta in zip(chunks, metadata_list):
            uid = hashlib.md5((text + meta.get("source_path", "")).encode()).hexdigest()
            ids.append(uid)
            payload = {"text": text}
            payload.update({k: v for k, v in meta.items() if v})
            payloads.append(payload)

        self.store.upload_vectors(vectors=vectors, payloads=payloads, ids=ids)
        return True

class ExecutionReporter:
    """
    Отчет выполнения индексации.
    """

    def __init__(self) -> None:
        """
        Инициализация репортера.
        """
        self.start_time = time.time()
        self.total_files = 0
        self.success_count = 0
        self.errors = []
        self.total_chunks = 0

    def log_success(self) -> None:
        """
        Логирование успешной обработки файла.
        """
        self.success_count += 1

    def log_error(self, file: str, reason: str) -> None:
        """
         Логирование ошибки обработки файла.
         """
        self.errors.append((file, reason))

    def log_chunks(self, count: int) -> None:
        """
        Логирование количества созданных чанков.
        """
        self.total_chunks += count

    def print_summary(self) -> None:
        """
        Вывод итоговой статистики индексации.
        """
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
    """
    Основной класс построения индекса.
    """

    def __init__(
            self,
            scanner: PdfScanner,
            parser: PdfParser,
            chunker: TextChunker,
            embedder: LocalEmbedder,
            uploader: QdrantUploader
    ) -> None:
        """
        Инициализация построителя индекса
        """
        self.scanner = scanner
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.uploader = uploader
        self.reporter = ExecutionReporter()

    def run(self) -> None:
        """
        Запуск процесса индексации всех найденных PDF-файлов
        """
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
    # Конфигурация и запуск индексации
    scanner = PdfScanner(folder=DATA_FOLDER)
    parser = PdfParser(use_ocr=True)
    chunker = TextChunker(size=800, overlap=150)
    embedder = LocalEmbedder(model=EMBEDDING_MODEL)
    uploader = QdrantUploader(collection_name=QDRANT_COLLECTION, url=QDRANT_URL)

    builder = IndexBuilder(scanner, parser, chunker, embedder, uploader)
    builder.run()