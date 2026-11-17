"""
Document Processor Module: Extracts structured text from PDF, DOCX, TXT files, and raw strings.
Handles scanned PDF detection, structural formatting preservation, and multi-document ingestion.
"""

from pathlib import Path
from typing import List, Dict, Union, Optional
import io
import re

from config import MIN_CHAR_COUNT_PER_PAGE
from src.utils import (
    get_logger,
    DocumentProcessingError,
    ScannedDocumentException,
    clean_text_whitespace
)

logger = get_logger("DocumentProcessor")


class DocumentProcessor:
    """Handles text extraction and structure parsing from clinical documents."""

    def __init__(self):
        self._check_available_libraries()

    def _check_available_libraries(self):
        """Checks available third-party document processing libraries."""
        self.has_fitz = False
        self.has_pypdf = False
        self.has_docx = False

        try:
            import fitz  # PyMuPDF
            self.has_fitz = True
        except ImportError:
            try:
                import pypdf
                self.has_pypdf = True
            except ImportError:
                pass

        try:
            import docx
            self.has_docx = True
        except ImportError:
            pass

    def extract_from_file(self, file_path: Union[str, Path]) -> Dict[str, Union[str, int, bool]]:
        """
        Extracts text from a single file path (.pdf, .docx, .txt).

        Returns:
            Dict containing:
                - text: Extracted string content
                - file_type: Extension
                - page_count: Total pages (for PDF) or paragraphs (for DOCX)
                - is_scanned: True if PDF requires OCR
                - source_name: Filename
        """
        path = Path(file_path)
        if not path.exists():
            raise DocumentProcessingError(f"File not found: {path}")

        ext = path.suffix.lower()
        logger.info("Processing document (type: %s)", ext)

        if ext == ".txt":
            return self._extract_txt(path)
        elif ext == ".pdf":
            return self._extract_pdf(path)
        elif ext in [".docx", ".doc"]:
            return self._extract_docx(path)
        else:
            raise DocumentProcessingError(f"Unsupported file format: '{ext}'. Supported: .pdf, .docx, .txt")

    def extract_from_bytes(self, file_bytes: bytes, filename: str) -> Dict[str, Union[str, int, bool]]:
        """Extracts text from byte streams (e.g. Streamlit uploaded files)."""
        ext = Path(filename).suffix.lower()
        logger.info("Processing uploaded byte stream (type: %s, bytes=%d)", ext, len(file_bytes))

        if ext == ".txt":
            try:
                text = file_bytes.decode("utf-8", errors="replace")
                return {
                    "text": clean_text_whitespace(text),
                    "file_type": "txt",
                    "page_count": 1,
                    "is_scanned": False,
                    "source_name": filename
                }
            except Exception as e:
                raise DocumentProcessingError(f"Failed to read TXT bytes: {e}")

        elif ext == ".pdf":
            return self._extract_pdf_bytes(file_bytes, filename)

        elif ext in [".docx", ".doc"]:
            return self._extract_docx_bytes(file_bytes, filename)

        else:
            raise DocumentProcessingError(f"Unsupported file extension: {ext}")

    def extract_from_text(self, raw_text: str, source_name: str = "Manual Input") -> Dict[str, Union[str, int, bool]]:
        """Processes manual string text input."""
        cleaned = clean_text_whitespace(raw_text)
        if not cleaned:
            raise DocumentProcessingError("Input text is empty.")
        return {
            "text": cleaned,
            "file_type": "raw_text",
            "page_count": 1,
            "is_scanned": False,
            "source_name": source_name
        }

    def process_multiple_documents(self, file_inputs: List[Union[str, Path, Dict]]) -> Dict[str, Union[str, List]]:
        """
        Combines and processes multiple clinical documents for a single patient case.
        """
        combined_texts = []
        doc_metadata = []

        for idx, item in enumerate(file_inputs, 1):
            if isinstance(item, (str, Path)):
                result = self.extract_from_file(item)
            elif isinstance(item, dict) and "bytes" in item and "name" in item:
                result = self.extract_from_bytes(item["bytes"], item["name"])
            elif isinstance(item, dict) and "text" in item:
                result = self.extract_from_text(item["text"], item.get("source_name", f"Document_{idx}"))
            else:
                continue

            header = f"=== DOCUMENT {idx}: {result['source_name']} ==="
            combined_texts.append(f"{header}\n{result['text']}")
            doc_metadata.append(result)

        full_combined_text = "\n\n".join(combined_texts)
        return {
            "combined_text": full_combined_text,
            "document_count": len(doc_metadata),
            "documents": doc_metadata
        }

    # --- Private Extractor Methods ---

    def _extract_txt(self, path: Path) -> Dict[str, Union[str, int, bool]]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            cleaned = clean_text_whitespace(text)
            if not cleaned:
                raise DocumentProcessingError(f"TXT file '{path.name}' is empty.")
            return {
                "text": cleaned,
                "file_type": "txt",
                "page_count": 1,
                "is_scanned": False,
                "source_name": path.name
            }
        except Exception as e:
            raise DocumentProcessingError(f"Error reading TXT file {path.name}: {e}")

    def _extract_pdf(self, path: Path) -> Dict[str, Union[str, int, bool]]:
        if self.has_fitz:
            return self._extract_pdf_fitz_path(path)
        elif self.has_pypdf:
            return self._extract_pdf_pypdf_path(path)
        else:
            return self._extract_pdf_fallback_path(path)

    def _extract_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> Dict[str, Union[str, int, bool]]:
        if self.has_fitz:
            import fitz
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                pages_text = []
                total_chars = 0
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")
                    pages_text.append(f"--- Page {page_num + 1} ---\n{text}")
                    total_chars += len(text.strip())

                avg_chars = total_chars / max(len(doc), 1)
                is_scanned = avg_chars < MIN_CHAR_COUNT_PER_PAGE

                if is_scanned and total_chars == 0:
                    raise ScannedDocumentException(
                        f"PDF '{filename}' appears to be a scanned or image-only document. "
                        "No text layer detected. OCR (Optical Character Recognition) processing is required."
                    )

                return {
                    "text": clean_text_whitespace("\n".join(pages_text)),
                    "file_type": "pdf",
                    "page_count": len(doc),
                    "is_scanned": is_scanned,
                    "source_name": filename
                }
            except ScannedDocumentException:
                raise
            except Exception as e:
                raise DocumentProcessingError(f"Failed to process PDF bytes for {filename}: {e}")
        else:
            # Fallback simple regex text recovery from PDF stream
            text = self._extract_text_from_pdf_raw_stream(pdf_bytes)
            if len(text.strip()) < MIN_CHAR_COUNT_PER_PAGE:
                raise ScannedDocumentException(
                    f"PDF '{filename}' has no detectable text layer. OCR required."
                )
            return {
                "text": clean_text_whitespace(text),
                "file_type": "pdf",
                "page_count": 1,
                "is_scanned": False,
                "source_name": filename
            }

    def _extract_pdf_fitz_path(self, path: Path) -> Dict[str, Union[str, int, bool]]:
        import fitz
        try:
            doc = fitz.open(path)
            pages_text = []
            total_chars = 0
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                pages_text.append(f"--- Page {page_num + 1} ---\n{text}")
                total_chars += len(text.strip())

            avg_chars = total_chars / max(len(doc), 1)
            is_scanned = avg_chars < MIN_CHAR_COUNT_PER_PAGE

            if is_scanned and total_chars == 0:
                raise ScannedDocumentException(
                    f"PDF '{path.name}' is a scanned or image-only document. "
                    "No text layer detected. OCR processing required."
                )

            return {
                "text": clean_text_whitespace("\n".join(pages_text)),
                "file_type": "pdf",
                "page_count": len(doc),
                "is_scanned": is_scanned,
                "source_name": path.name
            }
        except ScannedDocumentException:
            raise
        except Exception as e:
            raise DocumentProcessingError(f"PyMuPDF error reading {path.name}: {e}")

    def _extract_pdf_pypdf_path(self, path: Path) -> Dict[str, Union[str, int, bool]]:
        import pypdf
        try:
            reader = pypdf.PdfReader(str(path))
            pages_text = []
            total_chars = 0
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages_text.append(f"--- Page {idx + 1} ---\n{text}")
                total_chars += len(text.strip())

            avg_chars = total_chars / max(len(reader.pages), 1)
            is_scanned = avg_chars < MIN_CHAR_COUNT_PER_PAGE

            if is_scanned and total_chars == 0:
                raise ScannedDocumentException(
                    f"PDF '{path.name}' has no text layer. OCR required."
                )

            return {
                "text": clean_text_whitespace("\n".join(pages_text)),
                "file_type": "pdf",
                "page_count": len(reader.pages),
                "is_scanned": is_scanned,
                "source_name": path.name
            }
        except ScannedDocumentException:
            raise
        except Exception as e:
            raise DocumentProcessingError(f"pypdf error reading {path.name}: {e}")

    def _extract_pdf_fallback_path(self, path: Path) -> Dict[str, Union[str, int, bool]]:
        try:
            with open(path, "rb") as f:
                data = f.read()
            text = self._extract_text_from_pdf_raw_stream(data)
            if len(text.strip()) < MIN_CHAR_COUNT_PER_PAGE:
                raise ScannedDocumentException(f"PDF '{path.name}' is image-based. OCR required.")
            return {
                "text": clean_text_whitespace(text),
                "file_type": "pdf",
                "page_count": 1,
                "is_scanned": False,
                "source_name": path.name
            }
        except ScannedDocumentException:
            raise
        except Exception as e:
            raise DocumentProcessingError(f"PDF fallback reading failed for {path.name}: {e}")

    def _extract_docx(self, path: Path) -> Dict[str, Union[str, int, bool]]:
        if self.has_docx:
            import docx
            try:
                doc = docx.Document(str(path))
                paragraphs = []
                for p in doc.paragraphs:
                    if p.text.strip():
                        # Preserve heading distinction
                        if p.style.name.startswith("Heading"):
                            paragraphs.append(f"\n### {p.text.strip()}\n")
                        else:
                            paragraphs.append(p.text.strip())
                text = "\n".join(paragraphs)
                return {
                    "text": clean_text_whitespace(text),
                    "file_type": "docx",
                    "page_count": len(doc.paragraphs),
                    "is_scanned": False,
                    "source_name": path.name
                }
            except Exception as e:
                raise DocumentProcessingError(f"python-docx error reading {path.name}: {e}")
        else:
            return self._extract_docx_zip_fallback(path)

    def _extract_docx_bytes(self, docx_bytes: bytes, filename: str) -> Dict[str, Union[str, int, bool]]:
        if self.has_docx:
            import docx
            try:
                stream = io.BytesIO(docx_bytes)
                doc = docx.Document(stream)
                paragraphs = []
                for p in doc.paragraphs:
                    if p.text.strip():
                        if p.style.name.startswith("Heading"):
                            paragraphs.append(f"\n### {p.text.strip()}\n")
                        else:
                            paragraphs.append(p.text.strip())
                text = "\n".join(paragraphs)
                return {
                    "text": clean_text_whitespace(text),
                    "file_type": "docx",
                    "page_count": len(doc.paragraphs),
                    "is_scanned": False,
                    "source_name": filename
                }
            except Exception as e:
                raise DocumentProcessingError(f"Failed to read DOCX bytes for {filename}: {e}")
        else:
            return self._extract_docx_zip_bytes_fallback(docx_bytes, filename)

    def _extract_docx_zip_fallback(self, path: Path) -> Dict[str, Union[str, int, bool]]:
        import zipfile
        import xml.etree.ElementTree as ET
        try:
            with zipfile.ZipFile(path) as z:
                xml_content = z.read("word/document.xml")
            root = ET.fromstring(xml_content)
            paragraphs = []
            for elem in root.iter():
                if elem.tag.endswith("t") and elem.text:
                    paragraphs.append(elem.text)
            text = " ".join(paragraphs)
            return {
                "text": clean_text_whitespace(text),
                "file_type": "docx",
                "page_count": 1,
                "is_scanned": False,
                "source_name": path.name
            }
        except zipfile.BadZipFile:
            logger.info(f"File {path.name} is not a valid zip archive, reading as plain text fallback...")
            return self._extract_txt(path)
        except Exception as e:
            raise DocumentProcessingError(f"DOCX extraction failed for {path.name}: {e}")

    def _extract_docx_zip_bytes_fallback(self, docx_bytes: bytes, filename: str) -> Dict[str, Union[str, int, bool]]:
        import zipfile
        import xml.etree.ElementTree as ET
        try:
            with zipfile.ZipFile(io.BytesIO(docx_bytes)) as z:
                xml_content = z.read("word/document.xml")
            root = ET.fromstring(xml_content)
            paragraphs = []
            for elem in root.iter():
                if elem.tag.endswith("t") and elem.text:
                    paragraphs.append(elem.text)
            text = " ".join(paragraphs)
            return {
                "text": clean_text_whitespace(text),
                "file_type": "docx",
                "page_count": 1,
                "is_scanned": False,
                "source_name": filename
            }
        except Exception as e:
            raise DocumentProcessingError(f"DOCX Zip bytes fallback failed for {filename}: {e}")

    def _extract_text_from_pdf_raw_stream(self, data: bytes) -> str:
        """Extracts ASCII/UTF8 strings directly from PDF binary streams when no PDF library is available."""
        text_matches = re.findall(rb"\(([\w\s\.,;:!?\-\/\\+=%\#@\(\)]+)\)\s*T[jJ]", data)
        if text_matches:
            decoded = [m.decode("latin1", errors="ignore") for m in text_matches]
            return " ".join(decoded)
        # Fallback printable string scan
        strings = re.findall(rb"[A-Za-z0-9\s\.,;:!?\-\/]{4,}", data)
        decoded = [s.decode("latin1", errors="ignore") for s in strings if len(s.strip()) > 5]
        return "\n".join(decoded[:200])
