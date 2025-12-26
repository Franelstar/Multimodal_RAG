# app/core/pdf_processor.py

from pathlib import Path
from typing import List, Dict
import pdfplumber
import nltk

from app.utils.hashing import (
    make_pdf_id,
    make_image_id,
    make_llm_chunk_id,
    make_clip_chunk_id
)

nltk.download("punkt")


class PDFProcessor:
    """
    Pipeline complet pour un PDF + image :
    - Extraction de texte
    - Chunking LLM (paragraphes riches)
    - Chunking CLIP (phrases courtes)
    - Liaison via IDs
    """

    def __init__(
        self,
        clip_min_chars: int = 30,
        clip_max_chars: int = 250,
        llm_chunk_size: int = 800,
        llm_overlap: int = 100,
    ):
        self.clip_min_chars = clip_min_chars
        self.clip_max_chars = clip_max_chars
        self.llm_chunk_size = llm_chunk_size
        self.llm_overlap = llm_overlap

    # -------- TEXT EXTRACTION -------- #
    def extract_pages(self, pdf_path: str | Path) -> List[str]:
        pages = []
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return pages

    # -------- CLIP CHUNKING -------- #
    def clip_chunks(self, text: str) -> List[str]:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        return [
            s.strip()
            for s in sentences
            if self.clip_min_chars <= len(s) <= self.clip_max_chars
        ]

    # -------- LLM CHUNKING -------- #
    def llm_chunks(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.llm_chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - self.llm_overlap
        return chunks

    # -------- MAIN PIPELINE -------- #
    def process(self, pdf_path: str | Path, image_path: str | Path) -> Dict:
        pdf_id = make_pdf_id(pdf_path)
        image_id = make_image_id(image_path)

        pages = self.extract_pages(pdf_path)

        clip_docs = []
        clip_ids = []
        clip_meta = []
        llm_chunks_store = {}  # llm_chunk_id -> texte

        for page_num, page_text in enumerate(pages, start=1):
            # créer les LLM chunks
            llm_chunks = self.llm_chunks(page_text)

            for llm_idx, llm_text in enumerate(llm_chunks):
                llm_chunk_id = make_llm_chunk_id(pdf_id, page_num, llm_idx)
                llm_chunks_store[llm_chunk_id] = llm_text

                # CLIP chunks associés à ce LLM chunk
                for sentence in self.clip_chunks(llm_text):
                    clip_id = make_clip_chunk_id(llm_chunk_id, sentence)
                    clip_docs.append(sentence)
                    clip_ids.append(clip_id)
                    clip_meta.append({
                        "type": "text",
                        "pdf_id": pdf_id,
                        "image_id": image_id,
                        "page": page_num,
                        "llm_chunk_id": llm_chunk_id
                    })

        return {
            "pdf_id": pdf_id,
            "image_id": image_id,
            "clip_docs": clip_docs,
            "clip_ids": clip_ids,
            "clip_meta": clip_meta,
            "llm_chunks_store": llm_chunks_store
        }
