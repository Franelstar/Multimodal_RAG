# app/utils/hashing.py

import hashlib
from pathlib import Path


def sha1_from_text(text: str) -> str:
    """
    Hash SHA1 déterministe à partir d'un texte.
    Utilisé pour :
    - ids de chunks
    - ids logiques (llm_chunk_id, clip_id, etc.)
    """
    if not isinstance(text, str):
        raise TypeError("sha1_from_text attend une chaîne de caractères")

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha1_from_file(file_path: str | Path, chunk_size: int = 8192) -> str:
    """
    Hash SHA1 déterministe à partir du contenu d'un fichier.
    Utilisé pour :
    - pdf_id
    - image_id
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    sha1 = hashlib.sha1()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha1.update(chunk)

    return sha1.hexdigest()


def make_pdf_id(pdf_path: str | Path) -> str:
    return f"pdf_{sha1_from_file(pdf_path)}"


def make_image_id(image_path: str | Path) -> str:
    return f"img_{sha1_from_file(image_path)}"


def make_llm_chunk_id(pdf_id: str, page: int, chunk_index: int) -> str:
    base = f"{pdf_id}_page_{page}_llm_{chunk_index}"
    return sha1_from_text(base)


def make_clip_chunk_id(llm_chunk_id: str, clip_text: str) -> str:
    base = f"{llm_chunk_id}_{clip_text}"
    return sha1_from_text(base)


def make_session_id(user_id: str, conversation_id: str) -> str:
    """
    Session pour une conversation utilisateur concernant une image.
    """
    base = f"{user_id}_{conversation_id}"
    return sha1_from_text(base)