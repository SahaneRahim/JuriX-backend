"""
Utility functions for file operations.

Provides helpers for:
- File hashing (SHA-256)
- File type detection (magic bytes)
- File validation (PDF, DOCX structure)
- Filename generation (UUID-based)
- Storage management

Author: JuriX Development Team
Date: 2026-01-11
"""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# File type magic bytes signatures
MAGIC_BYTES = {
    "pdf": b"%PDF",
    "docx": b"PK\x03\x04",  # ZIP signature (DOCX is a ZIP archive)
}

# MIME types
MIME_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def get_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Calculate hash of a file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hex digest of file hash

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If algorithm not supported

    Example:
        >>> hash_value = get_file_hash(Path("document.pdf"))
        >>> len(hash_value)
        64
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Hash algorithm '{algorithm}' not supported")

    hash_obj = hashlib.new(algorithm)
    
    # Read file in chunks to handle large files
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def get_magic_bytes(file_path: Path, num_bytes: int = 4) -> bytes:
    """
    Read the first N bytes of a file (magic bytes).

    Args:
        file_path: Path to file
        num_bytes: Number of bytes to read (default: 4)

    Returns:
        First N bytes of file

    Raises:
        FileNotFoundError: If file doesn't exist

    Example:
        >>> magic = get_magic_bytes(Path("document.pdf"))
        >>> magic
        b'%PDF'
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        return f.read(num_bytes)


def detect_file_type(file_path: Path) -> Optional[str]:
    """
    Detect file type from magic bytes.

    Args:
        file_path: Path to file

    Returns:
        File type ('pdf' or 'docx') or None if unknown

    Example:
        >>> detect_file_type(Path("document.pdf"))
        'pdf'
    """
    try:
        magic = get_magic_bytes(file_path, num_bytes=4)
        
        for file_type, signature in MAGIC_BYTES.items():
            if magic.startswith(signature):
                return file_type
        
        return None
    except Exception as e:
        logger.warning(f"Failed to detect file type for {file_path}: {e}")
        return None


def is_valid_pdf(file_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate PDF file structure.

    Checks:
    - Magic bytes (%PDF)
    - File ends with %%EOF
    - Minimum file size

    Args:
        file_path: Path to PDF file

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> is_valid, error = is_valid_pdf(Path("document.pdf"))
        >>> is_valid
        True
    """
    try:
        # Check file exists and has minimum size
        if not file_path.exists():
            return False, "File does not exist"
        
        file_size = file_path.stat().st_size
        if file_size < 100:  # Minimum PDF size
            return False, "File too small to be a valid PDF"

        # Check magic bytes
        magic = get_magic_bytes(file_path, num_bytes=4)
        if not magic.startswith(b"%PDF"):
            return False, "Invalid PDF magic bytes"

        # Check EOF marker (read last 1024 bytes)
        with open(file_path, "rb") as f:
            f.seek(max(0, file_size - 1024))
            tail = f.read()
            if b"%%EOF" not in tail:
                return False, "Missing PDF EOF marker"

        return True, None

    except Exception as e:
        return False, f"PDF validation error: {str(e)}"


def is_valid_docx(file_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate DOCX file structure.

    Checks:
    - Magic bytes (PK - ZIP signature)
    - Contains required DOCX files (word/document.xml)
    - Minimum file size

    Args:
        file_path: Path to DOCX file

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> is_valid, error = is_valid_docx(Path("document.docx"))
        >>> is_valid
        True
    """
    try:
        # Check file exists and has minimum size
        if not file_path.exists():
            return False, "File does not exist"
        
        file_size = file_path.stat().st_size
        if file_size < 1000:  # Minimum DOCX size
            return False, "File too small to be a valid DOCX"

        # Check magic bytes (ZIP signature)
        magic = get_magic_bytes(file_path, num_bytes=4)
        if not magic.startswith(b"PK\x03\x04"):
            return False, "Invalid DOCX magic bytes (not a ZIP file)"

        # Try to open as ZIP and check for required DOCX structure
        import zipfile
        
        if not zipfile.is_zipfile(file_path):
            return False, "File is not a valid ZIP archive"

        with zipfile.ZipFile(file_path, "r") as zip_file:
            # Check for required DOCX files
            required_files = ["[Content_Types].xml", "word/document.xml"]
            zip_contents = zip_file.namelist()
            
            for required_file in required_files:
                if required_file not in zip_contents:
                    return False, f"Missing required DOCX file: {required_file}"

        return True, None

    except zipfile.BadZipFile:
        return False, "Corrupted ZIP/DOCX file"
    except Exception as e:
        return False, f"DOCX validation error: {str(e)}"


def generate_unique_filename(original_filename: str, preserve_extension: bool = True) -> str:
    """
    Generate a unique filename using UUID.

    Args:
        original_filename: Original filename
        preserve_extension: Whether to keep original extension (default: True)

    Returns:
        Unique filename (UUID-based)

    Example:
        >>> filename = generate_unique_filename("document.pdf")
        >>> filename.endswith(".pdf")
        True
        >>> len(filename.split(".")[0])
        36  # UUID length
    """
    unique_id = str(uuid.uuid4())
    
    if preserve_extension:
        # Extract extension from original filename
        extension = Path(original_filename).suffix
        return f"{unique_id}{extension}"
    else:
        return unique_id


def ensure_storage_directory(path: Path) -> None:
    """
    Ensure storage directory exists, create if necessary.

    Args:
        path: Directory path

    Raises:
        PermissionError: If directory cannot be created
        OSError: If path exists but is not a directory

    Example:
        >>> ensure_storage_directory(Path("/tmp/uploads"))
    """
    if path.exists():
        if not path.is_dir():
            raise OSError(f"Path exists but is not a directory: {path}")
        logger.debug(f"Storage directory already exists: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created storage directory: {path}")


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB (rounded to 2 decimals)

    Example:
        >>> size_mb = get_file_size_mb(Path("document.pdf"))
        >>> size_mb
        2.5
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)


def is_file_size_valid(file_path: Path, max_size_mb: int = 50) -> Tuple[bool, Optional[str]]:
    """
    Check if file size is within allowed limit.

    Args:
        file_path: Path to file
        max_size_mb: Maximum allowed size in MB (default: 50)

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> is_valid, error = is_file_size_valid(Path("document.pdf"), max_size_mb=50)
        >>> is_valid
        True
    """
    try:
        size_mb = get_file_size_mb(file_path)
        
        if size_mb > max_size_mb:
            return False, f"File size {size_mb} MB exceeds maximum {max_size_mb} MB"
        
        if size_mb == 0:
            return False, "File is empty (0 bytes)"
        
        return True, None

    except Exception as e:
        return False, f"Size validation error: {str(e)}"


def get_mime_type(file_type: str) -> str:
    """
    Get MIME type for file type.

    Args:
        file_type: File type ('pdf' or 'docx')

    Returns:
        MIME type string

    Raises:
        ValueError: If file type not supported

    Example:
        >>> get_mime_type("pdf")
        'application/pdf'
    """
    if file_type not in MIME_TYPES:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return MIME_TYPES[file_type]
