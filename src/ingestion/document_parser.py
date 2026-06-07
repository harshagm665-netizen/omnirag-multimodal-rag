import os
import io
from typing import List, Dict, Any
import fitz  # PyMuPDF
from PIL import Image

class DocumentPage:
    def __init__(self, image: Image.Image, metadata: Dict[str, Any]):
        self.image = image
        self.metadata = metadata

class DocumentParser:
    """
    Parses documents (PDFs, Images) into a list of DocumentPages containing images and metadata.
    """
    def __init__(self, temp_dir: str = "/tmp/rag_uploads"):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def parse_pdf(self, file_path: str, source_filename: str) -> List[DocumentPage]:
        """
        Convert PDF into images (one per page) using PyMuPDF.
        """
        pages = []
        try:
            doc = fitz.open(file_path)
            for i in range(len(doc)):
                page = doc.load_page(i)
                # Render page to an image pixmap
                pix = page.get_pixmap(dpi=150)
                # Convert pixmap to PIL Image
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                
                metadata = {
                    "source": source_filename,
                    "page_number": i + 1,
                    "type": "pdf",
                    "extracted_text": page.get_text("text").strip()
                }
                pages.append(DocumentPage(image=img, metadata=metadata))
            return pages
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF {source_filename}: {str(e)}")

    def parse_image(self, file_path: str, source_filename: str) -> List[DocumentPage]:
        """
        Load a single image file.
        """
        try:
            img = Image.open(file_path).convert("RGB")
            metadata = {
                "source": source_filename,
                "page_number": 1,
                "type": "image"
            }
            return [DocumentPage(image=img, metadata=metadata)]
        except Exception as e:
            raise RuntimeError(f"Failed to parse Image {source_filename}: {str(e)}")

    def parse_file(self, file_path: str, source_filename: str) -> List[DocumentPage]:
        ext = source_filename.lower().split('.')[-1]
        if ext == 'pdf':
            return self.parse_pdf(file_path, source_filename)
        elif ext in ['png', 'jpg', 'jpeg', 'webp']:
            return self.parse_image(file_path, source_filename)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
