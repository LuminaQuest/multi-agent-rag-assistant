"""
chunking.py — Context-Aware Semantic & Sliding-Window Chunker.
Splits enterprise documents (text, markdown, documentation) into overlapping,
semantically coherent passages while preserving headings and structural hierarchy.
"""

import hashlib
import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    chunk_index: int
    content: str
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

def chunk_document(
    text: str,
    doc_name: str,
    chunk_size: int = 450,
    chunk_overlap: int = 90
) -> List[DocumentChunk]:
    """
    Splits text by structural paragraph delimiters, falling back to sliding window.
    """
    doc_id = hashlib.sha256(f"{doc_name}:{text[:100]}".encode()).hexdigest()[:12]
    
    # Split by double newline or headers
    raw_paragraphs = re.split(r"\n\s*\n|(?=^#{1,3}\s)", text, flags=re.MULTILINE)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
    
    chunks: List[DocumentChunk] = []
    current_chunk = ""
    chunk_idx = 0
    
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}".strip() if current_chunk else para
        else:
            if current_chunk:
                cid = f"{doc_id}-c{chunk_idx:03d}"
                chunks.append(DocumentChunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    chunk_index=chunk_idx,
                    content=current_chunk,
                    token_count=len(current_chunk.split()),
                    metadata={"source": doc_name, "char_len": len(current_chunk)}
                ))
                chunk_idx += 1
                
                # Apply sliding window overlap from end of current chunk
                words = current_chunk.split()
                overlap_text = " ".join(words[-min(len(words), chunk_overlap // 5):])
                current_chunk = f"{overlap_text}\n\n{para}".strip()
            else:
                current_chunk = para
                
    if current_chunk:
        cid = f"{doc_id}-c{chunk_idx:03d}"
        chunks.append(DocumentChunk(
            chunk_id=cid,
            doc_id=doc_id,
            doc_name=doc_name,
            chunk_index=chunk_idx,
            content=current_chunk,
            token_count=len(current_chunk.split()),
            metadata={"source": doc_name, "char_len": len(current_chunk)}
        ))
        
    return chunks
