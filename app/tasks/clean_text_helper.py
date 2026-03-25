
def _clean_extracted_text(text: str) -> str:
    """
    Clean extracted text by merging lines that were split by PDF layout,
    while preserving paragraph breaks.
    """
    if not text:
        return ""

    # 1. Replace potential double newlines (paragraphs) with a placeholder
    # PDF paragraphs are often separated by empty lines (\n\n)
    text = text.replace("\r\n", "\n")
    
    # Handle hyphenated words at end of line (e.g. "re-\nvolu-\ntion")
    # Replace "-\n" with "" to join the word
    text = text.replace("-\n", "")
    
    # 2. Heuristic: If we have \n\n, it's definitely a paragraph break.
    # Preserve it with a special token.
    text = text.replace("\n\n", "___PARAGRAPH___")
    
    # 3. Replace remaining single newlines with a space
    text = text.replace("\n", " ")
    
    # 4. Collapse multiple spaces
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # 5. Restore paragraphs
    text = text.replace("___PARAGRAPH___", "\n\n")
    
    return text.strip()
