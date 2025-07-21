import re

def clean_text(text: str) -> str:
    """
    Clean the text by removing bracketed content, multiple spaces, and normalizing hyphens.
    
    Args:
        text (str): The input text.
    
    Returns:
        str: Cleaned text.
    """
    cleaned_text = _remove_brackets(text)
    cleaned_text = _remove_square_brackets(cleaned_text)
    cleaned_text = _remove_multiple_spaces(cleaned_text)
    cleaned_text = _replace_weird_hyphen(cleaned_text)
    return cleaned_text


def _remove_brackets(text: str) -> str:
    """
    Remove content within parentheses, including the parentheses themselves.
    
    Example:
        "The koala has a body length of 60–85 cm (24–33 in)." → 
        "The koala has a body length of 60–85 cm."
    
    Args:
        text (str): The input text.
    
    Returns:
        str: Text without parenthetical content.
    """
    return re.sub(r'\(.*?\)', '', text)


def _remove_square_brackets(text: str) -> str:
    """
    Remove content within square brackets, including the brackets themselves.
    
    Example:
        "The koala[1] is cool." → "The koala is cool."
    
    Args:
        text (str): The input text.
    
    Returns:
        str: Text without bracketed citations or notes.
    """
    return re.sub(r'\[.*?\]', '', text)


def _remove_multiple_spaces(text: str) -> str:
    """
    Replace multiple spaces with a single space.
    
    Example:
        "The    koala    is   angry!" → "The koala is angry!"
    
    Args:
        text (str): The input text.
    
    Returns:
        str: Normalized text with single spaces.
    """
    return re.sub(r' +', ' ', text).strip()


def _replace_weird_hyphen(text: str) -> str:
    """
    Replace en dashes (–) with standard hyphens (-).
    
    Some tokenizers may not recognize en dashes as delimiters.
    
    Example:
        "4–15 kg" → "4-15 kg"
    
    Args:
        text (str): The input text.
    
    Returns:
        str: Text with normalized hyphens.
    """
    return text.replace('–', '-')
