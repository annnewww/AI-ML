import re


def clean_ticket_text(text):
    if not isinstance(text, str):
        return ""

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove HTML tags (e.g., <br>, <div>) if tickets come from web forms
    text = re.sub(r'<[^>]+>', ' ', text)

    # 3. Remove URLs/hyperlinks
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 4. Remove illegal/special characters (Keep only letters, numbers, and basic punctuation)
    # This removes emoji, control characters, and weird symbols like ^, ~, *, etc.
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\-\_\?]', '', text)

    # 5. Replace multiple spaces, tabs, or newlines with a single space
    text = re.sub(r'\s+', ' ', text)

    # 6. Strip leading and trailing whitespace
    return text.strip()