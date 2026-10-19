r"""
ChromaDB access layer -- Standards 6, 7 and 8.

    - ONE PersistentClient per standard, each pointed at that standard's
      own folder.
    - Collection names: "english_<standard>", "tamil_<standard>".
    - No hnsw:space override at creation -- Chroma's default distance
      metric (squared L2) is in effect. retrieve_chroma.py computes true
      cosine itself, so this does not matter.

Folder structure:
    ...\Desktop\ALIGNED\6ALIGNED\6\   (standard 6's Chroma data)
    ...\Desktop\ALIGNED\7ALIGNED\7\   (standard 7's Chroma data)
    ...\Desktop\ALIGNED\8ALIGNED\8\   (standard 8's Chroma data)
"""
import os
import chromadb

#CHROMA_PATH_STD6 = r"C:\Users\DELL\OneDrive\Desktop\ALIGNED\6ALIGNED\6"
#CHROMA_PATH_STD7 = r"C:\Users\DELL\OneDrive\Desktop\ALIGNED\7ALIGNED\7"

CHROMA_PATH_STD6 = r"C:\Users\sahim\OneDrive\Documents\Attachments\Desktop\ALIGNED\6ALIGNED\6"
CHROMA_PATH_STD7 = r"C:\Users\sahim\OneDrive\Documents\Attachments\Desktop\ALIGNED\7ALIGNED\7"
CHROMA_PATH_STD8 = r"C:\Users\sahim\OneDrive\Documents\Attachments\Desktop\ALIGNED\8ALIGNED\8"

VALID_STANDARDS = {6, 7, 8}

CHROMA_PATHS = {
    6: CHROMA_PATH_STD6,
    7: CHROMA_PATH_STD7,
    8: CHROMA_PATH_STD8,
}

_clients = {}


def _validate_standard(standard):
    if standard not in VALID_STANDARDS:
        raise ValueError(
            f"Invalid standard: {standard}. Supported standards: 6, 7, 8"
        )


def get_client(standard):
    _validate_standard(standard)
    if standard not in _clients:
        path = CHROMA_PATHS[standard]
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"No ChromaDB folder found for standard {standard} at:\n  {path}\n"
                f"Check CHROMA_PATH_STD{standard} in db.py."
            )
        _clients[standard] = chromadb.PersistentClient(path=path)
    return _clients[standard]


def _collection_name(standard, language):
    if language not in ("english", "tamil"):
        raise ValueError(f"Invalid language: {language}. Must be 'english' or 'tamil'")
    return f"{language}_{standard}"  # "english_8", "tamil_8"


def get_collection(standard, language, create_if_missing=False):
    """
    standard: 6, 7, or 8.
    language: "english" or "tamil"
    """
    _validate_standard(standard)
    name = _collection_name(standard, language)
    client = get_client(standard)
    if create_if_missing:
        return client.get_or_create_collection(name)
    return client.get_collection(name)


def get_standard_english_db(standard):
    return get_collection(standard, "english")


def get_standard_tamil_db(standard):
    return get_collection(standard, "tamil")


if __name__ == "__main__":
    # Quick connectivity check -- all three standards.
    for std in sorted(VALID_STANDARDS):
        try:
            en = get_standard_english_db(std)
            ta = get_standard_tamil_db(std)
            print(f"Standard {std}: english={en.count()} records, tamil={ta.count()} records")
        except FileNotFoundError as e:
            print(f"Standard {std}: NOT CONFIGURED -- {e}")
        except Exception as e:
            print(f"Standard {std}: ERROR -- {e}")