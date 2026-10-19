import os
import sys
import json
import uuid
import chromadb


# ============================================================
# CONFIGURATION
# ============================================================
#
# STANDARD is now the ONE thing that changes per run -- everything
# below derives from it. Pass it as a command-line argument:
#     python ingest_from_json.py 7
#     python ingest_from_json.py 6
#     python ingest_from_json.py 8
# Defaults to "7" if you run it with no argument, so your existing
# habit of just running the script still works unchanged.
# ============================================================

STANDARD = sys.argv[1] if len(sys.argv) > 1 else "7"

if STANDARD not in ("6", "7", "8"):
    print(f"ERROR: Invalid standard '{STANDARD}'. Must be 6, 7, or 8.")
    raise SystemExit

# Folder containing all your *_aligned.json files
# REAL structure confirmed from screenshots: ONE parent folder
# "Desktop\7\" contains "6ALIGNED\" and "7ALIGNED\" as siblings --
# NOT separate "Desktop\6\ALIGNED\" / "Desktop\7\ALIGNED\" folders.
# Standard 8's real location is not yet known -- deliberately left
# unconfigured per instruction to ignore 8 for now.
#INPUT_FOLDER = rf"C:\Users\DELL\OneDrive\Desktop\ALIGNED\6ALIGNED"

INPUT_FOLDER = rf"C:\Users\sahim\OneDrive\Documents\Attachments\Desktop\ALIGNED\{STANDARD}ALIGNED"
#INPUT_FOLDER = rf"C:\Users\sahim\OneDrive\Documents\Attachments\Desktop\ALIGNED\8ALIGNED"

# ChromaDB folder
CHROMA_FOLDER = os.path.join(INPUT_FOLDER, STANDARD)

# ChromaDB collection names
ENGLISH_COLLECTION_NAME = f"english_{STANDARD}"
TAMIL_COLLECTION_NAME = f"tamil_{STANDARD}"


# ============================================================
# CREATE CHROMADB CLIENT
# ============================================================

print("=" * 70)
print("CREATING CHROMADB")
print("=" * 70)

os.makedirs(CHROMA_FOLDER, exist_ok=True)

client = chromadb.PersistentClient(
    path=CHROMA_FOLDER
)


# ============================================================
# CREATE / GET COLLECTIONS
# ============================================================

english_collection = client.get_or_create_collection(
    name=ENGLISH_COLLECTION_NAME,
    metadata={
        "description": f"English aligned textbook content - Standard {STANDARD}"
    }
)

tamil_collection = client.get_or_create_collection(
    name=TAMIL_COLLECTION_NAME,
    metadata={
        "description": f"Tamil aligned textbook content - Standard {STANDARD}"
    }
)


print()
print("Collections created:")
print("  English :", ENGLISH_COLLECTION_NAME)
print("  Tamil   :", TAMIL_COLLECTION_NAME)
print()


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_first_value(data, possible_keys, default=""):
    """
    Find the first available value from possible keys.
    """

    if not isinstance(data, dict):
        return default

    for key in possible_keys:

        if key in data:

            value = data[key]

            if value is not None:
                return value

    return default


# ============================================================
# CONVERT TEXT TO STRING
# ============================================================

def convert_to_text(value):
    """
    Convert string/list content into one complete text string.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):

        text_parts = []

        for item in value:

            if isinstance(item, str):
                text_parts.append(item.strip())

            elif isinstance(item, dict):

                # Try common text fields
                text = get_first_value(
                    item,
                    [
                        "text",
                        "content",
                        "sentence",
                        "paragraph"
                    ],
                    ""
                )

                if text:
                    text_parts.append(str(text).strip())

        return " ".join(
            part for part in text_parts if part
        ).strip()

    return str(value).strip()


# ============================================================
# GET EMBEDDING
# ============================================================

def get_embedding(data, language):
    """
    Extract existing embedding from JSON.

    Supports several possible naming conventions.
    """

    if not isinstance(data, dict):
        return None

    if language == "english":

        possible_keys = [
            "english_embedding",
            "embedding_english",
            "english_embeddings",
            "embedding"
        ]

    else:

        possible_keys = [
            "tamil_embedding",
            "embedding_tamil",
            "tamil_embeddings"
        ]

    for key in possible_keys:

        if key in data:

            embedding = data[key]

            if isinstance(embedding, list):

                # Check that it is numeric
                try:

                    embedding = [
                        float(x)
                        for x in embedding
                    ]

                    return embedding

                except Exception:
                    pass

    # --------------------------------------------------------
    # Sometimes embedding can be inside:
    #
    # "english": {
    #     "text": "...",
    #     "embedding": [...]
    # }
    # --------------------------------------------------------

    language_data = data.get(language)

    if isinstance(language_data, dict):

        embedding = language_data.get("embedding")

        if isinstance(embedding, list):

            try:

                return [
                    float(x)
                    for x in embedding
                ]

            except Exception:
                pass

    return None


# ============================================================
# GET TEXT
# ============================================================

def get_language_text(data, language):
    """
    Extract complete English/Tamil text.
    """

    value = data.get(language)

    return convert_to_text(value)


# ============================================================
# CREATE UNIQUE ALIGNMENT ID
# ============================================================

def create_alignment_id(file_path, record_index):
    """
    Creates one unique alignment ID for each aligned pair.
    """

    file_name = os.path.basename(file_path)

    base_name = os.path.splitext(file_name)[0]

    return f"{base_name}_align_{record_index:05d}"


# ============================================================
# FIND ALL ALIGNED JSON FILES
# ============================================================

def find_json_files(folder):
    """
    Recursively find all *_aligned.json files.
    """

    json_files = []

    for root, dirs, files in os.walk(folder):

        # Don't read ChromaDB's own files
        if os.path.abspath(root).startswith(
            os.path.abspath(CHROMA_FOLDER)
        ):
            continue

        for file_name in files:

            if file_name.endswith("_aligned.json"):

                full_path = os.path.join(
                    root,
                    file_name
                )

                json_files.append(full_path)

    json_files.sort()

    return json_files


# ============================================================
# LOAD JSON FILE
# ============================================================

def load_json_file(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# NORMALIZE JSON STRUCTURE
# ============================================================

def get_records(json_data):
    """
    Handles common JSON structures.
    """

    if isinstance(json_data, list):

        return json_data

    if isinstance(json_data, dict):

        for key in [
            "data",
            "records",
            "alignments",
            "aligned_data",
            "results"
        ]:

            if key in json_data:

                value = json_data[key]

                if isinstance(value, list):

                    return value

        if (
            "english" in json_data
            and
            "tamil" in json_data
        ):

            return [json_data]

    return []


# ============================================================
# PROCESS ONE RECORD
# ============================================================

def process_record(
    record,
    file_path,
    record_index
):

    if not isinstance(record, dict):

        print(
            f"  WARNING: Record {record_index} is not a dictionary"
        )

        return None


    english_value = record.get("english", "")

    tamil_value = record.get("tamil", "")

    english_text = convert_to_text(
        english_value
    )

    tamil_text = convert_to_text(
        tamil_value
    )


    if not english_text:

        print(
            f"  WARNING: Empty English text in record {record_index}"
        )

        return None

    if not tamil_text:

        print(
            f"  WARNING: Empty Tamil text in record {record_index}"
        )

        return None


    alignment_id = create_alignment_id(
        file_path,
        record_index
    )


    english_embedding = get_embedding(
        record,
        "english"
    )

    tamil_embedding = get_embedding(
        record,
        "tamil"
    )


    if english_embedding is None:

        print(
            f"  WARNING: English embedding missing "
            f"in record {record_index}"
        )

    else:

        if len(english_embedding) != 768:

            print(
                f"  WARNING: English embedding dimension = "
                f"{len(english_embedding)}, expected 768 (LaBSE)"
            )


    if tamil_embedding is None:

        print(
            f"  WARNING: Tamil embedding missing "
            f"in record {record_index}"
        )

    else:

        if len(tamil_embedding) != 768:

            print(
                f"  WARNING: Tamil embedding dimension = "
                f"{len(tamil_embedding)}, expected 768 (LaBSE)"
            )


    source_id = get_first_value(
        record,
        [
            "source_id",
            "sourceId",
            "id"
        ],
        ""
    )

    unit = get_first_value(
        record,
        [
            "unit",
            "unit_id",
            "unit_number"
        ],
        ""
    )

    page = get_first_value(
        record,
        [
            "page",
            "page_number",
            "page_no"
        ],
        ""
    )

    block_type = get_first_value(
        record,
        [
            "category",
            "block_type",
            "type",
            "label",
            "content_type"
        ],
        ""
    )


    # ========================================================
    # GET LOGICAL SECTION ID
    # ========================================================
    #
    # This is what heading-context-expansion at retrieval time will
    # group by: "pull every chunk sharing the same logical_section_id
    # as a top result, so no fragment under that heading is missed."
    #
    # The exact key name in your source JSON isn't confirmed yet, so
    # this tries several plausible names, same pattern as source_id/
    # unit/page/block_type above. If NONE of these match your real
    # JSON's actual key, this will be empty for every record -- watch
    # the WARNING below, which fires exactly when that happens.
    # ========================================================

    logical_section_id = get_first_value(
        record,
        [
            "logical_group_id",
            "section_id",
            "sectionId",
            "logical_section_id",
            "section",
            "heading_id",
            "heading"
        ],
        ""
    )

    if not logical_section_id:

        print(
            f"  WARNING: No logical_section_id found in record {record_index} "
            f"-- tried section_id/sectionId/logical_section_id/section/"
            f"heading_id/heading, none present. Heading-context expansion "
            f"will not work for this record until the real key name is confirmed."
        )


    file_name = os.path.basename(
        file_path
    )

    file_name_without_extension = os.path.splitext(
        file_name
    )[0]


    term = ""

    chapter = ""

    parts = file_name_without_extension.split("_")

    for part in parts:

        if part.startswith("term"):

            term = part

        if part.startswith("ch"):

            chapter = part


    original_json = json.dumps(
        record,
        ensure_ascii=False
    )


    common_metadata = {

        "alignment_id": alignment_id,

        "source_id": str(source_id),

        "logical_section_id": str(logical_section_id),

        "standard": STANDARD,

        "term": str(term),

        "chapter": str(chapter),

        "unit": str(unit),

        "page": str(page),

        "block_type": str(block_type),

        "source_file": file_name,

        "language_pair": "english-tamil",

        "original_json": original_json
    }


    english_metadata = common_metadata.copy()

    english_metadata["language"] = "english"


    tamil_metadata = common_metadata.copy()

    tamil_metadata["language"] = "tamil"


    return {

        "alignment_id": alignment_id,

        "english_text": english_text,

        "tamil_text": tamil_text,

        "english_embedding": english_embedding,

        "tamil_embedding": tamil_embedding,

        "english_metadata": english_metadata,

        "tamil_metadata": tamil_metadata
    }


# ============================================================
# MAIN PROCESSING
# ============================================================

print("=" * 70)
print("SEARCHING FOR ALIGNED JSON FILES")
print("=" * 70)

json_files = find_json_files(
    INPUT_FOLDER
)


print()
print(
    f"Found {len(json_files)} aligned JSON files."
)
print()


if len(json_files) == 0:

    print(
        "ERROR: No *_aligned.json files found."
    )

    print()
    print(
        "Check INPUT_FOLDER:"
    )

    print(
        INPUT_FOLDER
    )

    raise SystemExit


total_files = 0
total_records = 0
successful_records = 0
skipped_records = 0

english_count = 0
tamil_count = 0


for file_number, file_path in enumerate(
    json_files,
    start=1
):

    total_files += 1

    print()
    print("=" * 70)

    print(
        f"FILE {file_number}/{len(json_files)}"
    )

    print(
        os.path.basename(file_path)
    )

    print("=" * 70)


    try:

        json_data = load_json_file(
            file_path
        )

    except Exception as e:

        print(
            f"ERROR loading file: {e}"
        )

        continue


    records = get_records(
        json_data
    )


    print(
        f"Records found: {len(records)}"
    )


    for record_index, record in enumerate(
        records,
        start=1
    ):

        total_records += 1

        result = process_record(
            record,
            file_path,
            record_index
        )


        if result is None:

            skipped_records += 1

            continue


        alignment_id = result[
            "alignment_id"
        ]


        english_text = result[
            "english_text"
        ]

        english_embedding = result[
            "english_embedding"
        ]

        english_metadata = result[
            "english_metadata"
        ]


        tamil_text = result[
            "tamil_text"
        ]

        tamil_embedding = result[
            "tamil_embedding"
        ]

        tamil_metadata = result[
            "tamil_metadata"
        ]


        if english_embedding is not None:

            english_collection.add(

                ids=[
                    alignment_id
                ],

                documents=[
                    english_text
                ],

                embeddings=[
                    english_embedding
                ],

                metadatas=[
                    english_metadata
                ]
            )

            english_count += 1


        if tamil_embedding is not None:

            tamil_collection.add(

                ids=[
                    alignment_id
                ],

                documents=[
                    tamil_text
                ],

                embeddings=[
                    tamil_embedding
                ],

                metadatas=[
                    tamil_metadata
                ]
            )

            tamil_count += 1


        print()
        print(
            f"Record {record_index}"
        )

        print(
            f"Alignment ID : {alignment_id}"
        )

        print(
            f"Section ID   : {result['english_metadata']['logical_section_id'] or '(none found)'}"
        )

        print(
            f"English chars: {len(english_text)}"
        )

        print(
            f"Tamil chars  : {len(tamil_text)}"
        )

        print(
            f"English emb  : "
            f"{len(english_embedding) if english_embedding else 'MISSING'}"
        )

        print(
            f"Tamil emb    : "
            f"{len(tamil_embedding) if tamil_embedding else 'MISSING'}"
        )


        successful_records += 1


print()
print()
print("=" * 70)
print("CHROMADB STORAGE COMPLETED")
print("=" * 70)

print()
print(
    f"JSON files processed       : {total_files}"
)

print(
    f"Total records found        : {total_records}"
)

print(
    f"Successful records         : {successful_records}"
)

print(
    f"Skipped records            : {skipped_records}"
)

print()
print(
    f"English embeddings stored  : {english_count}"
)

print(
    f"Tamil embeddings stored    : {tamil_count}"
)

print()
print(
    f"ChromaDB location:"
)

print(
    CHROMA_FOLDER
)

print()
print(
    f"English collection:"
)

print(
    ENGLISH_COLLECTION_NAME
)

print()
print(
    f"Tamil collection:"
)

print(
    TAMIL_COLLECTION_NAME
)

print()
print("=" * 70)


print()
print("=" * 70)
print("DATABASE VERIFICATION")
print("=" * 70)

print()

print(
    "English collection count:",
    english_collection.count()
)

print(
    "Tamil collection count:",
    tamil_collection.count()
)

print()


if english_collection.count() > 0:

    print("=" * 70)
    print("SAMPLE ENGLISH RECORD")
    print("=" * 70)

    sample = english_collection.get(
        limit=1,
        include=[
            "documents",
            "metadatas",
            "embeddings"
        ]
    )

    print()

    print(
        "ID:"
    )

    print(
        sample["ids"][0]
    )

    print()

    print(
        "English text:"
    )

    print(
        sample["documents"][0]
    )

    print()

    print(
        "Metadata:"
    )

    print(
        json.dumps(
            sample["metadatas"][0],
            ensure_ascii=False,
            indent=2
        )
    )

    print()

    if sample["embeddings"] is not None and len(sample["embeddings"]) > 0:

        print(
            "Embedding dimension:",
            len(sample["embeddings"][0])
        )

    print()


if tamil_collection.count() > 0:

    print("=" * 70)
    print("SAMPLE TAMIL RECORD")
    print("=" * 70)

    sample = tamil_collection.get(
        limit=1,
        include=[
            "documents",
            "metadatas",
            "embeddings"
        ]
    )

    print()

    print(
        "ID:"
    )

    print(
        sample["ids"][0]
    )

    print()

    print(
        "Tamil text:"
    )

    print(
        sample["documents"][0]
    )

    print()

    print(
        "Metadata:"
    )

    print(
        json.dumps(
            sample["metadatas"][0],
            ensure_ascii=False,
            indent=2
        )
    )

    print()

    if sample["embeddings"] is not None and len(sample["embeddings"]) > 0:
        print(
            "Embedding dimension:",
            len(sample["embeddings"][0])
        )

    print()


print("=" * 70)
print("DONE")
print("=" * 70)