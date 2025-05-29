import csv
import re
import argparse
from pathlib import Path

def extract_arxiv_id(text: str) -> str | None:
    """
    Extracts arXiv ID from a given text string.
    The pattern looks for 'arXiv:YYYY.XXXXX' and captures 'YYYY.XXXXX'.
    """
    match = re.search(r'arXiv:(\d{4}\.\d{5})', text)
    if match:
        return match.group(1)
    return None

def generate_preference_csv(input_csv_path: Path, output_csv_path: Path, preference_value: str) -> None:
    """
    Reads a Zotero CSV, extracts arXiv IDs, and generates a preference CSV.
    All extracted IDs will have a 'like' preference.
    """
    arxiv_ids = set() # Use a set to store unique arXiv IDs

    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        # Assuming 'Extra' column contains the arXiv ID in the format 'arXiv:YYYY.XXXXX'
        # Or 'Publication Title' might contain 'arXiv preprint arXiv:YYYY.XXXXX'
        # Or 'Url' might contain 'http://arxiv.org/abs/YYYY.XXXXX'
        # We will prioritize 'Extra', then 'Url', then 'Publication Title'
        for row in reader:
            arxiv_id = None
            if 'Extra' in row and row['Extra']:
                arxiv_id = extract_arxiv_id(row['Extra'])
            
            if not arxiv_id and 'Url' in row and row['Url']:
                # Try to extract from URL if Extra didn't yield anything
                url_match = re.search(r'arxiv\.org/abs/(\d{4}\.\d{5})', row['Url'])
                if url_match:
                    arxiv_id = url_match.group(1)

            if not arxiv_id and 'Publication Title' in row and row['Publication Title']:
                # Try to extract from Publication Title if previous attempts failed
                pub_title_match = re.search(r'arXiv:(\d{4}\.\d{5})', row['Publication Title'])
                if pub_title_match:
                    arxiv_id = pub_title_match.group(1)
            
            if arxiv_id:
                arxiv_ids.add(arxiv_id)

    # Ensure the output directory exists
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['id', 'preference']) # Write header
        for arxiv_id in sorted(list(arxiv_ids)): # Sort for consistent output
            writer.writerow([arxiv_id, preference_value])

def main():
    parser = argparse.ArgumentParser(description="Extract arXiv IDs from Zotero CSV and generate a preference CSV.")
    parser.add_argument("input_csv", type=str, help="Path to the input Zotero CSV file (e.g., 'test/data/zotero.csv').")
    parser.add_argument("-o", "--output_csv", type=str, default="preference/init.csv", help="Path to the output preference CSV file. Defaults to 'preference/init.csv'.")
    parser.add_argument("-p", "--preference", type=str, default="like", help="The preference value to set for all extracted IDs. Defaults to 'like'.")

    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    preference_value = args.preference

    if not input_path.is_file():
        print(f"Error: Input CSV file not found at {input_path}")
        return

    try:
        generate_preference_csv(input_path, output_path, preference_value)
        print(f"Successfully generated preference CSV at {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
