import re

def clean_dataset(input_file, output_file):
    """
    Cleans dataset by removing file paths and keeping only date + values.
    Automatically detects 'E' (Envelope) or 'V' (Vibracao) headers.
    """
    cleaned_lines = []
    current_header = None

    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Detect headers (E or V) in the text
            if re.match(r"^\s*E\b", line, re.IGNORECASE):
                current_header = "E"
                cleaned_lines.append(current_header)
                continue
            elif re.match(r"^\s*V\b", line, re.IGNORECASE):
                current_header = "V"
                cleaned_lines.append(current_header)
                continue

            # Match lines starting with a date (dd/mm/yyyy)
            match = re.match(r"(\d{2}/\d{2}/\d{4})\s+([\d,]+)\s+([\d,]+)", line)
            if match and current_header:
                date, val1, val2 = match.groups()
                cleaned_lines.append(f"{date}\t{val1}\t{val2}")

    # Write cleaned output
    with open(output_file, "w", encoding="utf-8") as f:
        for l in cleaned_lines:
            f.write(l + "\n")

# Example usage:
# clean_dataset("Dataset.txt", "CleanedDataset.txt")


if __name__ == "__main__":
    clean_dataset("data/Dataset.txt", "data/CleanedDataset.txt")