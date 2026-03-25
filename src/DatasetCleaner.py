import re

def clean_dataset(input_file, output_file):
    """
    Cleans dataset by removing file paths and keeping only date + values.
    Automatically detects 'E' (Envelope) or 'V' (Vibracao) headers.
    Removes duplicate sets if they are completely identical.
    """
    cleaned_lines = []
    current_header = None
    current_block = []
    seen_blocks = set()

    def save_block():
        """Save block if it's unique"""
        if current_block:
            block_str = "\n".join(current_block)
            if block_str not in seen_blocks:
                seen_blocks.add(block_str)
                cleaned_lines.extend(current_block)

    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Detect headers (E or V)
            if re.match(r"^\s*E\b", line, re.IGNORECASE):
                save_block()
                current_header = "E"
                current_block = [current_header]
                continue
            elif re.match(r"^\s*V\b", line, re.IGNORECASE):
                save_block()
                current_header = "V"
                current_block = [current_header]
                continue

            # Match lines starting with a date (dd/mm/yyyy)
            match = re.match(r"(\d{2}/\d{2}/\d{4})\s+([\d,]+)\s+([\d,]+)", line)
            if match and current_header:
                date, val1, val2 = match.groups()
                current_block.append(f"{date}\t{val1}\t{val2}")

    # Save last block
    save_block()

    # Write cleaned output
    with open(output_file, "w", encoding="utf-8") as f:
        for l in cleaned_lines:
            f.write(l + "\n")

# Example usage:
# clean_dataset("Dataset.txt", "CleanedDataset.txt")

if __name__ == "__main__":
    clean_dataset("data/Dataset.txt", "data/CleanedDataset.txt")