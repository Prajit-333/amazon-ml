"""Quick, line-by-line sanity check for a TSV file before pandas loads it."""
import argparse


def diagnose(path, expected_cols=None, max_report=10):
    quote_lines = []
    bad_col_lines = []
    n_lines = 0
    header_cols = None
    cols = expected_cols

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as file:
        for line_number, line in enumerate(file):
            n_lines += 1
            if line_number == 0:
                header_cols = line.rstrip("\r\n").count("\t") + 1
                cols = expected_cols or header_cols
                continue
            if '"' in line and len(quote_lines) < max_report:
                quote_lines.append(line_number + 1)
            ncols = line.rstrip("\r\n").count("\t") + 1
            if ncols != cols and len(bad_col_lines) < max_report:
                bad_col_lines.append((line_number + 1, ncols))

    print(path)
    print(f"  lines (incl. header): {n_lines}")
    print(f"  header columns: {header_cols}")
    if quote_lines:
        print(f'  WARNING: found " characters in line(s): {quote_lines}')
        print("    -> read_tsv uses quoting=csv.QUOTE_NONE for these fields.")
    else:
        print('  no " characters found - safe from the quoting issue.')
    if bad_col_lines:
        print(f"  WARNING: rows with a different column count: {bad_col_lines}")
        print("    -> inspect these rows for unescaped tabs or malformed data.")
    else:
        print("  column counts consistent with header across all lines.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--expected-cols", type=int, default=None)
    args = parser.parse_args()
    diagnose(args.path, args.expected_cols)