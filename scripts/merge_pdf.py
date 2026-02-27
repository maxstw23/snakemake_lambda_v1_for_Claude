import pymupdf
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge PDF files')
    parser.add_argument('input_files', nargs='+', help='Input PDF files')
    parser.add_argument('output_file', help='Output PDF file')
    args = parser.parse_args()

    for i, file in enumerate(args.input_files):
        if i == 0:
            pdf = pymupdf.open(file)
        else:
            pdf.insert_pdf(pymupdf.open(file))

    pdf.save(args.output_file)