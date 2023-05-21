import argparse
from skeleteon_adding import transfer_video


def main() -> None:
    parser = argparse.ArgumentParser(description='Script to process input and output file paths.')
    # Add the --input flag
    parser.add_argument('--input', type=str, help='Input file path')
    # Add the --output flag
    parser.add_argument('--output', type=str, help='Output file path')

    # add the --type flag
    parser.add_argument('--type', type=str, help='Output file path')

    # Parse the command-line arguments
    args = parser.parse_args()

    input_file: str = args.input
    output_file: str = args.output
    type_: str = args.type
    transfer_video(input_file, output_file, type_)

if __name__ == "__main__":
    main()