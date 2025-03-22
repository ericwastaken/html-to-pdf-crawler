# HTML to PDF Crawler

A Python tool that crawls websites and converts HTML pages to PDF documents. This tool can recursively crawl a website, starting from a given URL, and convert all pages within the same domain to PDF files.

## Features

- Crawl websites recursively, staying within the same domain
- Convert HTML pages to PDF documents with customizable orientation
- Concurrent processing for efficient crawling and conversion
- Option to merge all PDFs into a single document
- Detailed logging capabilities
- Support for both online URLs and local HTML files

## Requirements

- Python 3.12 or higher
- Dependencies (automatically installed by the bootstrap script):
  - weasyprint: HTML to PDF conversion
  - beautifulsoup4: HTML parsing
  - requests: HTTP requests
  - tqdm: Progress bars
  - PyPDF2: PDF manipulation (for merging PDFs)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/ericwastaken/html-to-pdf-crawler.git
   cd html_to_pdf_crawler
   ```

2. Run the bootstrap script to set up the Python environment:
   ```
   ./x_bootstrap.sh
   ```

   The bootstrap script will:
   - Check for Python 3.12+ installation
   - Create a virtual environment (.venv)
   - Install all required dependencies

## Usage

After setting up the environment, you can run the tool with various options:

### Basic Usage

```bash
./html_to_pdf_crawler.py --startUrl https://example.com --outputDirectory ./output
```

### Advanced Options

```bash
./html_to_pdf_crawler.py \
  --startUrl https://example.com \
  --outputDirectory ./output \
  --orientation landscape \
  --logLevel INFO \
  --logFile ./logs/crawl.log \
  --autoConfirm \
  --concurrentProcessors 5 \
  --mergePDF
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--startUrl` | Starting URL or local HTML file | (Required) |
| `--outputDirectory` | Directory to store output PDFs | (Required) |
| `--orientation` | PDF orientation (portrait or landscape) | portrait |
| `--logLevel` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |
| `--logFile` | Path to log file | (None) |
| `--autoConfirm` | Automatically confirm PDF generation without prompting | (False) |
| `--concurrentProcessors` | Number of concurrent PDF processors (1-10) | 3 |
| `--showWeasyprintErrors` | Show Weasyprint errors in the console | (False) |
| `--mergePDF` | Merge all generated PDFs into one file named _merged.pdf | (False) |

## Examples

### Convert a Single Page

```bash
./html_to_pdf_crawler.py --startUrl https://example.com/page.html --outputDirectory ./output --autoConfirm
```

### Crawl an Entire Website and Convert to PDFs

```bash
./html_to_pdf_crawler.py --startUrl https://example.com --outputDirectory ./output --autoConfirm
```

### Convert Local HTML Files

```bash
./html_to_pdf_crawler.py --startUrl ./local/index.html --outputDirectory ./output --autoConfirm
```

### Create a Single Merged PDF from All Pages

```bash
./html_to_pdf_crawler.py --startUrl https://example.com --outputDirectory ./output --autoConfirm --mergePDF
```

## Dependencies Explained

- **weasyprint**: A Python library that renders HTML documents to PDF. It's the core component for the HTML to PDF conversion.
- **beautifulsoup4**: A library for parsing HTML and XML documents, used for extracting links and processing HTML content.
- **requests**: A simple HTTP library for making requests to fetch web pages.
- **tqdm**: A fast, extensible progress bar for Python, used to display progress during crawling and conversion.
- **PyPDF2**: A pure-Python library for PDF document manipulation, used for merging PDFs.

## Limitations

- **No JavaScript Support**: HTML and CSS are rendered without JavaScript. Websites that require JavaScript for proper formatting will not be formatted correctly.
- **Performance on Large Websites**: Crawling and generating PDFs for very large websites can take a significant amount of time.
- **Memory Usage with mergePDF**: The `--mergePDF` option might hang or consume excessive memory when processing very large websites with many pages. It's recommended to avoid using this option for very large websites.

## Contributing

Contributions are welcome! Please contribute to this project by forking the repository and creating a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [WeasyPrint](https://weasyprint.org/) for the HTML to PDF conversion capabilities
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing
- All contributors who have helped improve this tool
