#!/usr/bin/env python3

import os
import sys

# Check if running in virtual environment
venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
if sys.platform == "win32":
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")

# If not running inside the venv, restart the script using the venv's Python
if sys.executable != venv_python and os.path.exists(venv_python):
    os.execv(venv_python, [venv_python] + sys.argv)
elif sys.executable != venv_python and not os.path.exists(venv_python):
    print("❌ Error: Virtual environment not found.")
    print("Please run x_bootstrap.sh to setup the Python environment for this script.")
    sys.exit(1)

import argparse
import logging
import urllib.parse
import time
import concurrent.futures
import csv
import io
import signal
from collections import deque
from bs4 import BeautifulSoup
# Configure logging before importing weasyprint
logging.getLogger('weasyprint').addHandler(logging.NullHandler())
logging.getLogger('weasyprint').setLevel(logging.CRITICAL + 1)
logging.getLogger('weasyprint').propagate = False
logging.getLogger('weasyprint.progress').addHandler(logging.NullHandler())
logging.getLogger('weasyprint.progress').setLevel(logging.CRITICAL + 1)
logging.getLogger('weasyprint.progress').propagate = False
import weasyprint
from weasyprint import HTML, CSS
from urllib.parse import urljoin, urlparse
from pathlib import Path
import requests
from tqdm import tqdm

# Global flag to signal threads to stop
stop_processing = False

# Custom CSV formatter for logs
class CSVFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()
        # Use QUOTE_MINIMAL and set escapechar to handle special characters
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        self.seq = 0

    def format(self, record):
        self.seq += 1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        # Let the CSV writer handle quoting automatically
        message = record.getMessage()
        self.writer.writerow([timestamp, self.seq, record.levelname, message])
        result = self.output.getvalue().strip()
        self.output.truncate(0)
        self.output.seek(0)
        return result


# Two lines above each function
def is_same_domain(base_url, test_url):
    base = urlparse(base_url)
    test = urlparse(test_url)
    return (test.scheme in ["http", "https", "file"] and
            (base.scheme == test.scheme and base.netloc == test.netloc))


# Two lines above each function
def normalize_url(base_url, href):
    # If the href is just a bookmark (#), return the base URL without the fragment
    if href.startswith('#'):
        base_parsed = urlparse(base_url)
        # Return the base URL without any fragment
        return f"{base_parsed.scheme}://{base_parsed.netloc}{base_parsed.path}{base_parsed.params}{base_parsed.query}"

    joined = urljoin(base_url, href)
    parsed = urlparse(joined)

    # Normalize the path: remove trailing slash if it's not the root path
    path = parsed.path
    if path.endswith('/') and path != '/':
        path = path[:-1]

    # Normalize query parameters: sort them to ensure consistent order
    query = ''
    if parsed.query:
        query_params = urllib.parse.parse_qsl(parsed.query)
        query_params.sort()
        query = '?' + urllib.parse.urlencode(query_params)

    # Build the normalized URL without fragment
    normalized_url = f"{parsed.scheme}://{parsed.netloc}{path}{parsed.params}{query}"

    return normalized_url


# Two lines above each function
def fetch_html(url):
    try:
        if url.startswith("file://"):
            local_path = urllib.parse.unquote(urlparse(url).path)
            with open(local_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None


# Two lines above each function
def slugify_url(url, start_url):
    parsed = urlparse(url)
    start_parsed = urlparse(start_url)

    # If the URL is the same as the startUrl, use "index"
    if parsed.path == start_parsed.path:
        return "index"

    # Extract the relative path from the URL path
    start_path = urllib.parse.unquote(start_parsed.path)
    current_path = urllib.parse.unquote(parsed.path)

    # Extract the directory part of the start path
    start_dir = os.path.dirname(start_path)

    # If the current path starts with the start directory, extract the relative path
    if current_path.startswith(start_dir):
        relative_path = current_path[len(start_dir):].lstrip('/')
        # Replace slashes with hyphens to flatten the path
        flattened_path = relative_path.replace('/', '-')
        return flattened_path or "index"

    # Fallback to just the file name
    file_name = os.path.basename(parsed.path)
    return file_name or "index"


# Two lines above each function
def convert_to_pdf(url, output_path, orientation='portrait'):
    try:
        # Set the page orientation using CSS
        css_string = f'@page {{ size: {orientation}; }}'
        css = CSS(string=css_string)

        # No need to modify weasyprint logging here
        # The main() function already configures it properly

        # Create the HTML object first to handle potential errors
        try:
            html = HTML(url)
        except Exception as e:
            logging.error(f"Failed to create HTML object for {url}: {e}")
            return False

        # Create a temporary file path for the initial PDF
        temp_path = output_path + ".temp"

        # Then try to write the PDF
        try:
            # Generate the PDF with WeasyPrint
            # Note: We're using optimize_images=False because it might make files larger
            html.write_pdf(temp_path, stylesheets=[css], optimize_images=False)

            # Rename the temp file
            os.replace(temp_path, output_path)
            logging.info(f"Saved PDF: {output_path} in {orientation} orientation")

            return True
        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            logging.error(f"Failed to write PDF for {url}: {e}")
            return False
    except Exception as e:
        logging.error(f"Error converting {url} to PDF: {e}")
        return False


# Two lines above each function
def crawl_site(start_url, concurrent_processors=3):
    """
    Crawl the site starting from start_url and collect all URLs in the same domain.
    Uses concurrent processing for efficient crawling.
    Returns a list of tuples (url, slug) for all pages found.
    """
    global stop_processing
    import threading
    from queue import Queue, Empty

    # Shared data structures with thread-safe access
    visited = set()
    visited_lock = threading.Lock()
    url_queue = Queue()
    url_queue.put(start_url)
    url_info = []
    url_info_lock = threading.Lock()
    active_workers = threading.Semaphore(0)
    threads_running = threading.Event()
    threads_running.set()

    # Create a progress bar for console output
    pbar = tqdm(desc="Crawling URLs", unit="url")

    logging.info(f"Starting crawl from {start_url} with {concurrent_processors} concurrent processors")

    def worker():
        global stop_processing
        while threads_running.is_set() and not stop_processing:
            try:
                # Get a URL from the queue
                current_url = url_queue.get(block=False)

                # Check if URL has already been visited
                with visited_lock:
                    if current_url in visited:
                        url_queue.task_done()
                        continue
                    visited.add(current_url)

                logging.info(f"Crawling: {current_url}")

                # Fetch and process the HTML content
                html_content = fetch_html(current_url)
                if html_content is None:
                    url_queue.task_done()
                    # Update progress bar even if fetch fails
                    pbar.update(1)
                    continue

                # Add URL to the results
                slug = slugify_url(current_url, start_url)
                with url_info_lock:
                    url_info.append((current_url, slug))

                # Update progress bar
                pbar.update(1)

                # Find and queue new URLs
                soup = BeautifulSoup(html_content, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = normalize_url(current_url, link['href'])
                    with visited_lock:
                        if is_same_domain(start_url, href) and href not in visited:
                            url_queue.put(href)
                            active_workers.release()  # Signal that there's work to do

                url_queue.task_done()
            except Empty:
                # No more URLs in the queue, try to get a permit to continue
                try:
                    # Wait for a short time for new URLs to be added
                    active_workers.acquire(timeout=0.1)
                    # If we got here, there's more work to do
                    continue
                except threading.BrokenBarrierError:
                    # Timeout occurred, no new URLs were added
                    break
                except:
                    # Any other error, exit the worker
                    break

            # Check if we should stop processing
            if stop_processing:
                logging.info("Worker thread stopping due to CTRL+C")
                break

    # Start worker threads
    threads = []
    for _ in range(concurrent_processors):
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        threads.append(thread)
        active_workers.release()  # Initial permit for each worker

    try:
        # Wait for all URLs to be processed or stop flag to be set
        while not stop_processing:
            # Check if all tasks are done
            if url_queue.empty() and url_queue.unfinished_tasks == 0:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        # Handle KeyboardInterrupt here to prevent it from propagating
        threads_running.clear()
        logging.info("CTRL+C detected. Stopping crawler threads...")
        stop_processing = True

    # Signal threads to stop
    threads_running.clear()

    # Wait for all threads to finish
    for thread in threads:
        thread.join(timeout=1.0)

    # Close the progress bar
    pbar.close()

    # De-duplicate URLs based on normalized URL
    seen_urls = {}
    unique_url_info = []
    for url, slug in url_info:
        if url not in seen_urls:
            seen_urls[url] = True
            unique_url_info.append((url, slug))

    original_count = len(url_info)
    unique_count = len(unique_url_info)
    if original_count > unique_count:
        logging.info(f"De-duplicated URLs: removed {original_count - unique_count} duplicate URLs.")

    logging.info(f"Crawl completed. Found {unique_count} unique pages.")
    return unique_url_info


# Two lines above each function
def process_pdf(idx, url_data, output_dir, orientation, total_pdfs):
    try:
        url, slug = url_data
        # Calculate the number of digits needed for zero padding
        padding = len(str(total_pdfs))
        pdf_name = f"{idx+1:0{padding}d}-{slug}.pdf"
        output_path = os.path.join(output_dir, pdf_name)
        # Log to file but not to console (progress bar will show this info)
        logging.info(f"Processing document {idx+1} of {total_pdfs} - {slug}")
        success = convert_to_pdf(url, output_path, orientation)
        if success:
            return output_path
        else:
            logging.warning(f"Skipping document {idx+1} of {total_pdfs} - {slug} due to conversion failure")
            return None
    except Exception as e:
        logging.error(f"Error processing document {idx+1}: {e}")
        return None


# Two lines above each function
def generate_pdfs(url_info, output_dir, orientation='portrait', concurrent_processors=3, merge_pdf=False):
    """
    Generate PDFs for all URLs in url_info.
    url_info is a list of tuples (url, slug).
    """
    global stop_processing

    os.makedirs(output_dir, exist_ok=True)
    total_pdfs = len(url_info)

    logging.info(f"Generating {total_pdfs} PDFs with {concurrent_processors} concurrent processors")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=concurrent_processors) as executor:
        # Submit all tasks
        future_to_idx = {}
        for idx, url_data in enumerate(url_info):
            if stop_processing:
                break
            future = executor.submit(process_pdf, idx, url_data, output_dir, orientation, total_pdfs)
            future_to_idx[future] = idx

        # Create a progress bar for console output
        with tqdm(total=total_pdfs, desc="Generating PDFs", unit="pdf") as pbar:
            try:
                for future in concurrent.futures.as_completed(future_to_idx):
                    if stop_processing:
                        # Cancel all pending futures
                        for f in future_to_idx:
                            if not f.done():
                                f.cancel()
                        break

                    try:
                        idx = future_to_idx[future]
                        url, slug = url_info[idx]
                        # Update progress bar description with current document
                        pbar.set_description(f"Processing {idx+1}/{total_pdfs}: {slug}")

                        result = future.result()
                        if result is not None:
                            results.append(result)
                        # Update progress bar regardless of success or failure
                        pbar.update(1)
                    except Exception as exc:
                        idx = future_to_idx[future]
                        logging.error(f"PDF generation failed for {url_info[idx][0]}: {exc}")
                        # Update progress bar even if an exception occurs
                        pbar.update(1)
            except KeyboardInterrupt:
                logging.info("CTRL+C detected. Stopping PDF generation processes...")
                stop_processing = True
                # Cancel all pending futures
                for f in future_to_idx:
                    if not f.done():
                        f.cancel()
                # Shutdown the executor
                executor.shutdown(wait=False)

    if stop_processing:
        logging.info(f"PDF generation stopped. Generated {len(results)} PDFs out of {total_pdfs} requested.")

    # Merge PDFs if requested
    if merge_pdf and results:
        try:
            import PyPDF2

            # Sort results by filename to ensure correct order
            results.sort()

            merged_pdf_path = os.path.join(output_dir, "_merged.pdf")
            logging.info(f"Merging {len(results)} PDFs into {merged_pdf_path}")

            merger = PyPDF2.PdfMerger()

            # Add each PDF to the merger
            for pdf in results:
                merger.append(pdf)

            # Write the merged PDF
            with open(merged_pdf_path, "wb") as output_file:
                merger.write(output_file)

            logging.info(f"Successfully created merged PDF: {merged_pdf_path}")

        except Exception as e:
            logging.error(f"Error merging PDFs: {e}")

    return results


# Two lines above each function
def crawl_and_convert(start_url, output_dir, orientation='portrait', auto_confirm=False, concurrent_processors=3, merge_pdf=False):
    """
    First crawl the site to collect all URLs, then generate PDFs for all pages.
    Uses concurrent processing for both crawling and PDF generation.
    """
    global stop_processing

    # First pass: crawl and collect URLs
    url_info = crawl_site(start_url, concurrent_processors)

    # Check if we should stop processing
    if stop_processing:
        logging.info("Crawling stopped due to CTRL+C. Exiting without generating PDFs.")
        return

    total_pdfs = len(url_info)

    # Confirmation step
    if not auto_confirm:
        confirm = input(f"This will generate {total_pdfs} PDFs. Please confirm (y/n): ")
        if confirm.lower() != 'y':
            logging.info("Operation cancelled by user.")
            return

    # Second pass: generate PDFs
    try:
        results = generate_pdfs(url_info, output_dir, orientation, concurrent_processors, merge_pdf)

        # Check if we should stop processing
        if stop_processing:
            logging.info("PDF generation stopped due to CTRL+C.")
            return

        successful_pdfs = len(results)
        if successful_pdfs < total_pdfs:
            logging.warning(f"Generated {successful_pdfs} PDFs out of {total_pdfs} requested. {total_pdfs - successful_pdfs} PDFs failed to generate.")
        else:
            logging.info(f"Successfully generated all {total_pdfs} PDFs.")
    except Exception as e:
        logging.error(f"Error during PDF generation: {e}")
        logging.info("Continuing with partial results...")


# Two lines above each function
def main():
    parser = argparse.ArgumentParser(description="Convert HTML and subpages to PDFs recursively.")
    parser.add_argument('--startUrl', required=True, help='Starting URL or local HTML file')
    parser.add_argument('--outputDirectory', required=True, help='Directory to store output PDFs')
    parser.add_argument('--orientation', choices=['portrait', 'landscape'], default='portrait', help='PDF orientation (default: portrait)')
    parser.add_argument('--logLevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--logFile', help='Path to log file (if not specified, no logs will be output)')
    parser.add_argument('--autoConfirm', action='store_true', help='Automatically confirm PDF generation without prompting')
    parser.add_argument('--concurrentProcessors', type=int, default=3, help='Number of concurrent PDF processors (1-10, default: 3)')
    parser.add_argument('--showWeasyprintErrors', action='store_true', help='Show Weasyprint errors in the console (default: False)')
    parser.add_argument('--mergePDF', action='store_true', help='Merge all generated PDFs into one file named _merged.pdf (default: False)')
    args = parser.parse_args()

    # Validate concurrentProcessors
    if args.concurrentProcessors < 1 or args.concurrentProcessors > 10:
        parser.error("--concurrentProcessors must be between 1 and 10")

    # Configure logging
    if args.logFile:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(args.logFile)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Use custom CSV formatter
        handler = logging.FileHandler(args.logFile, mode='w', encoding='utf-8')
        handler.setFormatter(CSVFormatter())

        # Configure the root logger
        logging.basicConfig(
            level=getattr(logging, args.logLevel),
            handlers=[handler]
        )

        # Apply handler to all known loggers
        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            logger.handlers = []
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, args.logLevel))
            logger.propagate = False

        # Configure weasyprint logger
        # Apply to weasyprint global LOGGER if present
        if hasattr(weasyprint, 'LOGGER'):
            weasyprint.LOGGER.handlers = []
            weasyprint.LOGGER.addHandler(handler)
            weasyprint.LOGGER.setLevel(getattr(logging, args.logLevel))
            weasyprint.LOGGER.propagate = False

        # Also configure the weasyprint logger directly as suggested
        weasyprint_logger = logging.getLogger('weasyprint')
        weasyprint_logger.handlers = []  # Remove the default stderr handler
        weasyprint_logger.addHandler(handler)
        weasyprint_logger.setLevel(getattr(logging, args.logLevel))
        weasyprint_logger.propagate = False

        # Configure the weasyprint.progress logger
        weasyprint_progress_logger = logging.getLogger('weasyprint.progress')
        weasyprint_progress_logger.handlers = []  # Remove the default stderr handler
        weasyprint_progress_logger.addHandler(handler)
        weasyprint_progress_logger.setLevel(getattr(logging, args.logLevel))
        weasyprint_progress_logger.propagate = False
    else:
        # Disable all logging by setting handlers to NullHandler
        null_handler = logging.NullHandler()

        logging.basicConfig(
            level=logging.CRITICAL + 1,
            handlers=[null_handler]
        )

        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            logger.handlers = [null_handler]
            logger.setLevel(logging.CRITICAL + 1)
            logger.propagate = False

        # Configure weasyprint loggers based on showWeasyprintErrors flag
        if args.showWeasyprintErrors:
            # Show weasyprint errors in the console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

            if hasattr(weasyprint, 'LOGGER'):
                weasyprint.LOGGER.handlers = [console_handler]
                weasyprint.LOGGER.setLevel(getattr(logging, args.logLevel))
                weasyprint.LOGGER.propagate = False

            # Configure the weasyprint logger directly
            weasyprint_logger = logging.getLogger('weasyprint')
            weasyprint_logger.handlers = [console_handler]
            weasyprint_logger.setLevel(getattr(logging, args.logLevel))
            weasyprint_logger.propagate = False

            # Configure the weasyprint.progress logger
            weasyprint_progress_logger = logging.getLogger('weasyprint.progress')
            weasyprint_progress_logger.handlers = [console_handler]
            weasyprint_progress_logger.setLevel(getattr(logging, args.logLevel))
            weasyprint_progress_logger.propagate = False
        else:
            # Suppress weasyprint errors
            if hasattr(weasyprint, 'LOGGER'):
                weasyprint.LOGGER.handlers = [null_handler]
                weasyprint.LOGGER.setLevel(logging.CRITICAL + 1)
                weasyprint.LOGGER.propagate = False

            # Configure the weasyprint logger directly
            weasyprint_logger = logging.getLogger('weasyprint')
            weasyprint_logger.handlers = [null_handler]
            weasyprint_logger.setLevel(logging.CRITICAL + 1)
            weasyprint_logger.propagate = False

            # Configure the weasyprint.progress logger
            weasyprint_progress_logger = logging.getLogger('weasyprint.progress')
            weasyprint_progress_logger.handlers = [null_handler]
            weasyprint_progress_logger.setLevel(logging.CRITICAL + 1)
            weasyprint_progress_logger.propagate = False

    start_url = args.startUrl
    if not urllib.parse.urlparse(start_url).scheme:
        # Assume local file
        abs_path = Path(start_url).resolve()
        start_url = f"file://{abs_path}"

    output_dir = args.outputDirectory
    orientation = args.orientation
    auto_confirm = args.autoConfirm
    concurrent_processors = args.concurrentProcessors

    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        global stop_processing
        if not stop_processing:
            stop_processing = True
            print("\nCTRL+C detected. Stopping all threads and processes...")
            logging.info("CTRL+C detected. Stopping all threads and processes...")
        else:
            print("\nSecond CTRL+C detected. Forcing exit...")
            logging.info("Second CTRL+C detected. Forcing exit...")
            sys.exit(1)

    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    try:
        crawl_and_convert(start_url, output_dir, orientation, auto_confirm, concurrent_processors, args.mergePDF)
    except KeyboardInterrupt:
        # This should be handled by the signal handler, but just in case
        if not stop_processing:
            stop_processing = True
            print("\nCTRL+C detected. Stopping all threads and processes...")
            logging.info("CTRL+C detected. Stopping all threads and processes...")
            # Give some time for threads to stop
            time.sleep(1)
        sys.exit(0)
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
