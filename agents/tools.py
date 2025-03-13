import arxiv
import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from googlesearch import search  # pip install googlesearch-python

def search_academic_papers(query, max_results=5):
    """
    Search for academic papers on arXiv based on a query.
    Returns a list of dictionaries containing:
      - title
      - authors (as a list)
      - summary
      - pdf_url
    """
    search_obj = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for result in search_obj.results():
        papers.append({
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'summary': result.summary,
            'pdf_url': result.pdf_url
        })
    return papers

def download_pdf(pdf_url, filename):
    """
    Download the PDF from the given URL and save it to a file.
    Returns the filename if successful.
    """
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded PDF to {filename}")
        return filename
    else:
        raise Exception("Failed to download PDF")

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyMuPDF.
    Returns the full text extracted from the PDF.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def scrape_webpage(url):
    """
    Scrape text content from a webpage using Requests and BeautifulSoup.
    Returns the cleaned text extracted from the page.
    """
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    # response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve webpage: {url}")
    
    soup = BeautifulSoup(response.content, "html.parser")
    # Remove script and style elements
    for element in soup(["script", "style"]):
        element.decompose()
    
    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    clean_text = "\n".join(line for line in lines if line)
    return clean_text

def search_top_links(query, num_results=5):
    """
    Perform a Google search for the query and return the top num_results URLs.
    """
    links = []
    for url in search(query, num_results=num_results):
        links.append(url)
    return links

def get_web_content_from_query(query, num_results=5, filename='./extraction_result.txt'):
    """
    Search for top links based on the query and extract text content from each link.
    Returns a list of dictionaries with 'url' and 'text'.
    """
    links = search_top_links(query, num_results)
    content_list = []
    for link in links:
        try:
            text = scrape_webpage(link)
            content_list.append({'url': link, 'text': text})
        except Exception as e:
            print(f"Error scraping {link}: {e}")
            
        with open(filename, "a", encoding="utf-8") as f:
            f.write('='*100+'\n')
            f.write(f"Query: {query}\n")
            # for item in web_results:
            f.write(f"URL: {link}\n")
            snippet = text[:100].replace("\n", " ")
            f.write(f"Snippet: {snippet}...\n\n")
            f.write("="*80 + "\n\n")
    return content_list

def log_extraction_results(query, pdf_results, web_results, filename="./extraction_results.txt"):
    """
    Log the query along with details of the PDFs and web links extracted.
    pdf_results is expected to be a list of dictionaries with keys: title, pdf_url, filename.
    web_results is expected to be a list of dictionaries with keys: url, text.
    """
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"Query: {query}\n")
        f.write("PDFs Extracted:\n")
        for item in pdf_results:
            f.write(f"Title: {item['title']}\n")
            f.write(f"PDF URL: {item['pdf_url']}\n")
            f.write(f"Local File: {item['filename']}\n\n")
        f.write("Web Links Scraped:\n")
        for item in web_results:
            f.write(f"URL: {item['url']}\n")
            snippet = item['text'][:100].replace("\n", " ")
            f.write(f"Snippet: {snippet}...\n\n")
        f.write("="*80 + "\n\n")

# Example usage when run as a script
if __name__ == '__main__':
    # Define your query
    query = "deep learning natural language processing"
    
    # --- Academic Paper Extraction ---
    pdf_results = []
    papers = search_academic_papers(query, max_results=2)
    for idx, paper in enumerate(papers, start=1):
        print(f"\nPaper {idx}:")
        print("Title:", paper['title'])
        print("PDF URL:", paper['pdf_url'])
        pdf_filename = f"paper_{idx}.pdf"
        try:
            download_pdf(paper['pdf_url'], pdf_filename)
            # Optionally extract text (if needed)
            extracted_text = extract_text_from_pdf(pdf_filename)
            pdf_results.append({
                'title': paper['title'],
                'pdf_url': paper['pdf_url'],
                'filename': pdf_filename
            })
        except Exception as e:
            print(f"Error processing paper {idx}: {e}")
    
    # --- Web Content Extraction ---
    web_results = get_web_content_from_query(query, num_results=10)
    for item in web_results:
        print(f"\nScraped content from {item['url']} (first 300 characters):")
        print(item['text'][:300])
    
    # Log the results in a text file
    log_extraction_results(query, pdf_results, web_results)
    print("\nExtraction results logged in extraction_results.txt")
