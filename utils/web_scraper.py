import requests
from bs4 import BeautifulSoup
from typing import List
import time

class WebScraper:
    def __init__(self, urls_file: str):
        self.urls_file = urls_file
    
    def read_urls(self) -> List[str]:
        with open(self.urls_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    def scrape_url(self, url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text()
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return ""
    
    def scrape_all_urls(self) -> List[str]:
        scraped_content = []
        urls = self.read_urls()
        for url in urls:
            content = self.scrape_url(url)
            if content:
                scraped_content.append(content)
            time.sleep(1)  # Be nice to servers
        return scraped_content 