"""
News Extractor Module
Fetches financial news from NewsAPI and extracts article content using newspaper3k
"""
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from newspaper import Article
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class NewsExtractor:
    """Handles fetching news articles from NewsAPI and extracting content"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NewsExtractor with NewsAPI key
        
        Args:
            api_key: NewsAPI key. If None, will try to get from environment variable NEWSAPI_KEY
        """
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        if not self.api_key:
            raise ValueError("NewsAPI key is required. Set NEWSAPI_KEY environment variable or pass api_key parameter.")
        
        self.newsapi = NewsApiClient(api_key=self.api_key)
    
    def fetch_articles(self, 
                      query: str = "BTC",
                      from_date: Optional[str] = None,
                      sort_by: str = "popularity",
                      page_size: int = 5) -> List[Dict]:
        """
        Fetch articles from NewsAPI
        
        Args:
            query: Search query (default: "BTC")
            from_date: Date in YYYY-MM-DD format. If None, defaults to yesterday
            sort_by: Sort order (popularity, relevancy, publishedAt)
            page_size: Number of articles to fetch (default: 5)
        
        Returns:
            List of article dictionaries with metadata
        """
        if from_date is None:
            from_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        try:
            response = self.newsapi.get_everything(
                q=query,
                from_param=from_date,
                sort_by=sort_by,
                page_size=page_size,
                language="en"
            )
            
            articles = response.get("articles", [])
            logger.info(f"Fetched {len(articles)} articles from NewsAPI")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching articles from NewsAPI: {e}")
            raise
    
    def extract_article_content(self, url: str) -> Optional[str]:
        """
        Extract full article content using newspaper3k
        
        Args:
            url: URL of the article to extract
        
        Returns:
            Extracted article text, or None if extraction fails
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text:
                logger.info(f"Successfully extracted content from {url} ({len(article.text)} characters)")
                return article.text
            else:
                logger.warning(f"No content extracted from {url}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def fetch_and_extract_articles(self,
                                   query: str = "BTC",
                                   from_date: Optional[str] = None,
                                   sort_by: str = "popularity",
                                   page_size: int = 5) -> List[Dict]:
        """
        Fetch articles from NewsAPI and extract their full content
        
        Args:
            query: Search query (default: "BTC")
            from_date: Date in YYYY-MM-DD format. If None, defaults to yesterday
            sort_by: Sort order (popularity, relevancy, publishedAt)
            page_size: Number of articles to fetch (default: 5)
        
        Returns:
            List of dictionaries containing article metadata and extracted content
            Each dict has: title, source, url, publishedAt, content
        """
        articles = self.fetch_articles(query, from_date, sort_by, page_size)
        
        extracted_articles = []
        for article in articles:
            url = article.get("url")
            if not url:
                logger.warning(f"Skipping article without URL: {article.get('title', 'Unknown')}")
                continue
            
            content = self.extract_article_content(url)
            if content:
                extracted_articles.append({
                    "title": article.get("title", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "url": url,
                    "publishedAt": article.get("publishedAt", ""),
                    "content": content
                })
            else:
                logger.warning(f"Failed to extract content from {url}, skipping")
        
        logger.info(f"Successfully extracted content from {len(extracted_articles)}/{len(articles)} articles")
        return extracted_articles


if __name__ == "__main__":
    # Example usage
    extractor = NewsExtractor()
    articles = extractor.fetch_and_extract_articles(query="BTC", page_size=5)
    
    for article in articles:
        print(f"\n{'='*80}")
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']}")
        print(f"URL: {article['url']}")
        print(f"Content length: {len(article['content'])} characters")
        print(f"{'='*80}\n")
        print(f"{article['content'][:1000]}")

