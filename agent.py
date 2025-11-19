"""
Agentic Flow for Financial News Analysis
Orchestrates multi-agent synthesis of news articles with context passing
"""
import os
import math
from typing import List, Dict, Optional
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Handle both relative and absolute imports
try:
    from .news_extractor import NewsExtractor
except ImportError:
    from news_extractor import NewsExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SynthesisAgent:
    """Individual agent that synthesizes a document section"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize synthesis agent
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def synthesize(self, 
                   content: str, 
                   previous_context: Optional[str] = None,
                   section_number: int = 1,
                   total_sections: int = 1) -> str:
        """
        Synthesize and summarize a document section
        
        Args:
            content: The content to synthesize (should be <= 100k characters)
            previous_context: Summary from previous sections (if any)
            section_number: Current section number (1-indexed)
            total_sections: Total number of sections
        
        Returns:
            Synthesized summary of the content
        """
        # Build the prompt
        if previous_context:
            prompt = f"""You are a financial news analyst. Your task is to synthesize and summarize the following section of a financial news article.

CONTEXT FROM PREVIOUS SECTIONS:
{previous_context}

CURRENT SECTION TO ANALYZE (Section {section_number} of {total_sections}):
{content}

Please provide a comprehensive synthesis that:
1. Extracts key financial insights and information
2. Identifies important market movements, price changes, and trends
3. Highlights significant events, announcements, or data points
4. Notes any relevant metrics, percentages, or figures
5. If this is not the first section, integrate the new information with the context from previous sections
6. Maintains continuity with previous sections while adding new insights

Provide a clear, structured summary that captures the essential information."""
        else:
            prompt = f"""You are a financial news analyst. Your task is to synthesize and summarize the following section of a financial news article.

CURRENT SECTION TO ANALYZE (Section {section_number} of {total_sections}):
{content}

Please provide a comprehensive synthesis that:
1. Extracts key financial insights and information
2. Identifies important market movements, price changes, and trends
3. Highlights significant events, announcements, or data points
4. Notes any relevant metrics, percentages, or figures

Provide a clear, structured summary that captures the essential information."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert financial news analyst specializing in cryptocurrency and financial markets."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            synthesis = response.choices[0].message.content
            logger.info(f"Agent synthesized section {section_number}/{total_sections} ({len(content)} chars -> {len(synthesis)} chars)")
            return synthesis
            
        except Exception as e:
            logger.error(f"Error in synthesis agent: {e}")
            raise


class OrchestratorAgent:
    """Orchestrates multiple synthesis agents with context passing"""
    
    def __init__(self, 
                 news_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini"):
        """
        Initialize orchestrator
        
        Args:
            news_api_key: NewsAPI key
            openai_api_key: OpenAI API key
            model: OpenAI model to use
        """
        self.news_extractor = NewsExtractor(api_key=news_api_key)
        self.synthesis_agent = SynthesisAgent(api_key=openai_api_key, model=model)
        self.chunk_size = 100000  # 100k characters per chunk
    
    def split_document(self, content: str) -> List[str]:
        """
        Split document into chunks of 100k characters
        
        Args:
            content: Full document content
        
        Returns:
            List of document chunks
        """
        if len(content) <= self.chunk_size:
            return [content]
        
        chunks = []
        num_chunks = math.ceil(len(content) / self.chunk_size)
        
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, len(content))
            chunk = content[start_idx:end_idx]
            chunks.append(chunk)
        
        logger.info(f"Split document into {len(chunks)} chunks ({len(content)} total characters)")
        return chunks
    
    def synthesize_document(self, content: str) -> str:
        """
        Synthesize a document using multiple agents with context passing
        
        Args:
            content: Full document content
        
        Returns:
            Final synthesized summary
        """
        chunks = self.split_document(content)
        total_sections = len(chunks)
        
        if total_sections == 1:
            # Single section, no context passing needed
            logger.info("Document fits in single chunk, synthesizing directly")
            return self.synthesis_agent.synthesize(
                content=chunks[0],
                section_number=1,
                total_sections=1
            )
        
        # Multiple sections - need context passing
        previous_context = None
        all_syntheses = []
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing section {i}/{total_sections}")
            
            # Synthesize current section with previous context
            synthesis = self.synthesis_agent.synthesize(
                content=chunk,
                previous_context=previous_context,
                section_number=i,
                total_sections=total_sections
            )
            
            all_syntheses.append(synthesis)
            
            # Update context for next section
            # Combine all previous syntheses as context
            previous_context = "\n\n".join(all_syntheses)
        
        # Final synthesis combining all sections
        logger.info("Creating final synthesis from all sections")
        final_prompt = f"""You are a financial news analyst. Below are synthesized summaries from different sections of a financial news article. 

Please create a comprehensive final synthesis that:
1. Integrates all the key insights from each section
2. Eliminates redundancy while preserving all important information
3. Maintains a logical flow and structure
4. Highlights the most critical financial information, market movements, and insights
5. Provides a clear, concise summary suitable for decision-making

SECTION SUMMARIES:
{previous_context}

Provide the final comprehensive synthesis:"""
        
        try:
            response = self.synthesis_agent.client.chat.completions.create(
                model=self.synthesis_agent.model,
                messages=[
                    {"role": "system", "content": "You are an expert financial news analyst specializing in cryptocurrency and financial markets."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.3,
                max_tokens=3000
            )
            
            final_synthesis = response.choices[0].message.content
            logger.info(f"Final synthesis complete ({len(final_synthesis)} characters)")
            return final_synthesis
            
        except Exception as e:
            logger.error(f"Error creating final synthesis: {e}")
            # Fallback: return concatenated syntheses
            logger.warning("Falling back to concatenated syntheses")
            return previous_context
    
    def process_article(self, article: Dict) -> Dict:
        """
        Process a single article through the full pipeline
        
        Args:
            article: Article dictionary with title, source, url, content, etc.
        
        Returns:
            Article dictionary with added 'synthesis' field
        """
        logger.info(f"Processing article: {article.get('title', 'Unknown')}")
        
        content = article.get('content', '')
        if not content:
            logger.warning("Article has no content, skipping synthesis")
            article['synthesis'] = "No content available for synthesis."
            return article
        
        synthesis = self.synthesize_document(content)
        article['synthesis'] = synthesis
        
        return article
    
    def process_news_query(self,
                          query: str = "BTC",
                          from_date: Optional[str] = None,
                          sort_by: str = "popularity",
                          page_size: int = 5) -> List[Dict]:
        """
        Complete pipeline: fetch news, extract content, and synthesize
        
        Args:
            query: Search query (default: "BTC")
            from_date: Date in YYYY-MM-DD format. If None, defaults to yesterday
            sort_by: Sort order (popularity, relevancy, publishedAt)
            page_size: Number of articles to process (default: 5)
        
        Returns:
            List of articles with synthesized summaries
        """
        logger.info(f"Starting news processing pipeline for query: {query}")
        
        # Step 1: Fetch and extract articles
        articles = self.news_extractor.fetch_and_extract_articles(
            query=query,
            from_date=from_date,
            sort_by=sort_by,
            page_size=page_size
        )
        
        if not articles:
            logger.warning("No articles found or extracted")
            return []
        
        # Step 2: Process each article through synthesis
        processed_articles = []
        for i, article in enumerate(articles, 1):
            logger.info(f"Processing article {i}/{len(articles)}")
            try:
                processed_article = self.process_article(article)
                processed_articles.append(processed_article)
            except Exception as e:
                logger.error(f"Error processing article {i}: {e}")
                article['synthesis'] = f"Error during synthesis: {str(e)}"
                processed_articles.append(article)
        
        logger.info(f"Pipeline complete. Processed {len(processed_articles)} articles")
        return processed_articles


def main():
    """Example usage of the agentic flow"""
    import sys
    import argparse
    from pathlib import Path
    
    # Add parent directory to path for imports when running as script
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    # Re-import with absolute path if needed
    try:
        from agent.news_extractor import NewsExtractor
    except ImportError:
        pass  # Already imported
    
    parser = argparse.ArgumentParser(description="Financial News Analysis Agent")
    parser.add_argument("--query", type=str, default="BTC", help="Search query (default: BTC)")
    parser.add_argument("--from-date", type=str, default=None, help="Date in YYYY-MM-DD format (default: yesterday)")
    parser.add_argument("--sort-by", type=str, default="popularity", choices=["popularity", "relevancy", "publishedAt"], help="Sort order")
    parser.add_argument("--page-size", type=int, default=5, help="Number of articles to process (default: 5)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini)")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = OrchestratorAgent(model=args.model)
    
    # Process news
    articles = orchestrator.process_news_query(
        query=args.query,
        from_date=args.from_date,
        sort_by=args.sort_by,
        page_size=args.page_size
    )
    
    # Display results
    for i, article in enumerate(articles, 1):
        print(f"\n{'='*100}")
        print(f"ARTICLE {i}/{len(articles)}")
        print(f"{'='*100}")
        print(f"Title: {article.get('title', 'N/A')}")
        print(f"Source: {article.get('source', 'N/A')}")
        print(f"URL: {article.get('url', 'N/A')}")
        print(f"Published: {article.get('publishedAt', 'N/A')}")
        print(f"\nOriginal Content Length: {len(article.get('content', ''))} characters")
        print(f"\n{'─'*100}")
        print("SYNTHESIS:")
        print(f"{'─'*100}")
        print(article.get('synthesis', 'No synthesis available'))
        print(f"{'='*100}\n")


if __name__ == "__main__":
    main()

