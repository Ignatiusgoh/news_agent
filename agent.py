"""
Agentic Flow for Financial News Analysis
Orchestrates multi-agent synthesis of news articles with context passing
"""
import os
import math
from typing import List, Dict, Optional
import logging
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
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
    """Individual agent that synthesizes a document section using BART"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: Optional[str] = None):
        """
        Initialize synthesis agent with BART model
        
        Args:
            model_name: Hugging Face model name (default: facebook/bart-large-cnn)
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading BART model: {model_name} on device: {self.device}")
        try:
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("BART model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BART model: {e}")
            raise
    
    def _chunk_text_for_bart(self, text: str, max_length: int = 1024) -> List[str]:
        """
        Split text into chunks that fit within BART's token limit
        
        Args:
            text: Text to chunk
            max_length: Maximum token length (BART max is 1024, using 1000 for safety)
        
        Returns:
            List of text chunks
        """
        # Tokenize to get accurate length
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_length:
            return [text]
        
        chunks = []
        # Split by sentences first, then by token count
        sentences = text.split('. ')
        current_chunk = []
        current_tokens = []
        
        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            
            if len(current_tokens) + len(sentence_tokens) <= max_length:
                current_chunk.append(sentence)
                current_tokens.extend(sentence_tokens)
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                # Start new chunk
                current_chunk = [sentence]
                current_tokens = sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks if chunks else [text]
    
    def summarize(self, text: str, max_length: int = 142, min_length: int = 56) -> str:
        """
        Summarize text using BART
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary (in tokens)
            min_length: Minimum length of summary (in tokens)
        
        Returns:
            Summarized text
        """
        try:
            # BART has a max input of 1024 tokens, so we need to chunk if necessary
            chunks = self._chunk_text_for_bart(text, max_length=1000)
            
            summaries = []
            for chunk in chunks:
                inputs = self.tokenizer(
                    chunk,
                    max_length=1024,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    summary_ids = self.model.generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        min_length=min_length,
                        length_penalty=2.0,
                        num_beams=4,
                        early_stopping=True
                    )
                
                summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summaries.append(summary)
            
            # If multiple chunks, combine summaries
            if len(summaries) > 1:
                combined = ' '.join(summaries)
                # If combined summary is too long, summarize it again
                if len(self.tokenizer.encode(combined, add_special_tokens=False)) > 1000:
                    return self.summarize(combined, max_length=max_length, min_length=min_length)
                return combined
            
            return summaries[0] if summaries else ""
            
        except Exception as e:
            logger.error(f"Error in BART summarization: {e}")
            raise
    
    def synthesize(self, 
                   content: str, 
                   previous_context: Optional[str] = None,
                   section_number: int = 1,
                   total_sections: int = 1) -> str:
        """
        Synthesize and summarize a document section
        
        Args:
            content: The content to synthesize (should be within BART's token limit ~1000 tokens)
            previous_context: Summary from previous sections (if any)
            section_number: Current section number (1-indexed)
            total_sections: Total number of sections
        
        Returns:
            Synthesized summary of the content
        """
        try:
            # Build input text
            if previous_context:
                # Combine previous context with current content
                # Since BART has token limits (1024), we need to be smart about combining
                context_tokens = self.tokenizer.encode(previous_context, add_special_tokens=False)
                content_tokens = self.tokenizer.encode(content, add_special_tokens=False)
                
                # If context is too long, summarize it first to preserve space for current content
                # We want to keep most tokens for the current content
                max_context_tokens = min(300, 1024 - len(content_tokens) - 50)  # Reserve 50 for separators
                
                if len(context_tokens) > max_context_tokens:
                    # Summarize context to fit within available space
                    previous_context = self.summarize(
                        previous_context, 
                        max_length=max_context_tokens // 2,  # Conservative estimate
                        min_length=30
                    )
                
                # Combine context and content with clear separator
                combined_text = f"Context from previous sections: {previous_context}\n\nCurrent section content: {content}"
            else:
                combined_text = content
            
            # Summarize the combined text
            # Use longer max_length when we have context to capture more information
            max_summary_length = 250 if previous_context else 200
            synthesis = self.summarize(combined_text, max_length=max_summary_length, min_length=50)
            
            logger.info(f"Agent synthesized section {section_number}/{total_sections} ({len(content)} chars -> {len(synthesis)} chars)")
            return synthesis
            
        except Exception as e:
            logger.error(f"Error in synthesis agent: {e}")
            raise


class OrchestratorAgent:
    """Orchestrates multiple synthesis agents with context passing"""
    
    def __init__(self, 
                 news_api_key: Optional[str] = None,
                 model_name: str = "facebook/bart-large-cnn",
                 device: Optional[str] = None):
        """
        Initialize orchestrator
        
        Args:
            news_api_key: NewsAPI key
            model_name: Hugging Face BART model name (default: facebook/bart-large-cnn)
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.news_extractor = NewsExtractor(api_key=news_api_key)
        self.synthesis_agent = SynthesisAgent(model_name=model_name, device=device)
        # BART's max input is 1024 tokens, using 1000 for safety margin
        self.max_tokens_per_chunk = 1000
    
    def split_document(self, content: str) -> List[str]:
        """
        Split document into chunks based on BART's token limit (1024 tokens)
        Uses BART tokenizer to accurately measure token counts
        
        Args:
            content: Full document content
        
        Returns:
            List of document chunks, each within BART's token limit
        """
        tokenizer = self.synthesis_agent.tokenizer
        
        # Check if entire document fits within token limit
        full_tokens = tokenizer.encode(content, add_special_tokens=False)
        if len(full_tokens) <= self.max_tokens_per_chunk:
            logger.info(f"Document fits in single chunk ({len(full_tokens)} tokens, {len(content)} chars)")
            return [content]
        
        # Split by sentences first to avoid breaking sentences
        sentences = content.split('. ')
        chunks = []
        current_chunk = []
        current_tokens = []
        
        for sentence in sentences:
            # Add period back if it's not the last sentence
            sentence_with_period = sentence if sentence.endswith('.') else sentence + '.'
            sentence_tokens = tokenizer.encode(sentence_with_period, add_special_tokens=False)
            
            # Check if adding this sentence would exceed token limit
            if len(current_tokens) + len(sentence_tokens) <= self.max_tokens_per_chunk:
                # Add sentence to current chunk
                current_chunk.append(sentence_with_period)
                current_tokens.extend(sentence_tokens)
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                    logger.debug(f"Created chunk {len(chunks)}: {len(tokenizer.encode(chunk_text, add_special_tokens=False))} tokens")
                
                # If single sentence exceeds limit, we need to split it further
                if len(sentence_tokens) > self.max_tokens_per_chunk:
                    # Split long sentence by words
                    words = sentence_with_period.split(' ')
                    current_chunk = []
                    current_tokens = []
                    
                    for word in words:
                        word_tokens = tokenizer.encode(word + ' ', add_special_tokens=False)
                        if len(current_tokens) + len(word_tokens) <= self.max_tokens_per_chunk:
                            current_chunk.append(word)
                            current_tokens.extend(word_tokens)
                        else:
                            if current_chunk:
                                chunk_text = ' '.join(current_chunk)
                                chunks.append(chunk_text)
                                logger.debug(f"Created chunk {len(chunks)} from long sentence: {len(tokenizer.encode(chunk_text, add_special_tokens=False))} tokens")
                            current_chunk = [word]
                            current_tokens = word_tokens
                else:
                    # Start new chunk with this sentence
                    current_chunk = [sentence_with_period]
                    current_tokens = sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            logger.debug(f"Created final chunk {len(chunks)}: {len(tokenizer.encode(chunk_text, add_special_tokens=False))} tokens")
        
        # Verify all chunks are within token limit
        for i, chunk in enumerate(chunks, 1):
            chunk_tokens = tokenizer.encode(chunk, add_special_tokens=False)
            if len(chunk_tokens) > self.max_tokens_per_chunk:
                logger.warning(f"Chunk {i} exceeds token limit: {len(chunk_tokens)} tokens (max: {self.max_tokens_per_chunk})")
        
        total_tokens = sum(len(tokenizer.encode(chunk, add_special_tokens=False)) for chunk in chunks)
        logger.info(f"Split document into {len(chunks)} chunks based on BART token limit ({self.max_tokens_per_chunk} tokens/chunk)")
        logger.info(f"Total: {len(content)} chars -> {total_tokens} tokens across {len(chunks)} chunks")
        
        return chunks if chunks else [content]
    
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
        
        # Final synthesis combining all sections using BART
        logger.info("Creating final synthesis from all sections")
        
        try:
            # Use BART to summarize all section summaries
            # Structure the summaries with clear section markers
            structured_summaries = "\n\n".join([
                f"Section {i+1} Summary: {summary}" 
                for i, summary in enumerate(all_syntheses)
            ])
            
            # Summarize the combined summaries
            # Use longer max_length for final synthesis to capture more information
            final_synthesis = self.synthesis_agent.summarize(
                structured_summaries,
                max_length=300,  # Longer summary for final output
                min_length=100
            )
            
            logger.info(f"Final synthesis complete ({len(final_synthesis)} characters)")
            return final_synthesis
            
        except Exception as e:
            logger.error(f"Error creating final synthesis: {e}")
            # Fallback: return concatenated syntheses with structure
            logger.warning("Falling back to structured concatenated syntheses")
            structured_fallback = "\n\n".join([
                f"Section {i+1}:\n{summary}\n" 
                for i, summary in enumerate(all_syntheses)
            ])
            return structured_fallback
    
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
    parser.add_argument("--model", type=str, default="facebook/bart-large-cnn", help="Hugging Face BART model name (default: facebook/bart-large-cnn)")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Device to run model on (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = OrchestratorAgent(model_name=args.model, device=args.device)
    
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

