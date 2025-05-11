#!/usr/bin/env python
"""
Document Summarization with LangChain and Azure OpenAI

This script demonstrates how to load text documents using LangChain document loaders
and use Azure OpenAI to generate summaries using API Key authentication.
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# Load .env file from the correct path - ensure it's only loaded once at the start
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_document(file_path: str) -> str:
    """
    Load a document from a file path using LangChain document loaders
    and return its full text content
    
    Args:
        file_path: Path to the file or directory to load
        
    Returns:
        Document text content
    """
    try:
        logger.info(f"Loading document from {file_path}")
        
        if os.path.isdir(file_path):
            # Load all text files in the directory
            loader = DirectoryLoader(file_path, glob="**/*.txt")
            documents = loader.load()
            # Combine all texts with document separators
            text_content = "\n\n---\n\n".join([doc.page_content for doc in documents])
        else:
            # Load single file
            loader = TextLoader(file_path)
            documents = loader.load()
            text_content = documents[0].page_content if documents else ""
            
        logger.info(f"Loaded document with {len(text_content)} characters")
        return text_content
    except Exception as e:
        logger.error(f"Error loading document: {e}")
        raise

def create_azure_openai_llm(temperature: float = 0.3):
    """
    Create Azure OpenAI LLM client with API key authentication
    
    Args:
        temperature: Temperature for text generation (0.0 to 1.0)
        
    Returns:
        Configured AzureChatOpenAI instance
    """
    try:
        # Get configuration from environment variables - don't load_dotenv again here
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
        
        if not endpoint or not deployment or not api_key:
            raise ValueError("Missing required environment variables: "
                            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, and AZURE_OPENAI_API_KEY must be set")
        
        logger.info(f"Initializing Azure OpenAI client (endpoint={endpoint}, deployment={deployment})")
        
        # Initialize Azure OpenAI with API key authentication
        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=api_version,
            api_key=api_key,
            temperature=temperature
        )
        
        return llm
    except Exception as e:
        logger.error(f"Error creating Azure OpenAI client: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_text(chain, text):
    """
    Process text through the summarization chain with retry logic
    
    Args:
        chain: LangChain processing chain
        text: Text to summarize
        
    Returns:
        Summarized text
    """
    try:
        logger.debug(f"Processing text (size={len(text)})")
        return chain.invoke({"text": text})
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise

def summarize_document(file_path: str, output_file: Optional[str] = None) -> str:
    """
    Summarize a document using Azure OpenAI
    
    Args:
        file_path: Path to the document to summarize
        output_file: Optional path to save the summary
        
    Returns:
        Generated summary text
    """
    try:
        # Remove extra load_dotenv call here - we already loaded it at the module level
        
        # Step 1: Load the document
        document_text = load_document(file_path)
        
        # Step 2: Initialize Azure OpenAI with API key
        llm = create_azure_openai_llm()
        
        # Step 3: Create summarization prompt
        prompt_template = """
        You are an expert summarizer. Your task is to create a concise and comprehensive summary 
        of the following text. Focus on the main points, key information, and important details.
        
        TEXT:
        {text}
        
        SUMMARY:
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Step 4: Create processing chain
        chain = prompt | llm | StrOutputParser()
        
        # Step 5: Process the entire text
        logger.info("Generating summary for the document")
        summary = process_text(chain, document_text)
        
        # Save to file if requested
        if output_file:
            logger.info(f"Saving summary to {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)
        
        return summary
    except Exception as e:
        logger.error(f"Error in document summarization process: {e}")
        return f"Error summarizing document: {str(e)}"

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Summarize text documents using Azure OpenAI and LangChain")
    parser.add_argument("file_path", help="Path to the text file or directory of text files to summarize")
    parser.add_argument("--output", "-o", help="Path to save the summary output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.exists(args.file_path):
        logger.error(f"Error: File or directory not found at {args.file_path}")
        return
    
    try:
        print(f"Summarizing document(s) from {args.file_path}...\n")
        summary = summarize_document(args.file_path, args.output)
        print("\nSUMMARY:\n" + "="*80)
        print(summary)
        print("="*80)
        
        if args.output:
            print(f"\nSummary saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Failed to generate summary: {e}")

if __name__ == "__main__":
    main()