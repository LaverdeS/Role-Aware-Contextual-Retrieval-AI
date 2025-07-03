import os
import tiktoken
import requests
import logging
import trafilatura
from fire import Fire

from dotenv import load_dotenv
from supabase import create_client
from markitdown import MarkItDown
from bs4 import BeautifulSoup
from openai import OpenAI

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.file_management.read import ReadFileTool

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from warnings import filterwarnings
from tqdm import tqdm

from utils import print_tool_call, print_tool_response


filterwarnings('ignore')
CUSTOM_DEBUG = True  # Set to True to enable tool call/response prints

logger = logging.getLogger(__name__)

load_dotenv()

supabase_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

embeddings_model = OpenAIEmbeddings()


# supabase tools
@tool
def search_supabase(query: str, table_name: str) -> str:
    """Search Supabase vector store for relevant documents based on the query."""

    vector_store = SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings_model,
        table_name=table_name,
        query_name="match_documents_engineer_equipment_list"
    )
    retriever = vector_store.as_retriever(search_type="similarity", k=5)  # similarity or mmr (maximal marginal relevance)
    results = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in results])


@tool
def index_documents(docs: list[str], table_name: str) -> str:
    """Index documents (list of strings) into Supabase vector store."""
    if docs and isinstance(docs[0], str):
        docs = [Document(page_content=d) for d in docs]

    _ = SupabaseVectorStore.from_documents(
        docs,
        client=supabase_client,
        embedding=embeddings_model,
        table_name=table_name,
        query_name="match_documents"
    )
    return f"Indexed {len(docs)} documents into {table_name} in Supabase."


# general tools
MAX_NUMBER_OF_TAVILY_RESULTS = 4
RESERVED_NUMBER_OF_TOKENS_FOR_CONTEXT = 6_000
TPM_RATE_LIMIT = 30_000  # max tokens-per-minute for gpt-4o

tavily_tool = TavilySearchResults(max_results=MAX_NUMBER_OF_TAVILY_RESULTS,
                                  search_depth='advanced',
                                  include_answer=False,
                                  include_raw_content=True)
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/115.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br"
})
markitdown_llm_client = OpenAI()
md = MarkItDown(
    requests_session=session,
    llm_client = markitdown_llm_client,
    llm_model = "gpt-4o",
)


# general tools
@tool
def web_insight_scraper(query: str) -> list:
    """
    Searches the web for a given query, visits top relevant links, and extracts clean, token-limited summaries from each page.

    Returns a list of dictionaries containing the URL, title, and extracted text content, truncated to fit token constraints.
    Skips unsupported or slow-loading pages, and handles extraction in parallel for speed.
    """
    if CUSTOM_DEBUG:
        print_tool_call(
            web_insight_scraper,
            tool_name='web_insight_scraper',
            args={'query': query},
        )
    results = tavily_tool.invoke(query)
    docs = []

    def extract_content(url):
        """Helper function to extract content from a URL."""
        max_number_of_tokens = int((TPM_RATE_LIMIT - RESERVED_NUMBER_OF_TOKENS_FOR_CONTEXT) / (MAX_NUMBER_OF_TAVILY_RESULTS-1))
        text_title, truncated_text, error_title, error_content = "", "", "", ""

        try:
            extracted_info = md.convert(url)
            text_title = extracted_info.title.strip() if extracted_info.title else extracted_info.title
            text_content = extracted_info.text_content.strip() if extracted_info.text_content else extracted_info.text_content

            # Fix for garbled text content
            if not text_title:
                text_title = "No Title Found"
                text_content = clean_webpage_text.invoke({"url": url})

            encoding = tiktoken.encoding_for_model("gpt-4")
            tokens = encoding.encode(text_content)
            logger.debug(f"[Number of tokens: {len(tokens)}]")
            truncated_tokens = tokens[:max_number_of_tokens]
            truncated_text = encoding.decode(truncated_tokens)

        except requests.exceptions.HTTPError as http_err:

            error_title = f"HTTP error: {"".join(str(http_err).split(':')[0:-2]).strip()}"
            error_content = "Failed to extract content due to HTTP error."

        except Exception as e:
            error_title = str(e)
            error_content = "Failed to extract content due to an unexpected error."

        if error_title:
            logger.debug(error_title)
            return {
                'url': url,
                'title': error_title,
                'content': error_content
            }
        else:
            return {
                'url': url,
                'title': text_title,
                'content': truncated_text
            }

    with ThreadPoolExecutor() as executor:
        for result in tqdm(results):
            try:
                future = executor.submit(extract_content, result['url'])
                content = future.result(timeout=60)
                docs.append(content)
            except TimeoutError:
                logger.debug(f"Extraction timed out for url: {result['url']}")


            except Exception as e:
                logger.debug(f"Error in ThreadPoolExecutor: {e}")

    valid_docs = [doc for doc in docs if len(doc['content'])>100]
    logger.info(f"[Number of successful scrapped sources: {len(valid_docs)}]")
    if not valid_docs:
        valid_docs = [
            {
                "urls": [result["url"] for result in results],
                "title": f"The scrapper failed to extract content.",
                "content": "No meaningful content found from the provided URLs."
            }
        ]
    if CUSTOM_DEBUG:
        print_tool_response(docs)
        for doc in valid_docs:
            print_tool_response(f"# {doc['title']}\n\n{doc['content']}")
    return valid_docs


@tool
def clean_webpage_text(url: str) -> str:
    """
    Downloads and extracts the main readable content from a webpage using Trafilatura.
    Removes navigation, ads, and boilerplate, returning only meaningful text content.
    The output is token-limited to fit GPT context size.
    """
    if CUSTOM_DEBUG:
        print_tool_call(
            clean_webpage_text,
            tool_name='clean_webpage_text',
            args={'url': url},
        )
    downloaded = trafilatura.fetch_url(url)
    response = ""
    if not downloaded:
        response = "Failed to download the page. Please check the URL."

    if not "Failed" in response:
        response = trafilatura.extract(downloaded)

    response = response or "No meaningful content found."
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(response)
    logger.debug(f"[Number of tokens: {len(tokens)}]")
    truncated_tokens = tokens[:TPM_RATE_LIMIT - RESERVED_NUMBER_OF_TOKENS_FOR_CONTEXT]
    truncated_text = encoding.decode(truncated_tokens)

    if CUSTOM_DEBUG:
        print_tool_response(truncated_text)
    return truncated_text.strip()


read_tool = ReadFileTool()


@tool
def unified_text_loader(file_path: str) -> str:
    """
    Intelligently reads and extracts text content from a wide range of file types and URLs.
    It handles PDFs, DOCX, TXT, CSV, and web pages, while gracefully detecting unsupported formats like certain audio,
    image, and video types. Falls back to alternative extraction methods when needed for maximum resilience.
    """
    if CUSTOM_DEBUG:
        print_tool_call(
            unified_text_loader,
            tool_name='unified_text_loader',
            args={'file_path': file_path},
        )
    root, ext = os.path.splitext(file_path.lower())
    logger.debug(f"Reading file --> root, ext: {root, ext}")

    if ext in [".mp3", ".wav", ".m4a", ".flac"]:
        result = "Missing audio transcription tool."

    elif ext in [".gif", ".bmp", ".tiff", ".webp", ".svg", ".ico", ".avif"]:
        result = "The image format is not supported for analysis."

    elif any(url_pattern in root for url_pattern in ["http://", "https://", "www."]):
        if "youtube.com/watch?v=" in file_path:
            result = "Missing YouTube tools."
        else:
            try:
                result = md.convert(file_path)
                result = result.text_content
            except Exception as e:
                logger.debug(f"Error reading url {file_path} with MarkItDown: {e}")
                logger.debug(f"using extract_clean_text_from_url() for {file_path}")
                result = clean_webpage_text.invoke(file_path)

    else:  # image, pdf, docx, txt, csv, etc.
        try:
            result = md.convert(file_path)
            result = result.text_content

        except Exception as e:
            logger.debug(f"Error reading file {file_path} with MarkItDown: {e}")
            result = read_tool.invoke({"file_path": file_path})

    if CUSTOM_DEBUG:
        print_tool_response(result)
    return result


def execute_tool_test(test_switcher):
    """
    Executes a dummy test based on the switcher value.
    A: Tests web insight scraper with specific queries.
    B: Tests unified text loader with various file types.
    C: Tests Supabase search tool with different table names.
    """
    if test_switcher not in ["A", "B", "C"]:
        raise ValueError("Invalid dummy_test_switcher value. Use 'A', 'B', or 'C'.")

    logger.info(f"Executing test for switcher: {test_switcher}")

    if test_switcher=="A":
        query = "Multi-agent workflows for Building Information Modeling (BIM)."
        _ = web_insight_scraper.invoke({"query": query})

        #query = "BIM with Langchain Langgraph 2025?"
        #_ = web_insight_scraper.invoke({"query": query})

    if test_switcher=="B":
        import time
        file_paths = [
            "data/sample.csv",
            "data/sample.pdf",
            "data/sample.xlsx",
            "data/sample.png",
            "https://en.wikipedia.org/wiki/Agentic_AI",
            ]
        for file_path in file_paths:
            _ = unified_text_loader(file_path)
            time.sleep(3)

    if test_switcher=="C":
        # supabase
        query = "whatever"
        table_name = "engineer_equipment_list"
        _ = search_supabase.invoke({
            "query": query,
            "table_name": table_name
        })

        query = "nothing"
        table_name = "procurement_data"
        _ = search_supabase.invoke({
            "query": query,
            "table_name": table_name
        })
        # 🚧 Update Embeddings Field in supabase for effective similarity search


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(logging.INFO)

    Fire(execute_tool_test)

else:
    logger.setLevel(logging.ERROR)