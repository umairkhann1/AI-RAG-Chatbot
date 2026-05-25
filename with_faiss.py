import os
import sys
import pickle
import re
import logging
from abc import ABC, abstractmethod
from typing import List
import requests
import mimetypes
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
import warnings

# Set USER_AGENT for requests
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

# Suppress langchain-community deprecation warning (optional)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='langchain_community')

from langchain_openai import ChatOpenAI  # type: ignore[import]

from langchain_community.vectorstores import FAISS

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    WebBaseLoader as BaseWebBaseLoader,
)

from selenium import webdriver  # type: ignore[import]
from selenium.webdriver.chrome.options import Options  # type: ignore[import]
from selenium.webdriver.chrome.service import Service  # type: ignore[import]
from selenium.webdriver.common.by import By  # type: ignore[import]
from selenium.webdriver.support.ui import WebDriverWait  # type: ignore[import]
from selenium.webdriver.support import expected_conditions as EC  # type: ignore[import]

import time

load_dotenv()
logger = logging.getLogger(__name__)
CHROMEDRIVER_PATH = os.getenv('CHROMEDRIVER_PATH')

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ API key not found. Set GROQ_API_KEY in your .env file or environment variables."
    )

if not GROQ_API_KEY.startswith(("gsk_", "groq-")):
    raise ValueError(
        "Invalid GROQ API key detected. "
        "Please set GROQ_API_KEY to a valid GROQ key, not a GitHub token or other credential."
    )

os.environ['GROQ_API_KEY'] = GROQ_API_KEY

chat = ChatOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-8b-instant",
    temperature=0
)

class WebBaseLoader(BaseWebBaseLoader):

    def _fetch(
            self, url: str, selector: str = 'body', retries: int = 3, cooldown: int = 2, backoff: float = 1.5
    ) -> str:
        for i in range(retries):
            try:
                #Path to chromedriver executable
                webdriver_service = Service(CHROMEDRIVER_PATH)
                options = webdriver.ChromeOptions()
                options.add_argument('headless')
                driver = webdriver.Chrome(service=webdriver_service, options=options)
                driver.get(url)

                # Wait until the specific element is visible on the page
                WebDriverWait(driver, timeout=500).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, selector))
                )

                content = driver.page_source
                driver.quit()
                return content
            except Exception as e:
                if i == retries - 1:
                    raise
                else:
                    logger.warning(
                        f"Error fetching {url} with attempt "
                        f"{i + 1}/{retries}: {e}. Retrying..."
                    )
                    time.sleep(cooldown * backoff ** i)
        raise ValueError("retry count exceeded")


class DocumentLoader(ABC):
    @abstractmethod
    def load_and_split(self) -> List[str]:
        pass


class URLHandler:
    @staticmethod
    def is_valid_url(url):
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    @staticmethod
    def extract_links(url):
        urls = set()
        domain_name = urlparse(url).netloc
        soup = BeautifulSoup(requests.get(url).content, "html.parser")

        for a_tag in soup.findAll("a"):
            href = a_tag.attrs.get("href")
            if href == "" or href is None:
                continue
            href = urljoin(url, href)
            parsed_href = urlparse(href)
            if parsed_href.path.endswith(
                    ('.pdf', '.jpg', '.png', '.jpeg', '.gif', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx')):
                continue
            href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
            if not URLHandler.is_valid_url(href):
                continue
            if href in urls:
                continue
            if domain_name not in href:
                continue
            urls.add(href)
        return urls

    @staticmethod
    def extract_links_from_websites(websites):
        all_links = set()

        for website in websites:
            links = URLHandler.extract_links(website)
            all_links.update(links)

        return list(all_links)


def remove_phrase(data, phrase):
    # Create a pattern that matches the phrase and any words around it
    pattern = re.compile(r'\b\w*?\s*' + re.escape(phrase) + r'\s*\w*\b', re.IGNORECASE)

    # Remove the phrase and any surrounding words
    cleaned_data = re.sub(pattern, '', data)

    return cleaned_data


def get_loader(file_path_or_url):
    import mimetypes

    if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
        handle_website = URLHandler()
        return WebBaseLoader(
            handle_website.extract_links_from_websites([file_path_or_url])
        )

    mime_type, _ = mimetypes.guess_type(file_path_or_url)

    # FIX: handle CSV better
    if file_path_or_url.endswith(".csv") or mime_type in [
        "text/csv",
        "application/vnd.ms-excel"
    ]:
        return CSVLoader(file_path_or_url)

    elif mime_type == "application/pdf":
        return PyPDFLoader(file_path_or_url)

    elif mime_type == "text/plain" or file_path_or_url.endswith(".txt"):
        return TextLoader(file_path_or_url)

    elif mime_type in [
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]:
        return UnstructuredWordDocumentLoader(file_path_or_url)

    else:
        raise ValueError(f"Unsupported file type: {mime_type}")


def train_or_load_model(train, faiss_obj_path, file_path, index_name):
    if train:
        phrase = "Machine Translated by Google"
        loader = get_loader(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,
                                                       chunk_overlap=400)
        pages = loader.load_and_split(text_splitter=text_splitter)
        
        print(f"\n📊 Starting to process {len(pages)} document chunks...")
        
        for idx, i in enumerate(pages):
            if idx % 10 == 0:
                print(f"Processing chunk {idx+1}/{len(pages)}...")
            print("\n ______________")
            i.page_content = remove_phrase(i.page_content, phrase)  # remove if the data translate from google
            # i.page_content = structured_chunk(i.page_content)
            # print(i.page_content)  # Commenting out to reduce output verbosity

        # Save pages to a text file
        print("💾 Saving document pages...")
        with open('output.txt', 'w', encoding='utf-8') as f:
            sys.stdout = f  # Redirect standard output to the file
            print(pages)  # The output will be saved to 'output.txt'

            sys.stdout = sys.__stdout__  # Reset standard output

        print("🔍 Creating FAISS embeddings index (this may take a few minutes)...")
        if os.path.exists("models/index.faiss"):
            faiss_index = FAISS.load_local(
                "models",
                embeddings,
                allow_dangerous_deserialization=True)
        else:
            faiss_index = FAISS.from_documents(pages, embeddings)
            faiss_index.save_local("models")

        # Ensure a pickle file exists for easy loading by other scripts (e.g., app.py)
        try:
            faiss_index.save(faiss_obj_path)
        except Exception as e:
            logger.warning(f"Could not save FAISS pickle to {faiss_obj_path}: {e}")

        print("✅ Training complete!")

        return faiss_index
    else:
        print("📂 Loading existing FAISS index...")
        faiss_index = FAISS.load_local(
            "models",
            embeddings,
            allow_dangerous_deserialization=True)

        # If a pickle doesn't exist yet, create one so other code can load by path
        if not os.path.exists(faiss_obj_path):
            try:
                faiss_index.save(faiss_obj_path)
            except Exception as e:
                logger.warning(f"Could not save FAISS pickle to {faiss_obj_path}: {e}")

        return faiss_index


def structured_chunk(message):
    messages = [SystemMessage(
        content="Please enhance and refine the following text to ensure clarity and standardization. Remove all "
                "extraneous components, including HTML tags, miscellaneous characters, and any segments "
                "translated by automatic systems like Google Translate."), HumanMessage(content=message)]

    ai_response = chat.invoke(messages).content
    return ai_response


def answer_questions(faiss_index):
    messages = [
        SystemMessage(
            content='I want you to act as a document that I am having a conversation with. Your name is "AI '
                    'Assistant". You will provide me with answers from the given info. If the answer is not included, '
                    'say exactly "Hmm, I am not sure." and stop after that. Refuse to answer any question not about '
                    'the info. Never break character.')
    ]

    while True:
        question = input("Ask a question (type 'stop' to end): ")
        if question.lower() == "stop":
            break

        docs = faiss_index.similarity_search(query=question, k=2)
        print(f"\n🔍 Retrieved {len(docs)} relevant document chunks for the question.")
        

        main_content = question + "\n\n"
        for doc in docs:
            main_content += doc.page_content + "\n\n"

        messages.append(HumanMessage(content=main_content))
        ai_response = chat.invoke(messages).content
        messages.pop()
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=ai_response))

        print(ai_response)


def main():
    faiss_obj_path = "models/ycla.pickle"
    file_path = "data/shams.txt"
    index_name = "ycla"

    train = int(input("Do you want to train the model? (1 for yes, 0 for no): "))
    faiss_index = train_or_load_model(train, faiss_obj_path, file_path, index_name)
    answer_questions(faiss_index)

if __name__ == "__main__":
    main()
