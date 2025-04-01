import hashlib
import re
import socket
import requests
import os
from extractor import Extractor
from db_handler import DatabaseHandler
from datetime import datetime, timedelta, UTC
from domain import DomainHandler
from hashlib import md5


class URLHandler:
    BATCH_SIZE = 100
    LOCK_TIMEOUT = timedelta(minutes=10)

    def __init__(self):
        self.db_handler = DatabaseHandler()
        self.urls_collection = self.db_handler.db["urls"]

    def fetch_html(self, url):
        """Fetches the HTML content of a URL."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"❌ Error fetching {url}: {e}")
            return None

    def get_url_list_last_crawled_48hrs_before(self, num_of_urls=100):
        """Retrieve and lock URLs atomically to prevent duplicate pickups."""
        now = datetime.now(UTC)
        cutoff_time = now - timedelta(hours=48)
        process_id = f"crawler_{os.getpid()}"  # Unique identifier for the process
        hostname = socket.gethostname()

        urls_to_lock = list(self.urls_collection.find(
            {
                "$or": [
                    {"last_crawled": {"$lt": now - timedelta(hours=48)}},  # Older than 48 hours
                    {"last_crawled": {"$exists": False}},
                    {"status": {"$exists": False}},
                    {"status": "pending"},  # Always crawl "pending" URLs
                    {"status": ""},
                ],
            },
            {"_id": 1 }  # Fetch only required fields
            ).limit(num_of_urls))

        print(f"Found {len(urls_to_lock)} URLs to lock.")

        if not urls_to_lock:
            print("No more URLs to process.")
            return []

        url_ids = [url["_id"] for url in urls_to_lock]  # Step 2: Extract IDs to update

        # Lock the URLs by updating their status and adding a process lock
        self.urls_collection.update_many(
        {
            "_id": {"$in": url_ids},
            "$and": [
                {"status": {"$exists": False}},  # status doesn't exist
                {"status": {"$ne": "processing"}},  # status is not "processing"
                {"status": {"$in": ["", "pending"]}},  # status is either "" or "pending"
            ],
        },
            {
                "$set": {
                    "status": "processing",
                    "locked_at": now,
                    "locked_by": f"{hostname}_{process_id}",
                }
            }
        )

        urls_to_process = list(self.urls_collection.find(
        {
            "$and": [
                { "_id": {"$in": url_ids}},
                {"status": {"$exists": False}},
                {"status": "pending"},
                {"status": ""}
            ]
        },
        {"_id": 1, "url": 1, "status": 1, "locked_at": 1, "locked_by": 1, "domain_id": 1}
    ).limit(num_of_urls))

        for url_data in urls_to_process:
                self.process_url(url_data)

        self.urls_collection.update_many(
            {"_id": {"$in": url_ids}},
            {
                "$set": {
                    "status": "complete",
                    "locked_at": None,
                    "locked_by": None,
                    "last_crawled": now,
                }
            }
        )

    def process_url(self, url):
        html_content = self.fetch_html(url["url"])

        if html_content:
            extracted_links = Extractor.extract_links(html_content, url["url"])
            
            self.extrated_link(extracted_links,url)  # Store the extracted links in the database

              # Update the URL status to 'completed' after processing
            self.urls_collection.update_one(
                {"_id": url["_id"], "locked_by": url["locked_by"]},  # Ensure only the same process updates it
                {"$set": {
                    "last_crawled": datetime.now(UTC),
                    "status": "completed",
                    "locked_at": None,
                    "locked_by": None
                }}
            )
            print(f"✅ Successfully processed and completed URL: {url['url']}")
        else:
            print(f"❌ Error processing URL {url['url']}")
            self.urls_collection.update_one(
                {"_id": url["_id"], "locked_by": url["locked_by"]},
                {"$set": {
                    "status": "pending",
                    "locked_at": None,
                    "locked_by": None
                }}
            )

    def extrated_link(self, extracted_link,url):
        domain_from_url = self.reg_x(url["url"])
    
        now = datetime.now(UTC)
        for link in extracted_link:
            if domain_from_url in link:
                domain_id = url["domain_id"]

                link_hash = self.generate_md5_hash(link)
            
                self.urls_collection.update_one(
                    {"url": link},
                    {
                        "$setOnInsert": {
                            "md5_hash": link_hash,
                            "status": "pending",
                            "last_crawled": now,
                            "locked_at": None,
                            "locked_by": None,
                            "domain_id": url.domain_id
                        }
                    },
                    upsert=True  # This will insert the link if it's new
                )
                print(f"✅ Link added: {link} with domain_id {domain_id}")
            else:
                 print(f"❌ Domain mismatch for link: {link}")
    def generate_md5_hash(self, url):
        """Generates an MD5 hash for a given URL."""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    @staticmethod
    def reg_x(url):
        """Regex to extract the domain from the URL."""
        match = re.search(r'^(?:https?://)?([^/]+)', url)
        if match:
            return match.group(1)  # Extract the domain part
        return None
    
    @staticmethod
    def extract_emails(html):   
        """Extract email addresses from HTML content."""
        import re
        return list(set(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', html)))




#50 urls at a time lock 
#same 50 urls release 

