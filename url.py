
"""class url
{
    get_url_list_by_domian(domain_id)
    
    get_url_detail(url_id)

    get_url_list_last_crawled_48hrs_before(num-of-urls) 
    {

        get url from mongodb older then 48hrs last crawldate order by asc last-crawled-date 100 urls (num-of-urls)
        
        update url table with urls we get above with pickup-date 

        loop 100 urls (num-of-urls)
            crawl url 
            parse links from content
            parse emails from content

            save url with content, last-crawl-date, make pickup-date blank

        end look

    }
} """
import requests
import os
from extractor import Extractor
from db_handler import DatabaseHandler
from datetime import datetime, timedelta, UTC

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

        self.urls_collection.urls.updateMany(
             {
                  "$or":[
                        {"last_crawled": {"$lt": cutoff_time} },  # Older than 48 hours
                        {"last_crawled": { "$exists": False } },  
                        {"status": { "$exists": False } } ,
                        {"status": "pending"},  # Always crawl "pending" URLs
                        {"status": ""},  # Always crawl "pending" URLs
                  ]
             },
             {
                "$set": {
                    "status": "processing",
                    "locked_at": now,
                    "locked_by": process_id,
                }
             },
             { "limit": num_of_urls }
        );
        dbself.urls_collection.urls.find({
             "$or":[
                    "status": "processing",
                    "locked_at": now,
                    "locked_by": process_id,
             ]
        },
        {
             url: 1,   
            _id: 1         // Exclude the default '_id' field from the result
        }
        ).limit(num_of_urls),

        
        for loop num_of_rul
            process_url()
        end for


        self.urls_collection.urls.updateMany(
             {
                "_id" : in (urlid1,urlid2) 
             },
             {
                "$set": {
                    "status": "pending",
                    "locked_at": now,
                    "locked_by": 0,
                    "last_crawled":now,
                }
             },
             { "limit": num_of_urls }
        );


def process_url(url)
 
                html_content = self.fetch_html(url)

                if html_content:
                    extracted_links = Extractor.extract_links(html_content, url)
                    extracted_emails = self.extract_emails(html_content)

                    for link in extracted_links:
                        self.urls_collection.update_one(
                            {"url": link},
                            {
                                "$setOnInsert": {
                                    "status": "pending", 
                                    "last_crawled": datetime.now(UTC),
                                    "locked_at": None,
                                    "html_content": "",
                                    "extracted_email": extracted_emails,
                                    "locked_by": None
                                    }
                            },
                            upsert = True
                        )

                    self.urls_collection.update_one(
                        {"_id": url_data["_id"], "locked_by": url_data["locked_by"]},  # Ensure only the same process updates it
                        {"$set": {
                            "html_content": html_content,
                            "last_crawled": datetime.now(UTC),
                            "extracted_emails": extracted_emails,
                            "status": "completed",
                            "locked_at": None,
                            "locked_by": None
                        }}
                    )
            except Exception as e:
                print(f"❌ Error processing URL {url}: {e}")
                self.urls_collection.update_one(
                    {"_id": url_data["_id"], "locked_by": url_data["locked_by"]},
                    {"$set": {"status": "pending", "locked_at": None, "locked_by": None}}
                )







                        {"last_crawled": {"$lt": cutoff_time}},  # Older than 48 hours
                        {"status": not = "pickup"}  # Always crawl "pending" URLs


        urls_to_crawl = []
        for _ in range(num_of_urls):
            url_data = self.urls_collection.find_one_and_update(
                {
                    "$or": [
                        {"last_crawled": {"$lt": cutoff_time}},  # Older than 48 hours
                        {"status": "pending"}  # Always crawl "pending" URLs
                    ],
                    "status": {"$ne": "processing"}  # Ensure we don't double-process
                },
                {
                    "$set": {
                        "status": "processing",
                        "locked_at": now,
                        "locked_by": process_id
                    }
                },
                return_document=True
            )

            if url_data:
                urls_to_crawl.append(url_data)
            else:
                break  # No more available URLs

        return urls_to_crawl


    def process_urls(self, urls_to_crawl):
        """Crawl and process URLs, then mark them as completed."""
        for url_data in urls_to_crawl:
            url = url_data["url"]
            try:
                print(f"🔗 Crawling URL: {url}")
                html_content = self.fetch_html(url)

                if html_content:
                    extracted_links = Extractor.extract_links(html_content, url)
                    extracted_emails = self.extract_emails(html_content)

                    for link in extracted_links:
                        self.urls_collection.update_one(
                            {"url": link},
                            {
                                "$setOnInsert": {
                                    "status": "pending", 
                                    "last_crawled": datetime.now(UTC),
                                    "locked_at": None,
                                    "html_content": "",
                                    "extracted_email": extracted_emails,
                                    "locked_by": None
                                    }
                            },
                            upsert = True
                        )

                    self.urls_collection.update_one(
                        {"_id": url_data["_id"], "locked_by": url_data["locked_by"]},  # Ensure only the same process updates it
                        {"$set": {
                            "html_content": html_content,
                            "last_crawled": datetime.now(UTC),
                            "extracted_emails": extracted_emails,
                            "status": "completed",
                            "locked_at": None,
                            "locked_by": None
                        }}
                    )
            except Exception as e:
                print(f"❌ Error processing URL {url}: {e}")
                self.urls_collection.update_one(
                    {"_id": url_data["_id"], "locked_by": url_data["locked_by"]},
                    {"$set": {"status": "pending", "locked_at": None, "locked_by": None}}
                )


    @staticmethod
    def extract_emails(html):
        """Extract email addresses from HTML content."""
        import re
        return list(set(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', html)))



50 urls at a time lock 
same 50 urls release 