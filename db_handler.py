import pymongo
from datetime import datetime, timedelta, UTC
from db_config import MONGO_URI, DB_NAME
from  hashlib import md5
from urllib.parse import urlparse

class DatabaseHandler:
    def __init__(self):
        try:
            self.client = pymongo.MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.domains_collection = self.db["domains"]
            self.urls_collection = self.db["urls"]

            # Ensure indexes for fast lookups
            self.urls_collection.create_index("url", unique=True)
            
        except pymongo.errors.ConnectionFailure as e:
            print(f"❌ Could not connect to MongoDB: {e}")
            exit()

    def update_last_crawled(self, url):
        self.urls_collection.update_one(
            {"url": url},
            {"$set": {"last_crawled": datetime.now(UTC)}},
            upsert=True
        )

    def store_new_url(self, url, extracted_links, domain_id):
        url_hash = self.generate_md5_hash(url) 
        if domain_id:
            new_url_data = {
            "url": url,
            "md5_hash": url_hash,
            "last_crawled": datetime.now(UTC),
            "extracted_links": extracted_links,
        }
            
        try:
            result = self.urls_collection.update_one(
                {"md5_hash": url_hash}, {"$set": new_url_data}, upsert=True
            )
            if result.upserted_id:
                print(f"✅ New URL stored in DB: {url}")
            else:
                print(f"🔄 Updated existing URL in DB: {url}")

        except pymongo.errors.PyMongoError as e:
            print(f"❌ Database error: {e}")

    def generate_md5_hash(self, url):
        """Generates an MD5 hash for a given URL."""
        return md5(url.encode('utf-8')).hexdigest()

    def get_next_url_to_crawl(self):
        cutoff_time = datetime.now() - timedelta(hours=48)
        query = {
            "$or": [
                {"last_crawled": {"$exists": False}},
                {"last_crawled": {"$lt": cutoff_time}}
            ]
        }
        url_data = self.urls_collection.find_one(query)
        return url_data["url"] if url_data else None
