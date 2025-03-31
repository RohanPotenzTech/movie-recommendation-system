import pymongo
from datetime import datetime, timedelta, UTC
from db_config import MONGO_URI, DB_NAME

class DatabaseHandler:
    def __init__(self):
        try:
            self.client = pymongo.MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.domains_collection = self.db["domains"]
            self.urls_collection = self.db["urls"]

            # Ensure indexes for fast lookups
            self.urls_collection.create_index("url", unique=True)

            #print("✅ Connected to MongoDB successfully!")

        except pymongo.errors.ConnectionFailure as e:
            print(f"❌ Could not connect to MongoDB: {e}")
            exit()

    def url_exists(self, url):
        return bool(self.urls_collection.find_one({"url": url}))

    def update_last_crawled(self, url):
        self.urls_collection.update_one(
            {"url": url},
            {"$set": {"last_crawled": datetime.now(UTC)}},
            upsert=True
        )

    def store_new_url(self, url, html_content, extracted_links):
        new_url_data = {
            "url": url,
            "html_content": html_content,
            "last_crawled": datetime.now(),
            "extracted_links": extracted_links,
        }
        try:
            result = self.urls_collection.update_one(
                {"url": url}, {"$set": new_url_data}, upsert=True
            )
            if result.upserted_id:
                print(f"✅ New URL stored in DB: {url}")
            else:
                print(f"🔄 Updated existing URL in DB: {url}")

        except pymongo.errors.PyMongoError as e:
            print(f"❌ Database error: {e}")

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
