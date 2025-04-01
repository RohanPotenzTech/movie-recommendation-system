from url import URLHandler

import time

def main():
    """Main function to manage crawling process."""
    url_handler = URLHandler()
    
    # Process URLs older than 48 hours
    urls_to_crawl = url_handler.get_url_list_last_crawled_48hrs_before(100)
    if urls_to_crawl:
        url_handler.process_urls(urls_to_crawl)
    else:
        print("✅ No URLs pending for crawling.")

if __name__ == "__main__":
        main()
       # print("⏳ Waiting before next run...")
       # time.sleep(120)  # Wait 2 minutes before rechecking
