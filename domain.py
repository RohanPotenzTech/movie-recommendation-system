

"""
class domain
{

get_domain_list()
return []={1}{2}{3}{4}

get_domain(domain_id)
retunn {domain_id}


}"""


from db_handler import DatabaseHandler

class DomainHandler:
    def __init__(self):
        self.db_handler = DatabaseHandler()
        self.domains_collection = self.db_handler.db["domains"]

    def get_domain_list(self):
        """Retrieve all domains with their IDs and URLs."""
        return list(self.domains_collection.find({}, {"_id": 1, "url": 1}))
    def get_domain_id_from_url(self, url):
        """Fetch the domain ID from the URL."""
        from urllib.parse import urlparse
        domain_url = urlparse(url).netloc  # Extract the domain from the URL
        domain = self.domains_collection.find_one({"url": domain_url})
        
        if domain:
            return domain["_id"]  # Return the domain ID if found
        else:
            return None  # Return None if no domain is found

if __name__ == "__main__":
    domain_handler = DomainHandler()
    print("✅ Domain list:", domain_handler.get_domain_list())






    