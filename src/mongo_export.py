from pymongo import MongoClient

def create_mongo_database(records):    
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["ecommerce"]
    collection = db["predicted_intents"]
    collection.delete_many({})  # optional: clear existing
    collection.insert_many(records)
    
def get_collection():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["ecommerce"]
    return db["predicted_intents"]