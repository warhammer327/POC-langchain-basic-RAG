import weaviate


def create_vectordb_client():
    client = weaviate.connect_to_local(
        host="localhost",
        port=8087,
    )
    return client


client = create_vectordb_client()

try:
    # Get all collection names
    collections = client.collections.list_all()

    # Delete each collection (collections are strings, not objects)
    for collection_name in collections:
        client.collections.delete(collection_name)
        print(f"Deleted collection: {collection_name}")
finally:
    # Always close the connection, even if an error occurs
    client.close()
    print("Connection closed")
