import chromadb
from chromadb.config import Settings
from openai import OpenAI
import numpy as np # Add numpy for zero vector
import logging # Add logging
import os # For API keys in example

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialSituationMemory:
    def __init__(self, name, config):
        self.config = config # Store config
        self.embedding_model_name = config.get("embedding_llm")
        self.llm_provider = config.get("llm_provider")

        if self.llm_provider == "openrouter":
            # For OpenRouter, we assume chat functionalities. Embeddings might not be supported.
            # We'll initialize a client but get_embedding will handle it.
            self.client = OpenAI(base_url=config["backend_url"], api_key=os.environ.get("OPENROUTER_API_KEY"))
            logger.info("FinancialSituationMemory: OpenRouter is the LLM provider. Embedding support relies on OpenRouter's OpenAI API compatibility for embeddings.")
        else: # Assuming direct OpenAI or other compatible
            self.client = OpenAI(base_url=config.get("backend_url"), api_key=os.environ.get("OPENAI_API_KEY")) # backend_url might be None for direct OpenAI

        self.chroma_client = chromadb.Client(Settings(allow_reset=True)) # Consider making persistence configurable
        # Ensure collection name is valid for ChromaDB (e.g., no spaces, specific length)
        safe_collection_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in name)
        if not (3 <= len(safe_collection_name) <= 63):
            safe_collection_name = f"mem_{safe_collection_name[:50]}" # Adjust if still invalid
            logger.warning(f"Original collection name '{name}' was invalid. Using '{safe_collection_name}'.")

        try:
            self.situation_collection = self.chroma_client.get_or_create_collection(name=safe_collection_name)
        except Exception as e:
            logger.error(f"Error creating/getting ChromaDB collection '{safe_collection_name}': {e}")
            # Fallback or re-raise if critical
            self.situation_collection = None


    def get_embedding(self, text):
        """Get embedding for a text. Handles OpenRouter case where embeddings might not be supported."""
        if self.llm_provider == "openrouter":
            # Based on user feedback, OpenRouter may not support embeddings via this client.
            # Returning a dummy vector or raising an explicit error.
            # For now, log a warning and return a zero vector to prevent crashes,
            # effectively disabling embedding-based memory for OpenRouter.
            logger.warning(
                f"Attempting to get embedding with OpenRouter for model '{self.embedding_model_name}'. "
                "OpenRouter may not support this. Returning a zero vector. Memory functionality will be impaired."
            )
            # text-embedding-3-small has dimension 1536
            # text-embedding-ada-002 (common default) also 1536
            # nomic-embed-text (from original commented code) could vary
            # For robustness, ideally, dimension should be configurable or detected.
            # Using 1536 as a common default.
            return np.zeros(1536).tolist()

        if not self.client:
            logger.error("Embedding client not initialized.")
            return np.zeros(1536).tolist() # Or raise error

        try:
            response = self.client.embeddings.create(
                model=self.embedding_model_name, input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding for model '{self.embedding_model_name}' from provider '{self.llm_provider}': {e}")
            logger.warning("Returning a zero vector due to embedding error. Memory functionality will be impaired.")
            return np.zeros(1536).tolist()


    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""
        if not self.situation_collection:
            logger.error("ChromaDB collection not available. Cannot add situations.")
            return

        situations = []
        advice = []
        ids = []
        embeddings_list = [] # Renamed to avoid conflict

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            # Get embedding for each situation
            embedding_vector = self.get_embedding(situation)
            embeddings_list.append(embedding_vector) # Use the renamed list

        if not situations: # if all situations resulted in errors or list is empty
            logger.info("No valid situations to add to memory.")
            return

        try:
            self.situation_collection.add(
                documents=situations,
                metadatas=[{"recommendation": rec} for rec in advice],
                embeddings=embeddings_list, # Use the renamed list
                ids=ids,
            )
            logger.info(f"Added {len(situations)} situations to memory collection '{self.situation_collection.name}'.")
        except Exception as e:
            logger.error(f"Error adding situations to ChromaDB collection '{self.situation_collection.name}': {e}")


    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        if not self.situation_collection:
            logger.error("ChromaDB collection not available. Cannot get memories.")
            return []

        query_embedding = self.get_embedding(current_situation)

        # Check if query_embedding is a zero vector, which indicates an issue or purposeful disabling
        if np.all(np.array(query_embedding) == 0):
            logger.warning("Query embedding is a zero vector. Similarity search will likely be ineffective.")
            # Optionally return empty or handle as no useful memory can be retrieved.
            # For now, let ChromaDB attempt the query, it might return random results or handle it.

        try:
            results = self.situation_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_matches,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as e:
            logger.error(f"Error querying ChromaDB collection '{self.situation_collection.name}': {e}")
            return []

        matched_results = []
        if results and results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                # Ensure metadata and other fields exist to prevent KeyErrors
                meta = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] and i < len(results["metadatas"][0]) else {}
                doc = results["documents"][0][i] if results["documents"] and results["documents"][0] and i < len(results["documents"][0]) else "N/A"
                dist = results["distances"][0][i] if results["distances"] and results["distances"][0] and i < len(results["distances"][0]) else float('inf')

                matched_results.append(
                    {
                        "matched_situation": doc,
                        "recommendation": meta.get("recommendation", "N/A"),
                        "similarity_score": 1 - dist if dist != float('inf') else 0, # Handle potential None or invalid distance
                    }
                )

        return matched_results

# Minor change to the example usage for testing if run directly
if __name__ == "__main__":
    # Example config similar to what alpaca_trader might pass
    example_config_openrouter = {
        "llm_provider": "openrouter",
        "embedding_llm": "openai/text-embedding-3-small", # This will be problematic for OpenRouter
        "backend_url": "https://openrouter.ai/api/v1",
        # OPENROUTER_API_KEY should be in environment
    }
    # Ensure OPENROUTER_API_KEY is set for the example to run
    if "OPENROUTER_API_KEY" not in os.environ:
        os.environ["OPENROUTER_API_KEY"] = "YOUR_OPENROUTER_KEY_HERE" # Replace with a real key if testing __main__

    logger.info("Testing FinancialSituationMemory with OpenRouter config...")
    # Provide a valid name for the collection
    matcher_or = FinancialSituationMemory(name="test_openrouter_memory_main", config=example_config_openrouter)

    example_data = [
        ("High inflation", "Consider defensive stocks."),
        ("Tech rally", "Look into growth tech."),
    ]
    if matcher_or.situation_collection: # Check if collection was initialized
        matcher_or.add_situations(example_data) # This will use zero vectors for OpenRouter

        memories = matcher_or.get_memories("Market is volatile", n_matches=1)
        if memories:
            logger.info(f"OpenRouter - Retrieved memories: {memories[0]}")
        else:
            logger.info("OpenRouter - No memories retrieved or memory search ineffective.")
    else:
        logger.error("OpenRouter - ChromaDB collection not initialized for main test.")

    # Example config for direct OpenAI (assuming OPENAI_API_KEY is a real OpenAI key)
    # For this to work, OPENAI_API_KEY must be a valid OpenAI key.
    # if "OPENAI_API_KEY" not in os.environ or "YOUR_REAL_OPENAI_KEY" in os.environ["OPENAI_API_KEY"]:
    #     print("\nSkipping direct OpenAI test, set a real OPENAI_API_KEY environment variable.")
    # else:
    #     example_config_openai = {
    #         "llm_provider": "openai",
    #         "embedding_llm": "text-embedding-ada-002", # A common OpenAI model
    #         "backend_url": None, # Default OpenAI client will use standard base URL
    #     }
    #     logger.info("\nTesting FinancialSituationMemory with direct OpenAI config...")
    #     matcher_openai = FinancialSituationMemory(name="test_openai_memory_main", config=example_config_openai)
    #     if matcher_openai.situation_collection:
    #         matcher_openai.add_situations(example_data)
    #         memories_openai = matcher_openai.get_memories("Market is volatile", n_matches=1)
    #         if memories_openai:
    #             logger.info(f"OpenAI - Retrieved memories: {memories_openai[0]}")
    #         else:
    #             logger.info("OpenAI - No memories retrieved or memory search ineffective.")
    #     else:
    #         logger.error("OpenAI - ChromaDB collection not initialized for main test.")
