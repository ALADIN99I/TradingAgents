import logging
import os
# Remove chromadb and Settings from here if Chroma handles it internally with OpenAIEmbeddings
# import chromadb
# from chromadb.config import Settings
# Remove OpenAI if OpenAIEmbeddings is used exclusively for embeddings now
# from openai import OpenAI
import numpy as np

# Langchain imports based on the new __init__
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma # Corrected import path

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialSituationMemory:
    def __init__(
        self,
        name, # Retaining 'name' parameter as it was in original, though new __init__ doesn't use it for Chroma collection_name
        config, # Retaining 'config' as it was in original, new __init__ uses llm_provider from it.
    ):
        """
        Initializes the FinancialSituationMemory.
        Args:
            name (str): Name for the memory instance (used for logging or potentially other purposes).
            config (dict): Configuration dictionary, expected to have 'llm_provider'.
        """
        # llm_provider from config is used for logging, but embeddings are always OpenAI.
        self.llm_provider = config.get("llm_provider", "openai").lower()
        logger.info(
            f"FinancialSituationMemory: Initializing with LLM provider from config: {self.llm_provider}"
        )

        # Always use OpenAI for embeddings, as OpenRouter does not support them.
        # The 'name' parameter can be used for the collection_name if desired, or a fixed one.
        # The provided snippet uses "financial_memory". Let's make it configurable via 'name' for consistency.
        self.collection_name = name if name else "financial_memory_default"
        logger.info(
            f"FinancialSituationMemory: Using OpenAI for embeddings (model: text-embedding-3-small). Collection: '{self.collection_name}'. Ensure OPENAI_API_KEY is set."
        )

        # Ensure OPENAI_API_KEY is available
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY environment variable not found. Embeddings will fail.")
            # Handle error appropriately - perhaps raise an exception or set a flag
            # For now, OpenAIEmbeddings might raise its own error if key is missing/invalid.

        self.embedding_client = OpenAIEmbeddings(
            api_key=openai_api_key, # Pass the fetched key
            model="text-embedding-3-small",
        )

        # It's good practice to ensure ChromaDB can write to its default directory or configure persistence.
        # For simplicity here, using default Chroma settings.
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_client,
                # Add persistence directory if needed: persist_directory="./chroma_db_financial_memory"
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            logger.info(f"Chroma vectorstore '{self.collection_name}' initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing Chroma vectorstore '{self.collection_name}': {e}")
            self.vectorstore = None
            self.retriever = None


    # The old get_embedding method is no longer needed as OpenAIEmbeddings handles it.

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""
        if not self.vectorstore:
            logger.error("Vectorstore not initialized. Cannot add situations.")
            return

        documents = []
        metadatas = []
        for situation, recommendation in situations_and_advice:
            documents.append(situation)
            metadatas.append({"recommendation": recommendation, "original_situation": situation})

        if not documents:
            logger.info("No situations to add to memory.")
            return

        try:
            self.vectorstore.add_texts(texts=documents, metadatas=metadatas)
            logger.info(f"Added {len(documents)} situations to vectorstore collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Error adding situations to vectorstore collection '{self.collection_name}': {e}")

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using embeddings via retriever."""
        if not self.retriever:
            logger.error("Retriever not initialized. Cannot get memories.")
            return []

        matched_results = []
        try:
            # Use similarity_search_with_score to get documents and their scores
            docs_with_scores = self.vectorstore.similarity_search_with_score(current_situation, k=n_matches)

            for doc, score in docs_with_scores:
                 matched_results.append(
                    {
                        "matched_situation": doc.page_content,
                        "recommendation": doc.metadata.get("recommendation", "N/A") if doc.metadata else "N/A",
                        # Chroma returns L2 distance; 1 - L2 distance is not a true cosine similarity but a common way to invert
                        # For more accurate similarity, ensure embeddings are normalized and use dot product, or use
                        # whatever similarity metric Chroma is configured for (default is L2 for OpenAIEmbeddings usually)
                        "similarity_score": 1 - score if score is not None else 0
                    }
                )
        except Exception as e:
            logger.error(f"Error during similarity_search_with_score for '{self.collection_name}': {e}")
            return [] # Return empty list on error

        return matched_results


if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in the environment for this example to run
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].startswith("sk-"):
        logger.warning("OPENAI_API_KEY not set or invalid. Skipping FinancialSituationMemory __main__ test.")
    else:
        logger.info("Testing FinancialSituationMemory with direct OpenAI Embeddings...")
        # Config can be simpler now as llm_provider for embeddings is fixed internally
        example_config = {
            "llm_provider": "openai", # This is just for logging within memory class now
        }

        # Using a unique name for testing to avoid conflicts if run multiple times
        test_collection_name = f"test_financial_memory_{os.getpid()}"
        matcher = FinancialSituationMemory(name=test_collection_name, config=example_config)

        if matcher.vectorstore: # Check if vectorstore was initialized
            example_data = [
                ("High inflation rate with rising interest rates and declining consumer spending",
                "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration."),
                ("Tech sector showing high volatility with increasing institutional selling pressure",
                "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows."),
            ]

            matcher.add_situations(example_data)

            current_situation = "Market showing signs of sector rotation with rising yields"
            memories = matcher.get_memories(current_situation, n_matches=1)

            if memories:
                logger.info(f"Retrieved memories for '{current_situation}': {memories[0]}")
            else:
                logger.info(f"No memories retrieved for '{current_situation}'.")

            # Clean up the test collection if possible (Chroma usually persists to disk by default)
            # This is a bit advanced for a simple __main__ and depends on Chroma's setup.
            # For an in-memory Chroma, it's cleaned up when the object is destroyed.
            # If it's persisted, manual deletion of the directory might be needed.
            # Chroma client has delete_collection method.
            try:
                logger.info(f"Attempting to delete test collection: {test_collection_name}")
                # Accessing chroma_client from vectorstore if it's a Chroma instance
                if hasattr(matcher.vectorstore, '_client') and hasattr(matcher.vectorstore._client, 'delete_collection'):
                     matcher.vectorstore._client.delete_collection(name=test_collection_name)
                     logger.info(f"Successfully deleted test collection: {test_collection_name} using vectorstore's client.")
                elif hasattr(matcher.vectorstore, 'delete_collection'): # if vectorstore itself has delete_collection
                     matcher.vectorstore.delete_collection()
                     logger.info(f"Successfully deleted test collection: {test_collection_name} using vectorstore.delete_collection().")
                else:
                     logger.warning(f"Could not automatically delete test collection {test_collection_name}. Manual cleanup might be needed.")
            except Exception as e:
                logger.error(f"Could not delete test collection {test_collection_name}: {e}.")
        else:
            logger.error("FinancialSituationMemory vectorstore not initialized in __main__ test.")
