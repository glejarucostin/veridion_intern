import pandas as pd
import numpy as np
import re
import logging
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from joblib import Parallel, delayed
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def clean_text(text: str) -> str:
    """Transform text to lowercase and keep alphanumeric characters and spaces."""
    text = str(text).lower()
    return re.sub(r'[^a-z0-9\s]', '', text).strip()


def get_best_labels(similarity_scores: np.ndarray, labels: List[str], top_n: int = 3, min_score: float = 0.5) -> Tuple[
    List[str], List[float]]:
    """Return top N labels with scores."""
    sorted_indices = np.argsort(similarity_scores)[::-1]
    top_labels = [labels[i] for i in sorted_indices[:top_n] if similarity_scores[i] >= min_score]
    top_scores = [float(similarity_scores[i]) for i in sorted_indices[:top_n] if similarity_scores[i] >= min_score]
    return top_labels if top_labels else [labels[sorted_indices[0]]], top_scores if top_scores else [
        float(similarity_scores[sorted_indices[0]])]


def are_tags_similar(tags: List[str], compare_to: List[str], model: SentenceTransformer,
                     threshold: float = 0.75) -> bool:
    """Check if tags are similar to assigned labels or category."""
    if not tags or not compare_to:
        return False
    tags = tags if isinstance(tags, list) else [tags]
    compare_to = compare_to if isinstance(compare_to, list) else [compare_to]
    tag_embeddings = model.encode(tags, batch_size=256, convert_to_numpy=True)
    compare_embeddings = model.encode(compare_to, batch_size=256, convert_to_numpy=True)
    return np.max(cosine_similarity(tag_embeddings, compare_embeddings), axis=1).mean() >= threshold


def validate_predictions(df: pd.DataFrame, model: SentenceTransformer, sample_size: int = 20) -> float:
    """Validate accuracy on a random sample."""
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    correct = sum(are_tags_similar(row['business_tags'], row['insurance_label'].split('; '), model)
                  for _, row in sample_df.iterrows())
    return correct / len(sample_df) if len(sample_df) > 0 else 0


def process_batch(chunk: pd.DataFrame, taxonomy_labels: List[str], model: SentenceTransformer, taxonomy_emb: np.ndarray,
                  min_confidence: float = 0.5) -> pd.DataFrame:
    """Process a batch of companies, excluding those with empty business_tags."""
    # Convert tags
    chunk['business_tags'] = chunk.get('business_tags', pd.Series(dtype=str)).apply(
        lambda x: literal_eval(x) if pd.notna(x) and isinstance(x, str) else []
    )

    # Filter out companies with empty business_tags
    chunk = chunk[chunk['business_tags'].apply(lambda x: len(x) > 0)].copy()
    if chunk.empty:
        return pd.DataFrame()

    # Initial filtering and text combination
    available_columns = [col for col in ['description', 'category', 'niche'] if col in chunk.columns]
    if not available_columns:
        return pd.DataFrame()

    chunk["full_text"] = chunk[available_columns].fillna('').astype(str).agg(' '.join, axis=1).apply(clean_text)
    chunk = chunk[chunk["full_text"].str.strip() != ''].copy()

    # Generate embeddings
    company_embeddings = model.encode(chunk["full_text"].tolist(), batch_size=512, convert_to_numpy=True)
    cosine_similarities = cosine_similarity(company_embeddings, taxonomy_emb)

    # Assign labels in parallel
    results = Parallel(n_jobs=-1)(
        delayed(get_best_labels)(cosine_similarities[i], taxonomy_labels, top_n=3, min_score=min_confidence)
        for i in range(len(cosine_similarities))
    )
    chunk["insurance_label"] = ['; '.join(labels) for labels, _ in results]
    chunk["confidence_scores"] = [scores for _, scores in results]
    chunk["max_confidence"] = chunk["confidence_scores"].apply(lambda x: max(x) if x else 0.0)

    # Final filtering based on tag similarity
    if 'business_tags' in chunk.columns and 'category' in chunk.columns:
        mask_similar = chunk.apply(
            lambda row: are_tags_similar(row['business_tags'], row['insurance_label'].split('; '), model) and
                        are_tags_similar(row['business_tags'], row['category'], model),
            axis=1
        )
        chunk = chunk[mask_similar]

    return chunk


# Start process
start_time = time.time()
logging.info("Starting classification...")

# Load taxonomy
taxonomy_df = pd.read_csv("insurance_taxonomy - insurance_taxonomy.csv")
if 'label' not in taxonomy_df.columns:
    raise ValueError("Column 'label' not found in taxonomy data.")
taxonomy_labels = taxonomy_df["label"].tolist()

# Load model
logging.info("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
taxonomy_embeddings = model.encode(taxonomy_labels, batch_size=512, convert_to_numpy=True)

# Process companies in batches
batch_size = 20000
chunk_iterator = pd.read_csv("ml_insurance_challenge.csv", chunksize=batch_size)
output_file = "classified_companies.csv"
all_results = []

# Adjustable parameter to control output size indirectly
min_confidence = 0.6  # Slightly higher than original 0.5 for quality

for i, chunk in enumerate(chunk_iterator):
    logging.info(f"Processing batch {i + 1} with {len(chunk)} records...")
    processed_chunk = process_batch(chunk, taxonomy_labels, model, taxonomy_embeddings, min_confidence=min_confidence)
    all_results.append(processed_chunk)

    # Save incrementally
    processed_chunk.to_csv(output_file, mode='a', index=False, header=(i == 0))

# Combine all results for validation
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    logging.info(f"Processed {len(final_df)} companies")

    # Sort and save all (no hard limit)
    final_df = final_df.sort_values('max_confidence', ascending=False)
    final_df.to_csv(output_file, index=False, mode='w')  # Overwrite with sorted results

    # Validate
    accuracy = validate_predictions(final_df, model)
    logging.info(f"Estimated accuracy on sample: {accuracy:.2%}")
else:
    logging.warning("No companies met the criteria!")
    pd.DataFrame().to_csv(output_file, index=False)

logging.info(f"Classification completed in {(time.time() - start_time):.2f} seconds!")