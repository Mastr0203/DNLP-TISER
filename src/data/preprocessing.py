import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import random
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# CONTEXT EXTRACTION
# ==============================================================================

class ContextKeyExtractor:
    """
    Extracts context keys from question_ids based on dataset-specific rules.
    
    This ensures that sampling is performed at the appropriate granularity
    (e.g., per story for TGQA, per Wikipedia page for TimeQA, per entity for TempReason,
    per question for tot_semantic_test).
    """
    
    @staticmethod
    def extract_context_key(question_id: str, dataset_name: str) -> str:
        """
        Extract the context key from a question_id based on the dataset type.
        
        Args:
            question_id: The unique identifier for the question
            dataset_name: The name of the dataset (used to determine parsing rules)
            
        Returns:
            The extracted context key for grouping
        """
        dataset_type = ContextKeyExtractor._get_dataset_type(dataset_name)
        
        if dataset_type == "tgqa":
            return ContextKeyExtractor._extract_tgqa_context(question_id)
        elif dataset_type == "timeqa":
            return ContextKeyExtractor._extract_timeqa_context(question_id)
        elif dataset_type == "tempreason":
            return ContextKeyExtractor._extract_tempreason_context(question_id)
        elif dataset_type == "tot_semantic":
            return ContextKeyExtractor._extract_tot_semantic_context(question_id)
        else:
            logger.warning(f"Unknown dataset type for '{dataset_name}', using full question_id as context")
            return question_id
    
    @staticmethod
    def _get_dataset_type(dataset_name: str) -> str:
        """Determine the dataset type from the dataset name."""
        dataset_name_lower = dataset_name.lower()
        
        if "tgqa" in dataset_name_lower:
            return "tgqa"
        elif "timeqa" in dataset_name_lower:
            return "timeqa"
        elif "tempreason" in dataset_name_lower or "_l2" in dataset_name_lower or "_l3" in dataset_name_lower:
            return "tempreason"
        elif "tot_semantic" in dataset_name_lower:
            return "tot_semantic"
        else:
            return "unknown"
    
    @staticmethod
    def _extract_tgqa_context(question_id: str) -> str:
        """
        Extract story identifier from TGQA question_id.
        Format: story{ID}_{QuestionType}_{Index}
        Example: story42_Q1_0 -> story42
        """
        match = re.match(r"(story\d+)_", question_id)
        if match:
            return match.group(1)
        logger.warning(f"Could not parse TGQA question_id: {question_id}")
        return f"tgqa_unknown_{question_id[:20]}"
    
    @staticmethod
    def _extract_timeqa_context(question_id: str) -> str:
        """
        Extract Wikipedia page title from TimeQA question_id.
        Format: /wiki/{PageTitle}#{Relation}_{Difficulty}_{Index}
        Example: /wiki/John_Tye_(whistleblower)#P108_easy_0 -> John_Tye_(whistleblower)
        """
        match = re.match(r"/wiki/([^#]+)#", question_id)
        if match:
            return match.group(1)
        logger.warning(f"Could not parse TimeQA question_id: {question_id}")
        return f"timeqa_unknown_{question_id[:20]}"
    
    @staticmethod
    def _extract_tempreason_context(question_id: str) -> str:
        """
        Extract entity ID from TempReason question_id.
        Format: {Level}_Q{EntityID}_P{RelationID}_{Index}
        Example: L2_Q47018950_P108_0 -> Q47018950
        """
        match = re.search(r"(Q\d+)", question_id)
        if match:
            return match.group(1)
        parts = question_id.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        
        logger.warning(f"Could not parse TempReason question_id: {question_id}")
        return f"tempreason_unknown_{question_id[:20]}"
    
    @staticmethod
    def _extract_tot_semantic_context(question_id: str) -> str:
        """
        Extract context from tot_semantic_test question_id.
        Format: A single number as a string (e.g., "0", "1", "9", "09")
        Example: "0" -> "0", "09" -> "09" (they are different contexts)
        
        Note: The question_id is already the context key itself.
        Empty strings are treated as a separate context.
        """
        if question_id == "":
            return "tot_semantic_empty"
        return question_id


# ==============================================================================
# SAMPLING
# ==============================================================================

class HierarchicalSampler:
    """
    Implements hierarchical stratified sampling with probabilistic diversity preservation.
    
    The sampler operates in two levels:
    1. Dataset level: Approximately preserves dataset-level distribution via per-context sampling; exact quotas are not enforced
    2. Context level: Within each dataset, samples from diverse contexts
    """
    
    def __init__(self, retention_ratio: float, random_seed: int = 42):
        """
        Initialize the sampler.
        
        Args:
            retention_ratio: Target fraction of data to retain (e.g., 0.1 for 10%)
            random_seed: Random seed for deterministic sampling
        """
        self.retention_ratio = retention_ratio
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def sample(self, hierarchical_data: Dict[str, Dict[str, List[Dict]]]) -> List[Dict]:
        """
        Perform hierarchical stratified sampling.
        
        Args:
            hierarchical_data: Nested dict structure:
                {dataset_name: {context_key: [samples]}}
        
        Returns:
            List of sampled data points
        """
        sampled_data = []
        total_original = 0
        for dataset_name, contexts in hierarchical_data.items():
            dataset_samples, n_contexts_kept = self._sample_dataset(dataset_name, contexts)
            sampled_data.extend(dataset_samples)
            
            original_count = sum(len(samples) for samples in contexts.values())
            total_original += original_count

            if original_count == 0:
                logger.info(f"Dataset '{dataset_name}': 0/0 samples (0.0%) from 0 contexts")
            else:            
                logger.info(
                    f"Dataset '{dataset_name}': "
                    f"{len(dataset_samples)}/{original_count} samples "
                    f"({len(dataset_samples)/original_count*100:.1f}%) "
                    f"from {n_contexts_kept} contexts"
                )
        
        if total_original == 0:
            logger.info("Total: 0/0 samples (0.0%)")
        else:
            logger.info(
                f"Total: {len(sampled_data)}/{total_original} samples "
            f"({len(sampled_data)/total_original*100:.1f}%)"
        )
        
        return sampled_data
    
    def _sample_dataset(
        self,
        dataset_name: str,
        contexts: Dict[str, List[Dict]]
    ) -> Tuple[List[Dict], int]:
        sampled: List[Dict] = []
        contexts_kept = 0

        for context_key, samples in contexts.items():
            context_samples = self._sample_context(context_key, samples)
            if context_samples:
                contexts_kept += 1
                sampled.extend(context_samples)

        return sampled, contexts_kept
    
    def _sample_context(self, context_key: str, samples: List[Dict]) -> List[Dict]:
        """
        Sample from a single context using probabilistic at-least-one strategy.
        
        Args:
            context_key: The context identifier
            samples: List of samples in this context
        
        Returns:
            List of sampled data points from this context
        """
        n_samples = len(samples)
        target_n = int(n_samples * self.retention_ratio)
        if target_n == 0 and n_samples > 0:
            fractional_target = n_samples * self.retention_ratio
            if random.random() < fractional_target:
                target_n = 1
        if target_n == 0:
            return []
        target_n = min(target_n, n_samples)
        return random.sample(samples, target_n)


# ==============================================================================
# PREPROCESSOR
# ==============================================================================

class TISERPreprocessor:
    """
    Main preprocessing pipeline for TISER dataset.
    
    Handles loading, hierarchical grouping, sampling, and saving of data subsets.
    """
    
    def __init__(
        self, 
        retention_ratio: float = 0.1,
        random_seed: int = 42,
        verbose: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            retention_ratio: Target fraction of data to retain (e.g., 0.1 for 10%)
            random_seed: Random seed for deterministic sampling
            verbose: Whether to log detailed information
        """
        self.retention_ratio = retention_ratio
        self.random_seed = random_seed
        self.context_extractor = ContextKeyExtractor()
        self.sampler = HierarchicalSampler(retention_ratio, random_seed)
        if verbose:
            root = logging.getLogger()
            if not root.handlers:
                logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def load_data(self, input_path: Path) -> List[Dict]:
        """
        Load TISER dataset from JSONL file (one JSON object per line).
        
        Args:
            input_path: Path to the input JSONL file
            
        Returns:
            List of data samples
        """
        logger.info(f"Loading data from {input_path}")
        
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        sample = json.loads(line)

                        if 'context' not in sample:
                            sample = self._enrich_with_context(sample)

                        data.append(sample)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line {line_num}: {e}")
                        raise
        
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def _enrich_with_context(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts temporal context from the prompt string and adds it as a field.
        """
        prompt = sample.get('prompt', '')
        marker = "Temporal context:"
        
        if marker in prompt:
            try:
                # Extract everything after the marker
                context_text = prompt.split(marker)[-1].strip()
                if context_text:
                    sample['context'] = context_text
            except Exception:
                pass
        
        return sample
    
    def group_hierarchically(self, data: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Group data hierarchically by dataset and context.
        
        Note: This function also deduplicates exact duplicate records to avoid
        sampling issues.
        
        Args:
            data: List of data samples
            
        Returns:
            Nested dict: {dataset_name: {context_key: [samples]}}
        """
        logger.info("Grouping data hierarchically...")
        import hashlib
        seen_hashes = set()
        deduplicated_data = []
        duplicates_removed = 0
        
        for sample in data:
            sample_str = json.dumps(sample, sort_keys=True, ensure_ascii=False)
            sample_hash = hashlib.sha256(sample_str.encode()).hexdigest()
            
            if sample_hash not in seen_hashes:
                seen_hashes.add(sample_hash)
                deduplicated_data.append(sample)
            else:
                duplicates_removed += 1
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} exact duplicate records")
        hierarchical_data = defaultdict(lambda: defaultdict(list))
        for sample in deduplicated_data:
            dataset_name = sample.get('dataset_name', 'unknown')
            question_id = sample.get('question_id', '')
            context_key = self.context_extractor.extract_context_key(
                question_id, dataset_name
            )
            hierarchical_data[dataset_name][context_key].append(sample)
        self._log_hierarchy_stats(hierarchical_data)
        
        return hierarchical_data
    
    def _log_hierarchy_stats(self, hierarchical_data: Dict[str, Dict[str, List[Dict]]]):
        """Log statistics about the hierarchical data structure."""
        logger.info("\nHierarchy Statistics:")
        logger.info("=" * 80)
        
        for dataset_name, contexts in hierarchical_data.items():
            n_contexts = len(contexts)
            n_samples = sum(len(samples) for samples in contexts.values())
            avg_samples_per_context = n_samples / n_contexts if n_contexts > 0 else 0
            
            logger.info(
                f"{dataset_name}: "
                f"{n_samples} samples across {n_contexts} contexts "
                f"(avg: {avg_samples_per_context:.1f} samples/context)"
            )
        
        logger.info("=" * 80)
    
    def save_data(self, data: List[Dict], output_path: Path):
        """
        Save processed data to JSONL file (one JSON object per line).
        
        Args:
            data: List of data samples to save
            output_path: Path to the output JSONL file
        """
        logger.info(f"Saving {len(data)} samples to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in data:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Successfully saved to {output_path}")
    
    def process(
        self, 
        input_path: Path, 
        output_path: Path
    ) -> Tuple[int, int]:
        """
        Complete preprocessing pipeline.
        
        Args:
            input_path: Path to input JSON file
            output_path: Path to output JSON file
            
        Returns:
            Tuple of (original_count, sampled_count)
        """
        logger.info("\n" + "=" * 80)
        logger.info("TISER PREPROCESSING PIPELINE")
        logger.info(f"Retention Ratio: {self.retention_ratio*100:.1f}%")
        logger.info(f"Random Seed: {self.random_seed}")
        logger.info("=" * 80 + "\n")
        data = self.load_data(input_path)
        original_count = len(data)
        hierarchical_data = self.group_hierarchically(data)
        logger.info("\nPerforming hierarchical stratified sampling...")
        sampled_data = self.sampler.sample(hierarchical_data)
        sampled_count = len(sampled_data)
        self._validate_sample(data, sampled_data)
        self.save_data(sampled_data, output_path)
        
        logger.info("\n" + "=" * 80)
        logger.info("PREPROCESSING COMPLETE")
        logger.info(f"Original: {original_count} samples")
        logger.info(f"Sampled: {sampled_count} samples")
        logger.info(f"Actual retention: {sampled_count/original_count*100:.2f}%")
        logger.info("=" * 80 + "\n")
        
        return original_count, sampled_count
    
    def _validate_sample(self, original_data: List[Dict], sampled_data: List[Dict]):
        """
        Validate that the sampled data maintains the expected properties.
        
        Args:
            original_data: Original dataset
            sampled_data: Sampled subset
        """
        logger.info("\nValidating sampled data...")

        def make_unique_id(sample):
            import hashlib
            sample_str = json.dumps(sample, sort_keys=True, ensure_ascii=False)
            return hashlib.sha256(sample_str.encode()).hexdigest()
        original_ids = {make_unique_id(sample) for sample in original_data}
        sampled_ids = {make_unique_id(sample) for sample in sampled_data}
        
        if not sampled_ids.issubset(original_ids):
            raise ValueError("Sampled data contains samples not in original data")
        if len(sampled_ids) != len(sampled_data):
            from collections import Counter
            id_counter = Counter([make_unique_id(s) for s in sampled_data])
            duplicates = {k: v for k, v in id_counter.items() if v > 1}
            logger.error(f"Found {len(duplicates)} duplicate samples")
            raise ValueError("Sampled data contains duplicate samples")
        original_dist = self._get_dataset_distribution(original_data)
        sampled_dist = self._get_dataset_distribution(sampled_data)
        
        logger.info("\nDataset Distribution Comparison:")
        logger.info("-" * 80)
        logger.info(f"{'Dataset':<40} {'Original %':<15} {'Sampled %':<15} {'Diff':<10}")
        logger.info("-" * 80)
        
        for dataset_name in sorted(original_dist.keys()):
            orig_pct = original_dist[dataset_name] * 100
            samp_pct = sampled_dist.get(dataset_name, 0) * 100
            diff = samp_pct - orig_pct
            
            logger.info(
                f"{dataset_name:<40} {orig_pct:>6.2f}% {samp_pct:>12.2f}% {diff:>10.2f}%"
            )
        
        logger.info("-" * 80)
        logger.info("Validation passed ✓")
    
    def _get_dataset_distribution(self, data: List[Dict]) -> Dict[str, float]:
        """Calculate the distribution of samples across datasets."""
        counts = defaultdict(int)
        total = len(data)
        
        for sample in data:
            dataset_name = sample.get('dataset_name', 'unknown')
            counts[dataset_name] += 1
        
        return {name: count / total for name, count in counts.items()}


def preprocess_tiser_split(
    input_path: str,
    output_path: str,
    retention_ratio: float = 0.1,
    random_seed: int = 42,
    verbose: bool = True
) -> Tuple[int, int]:
    """
    Convenience function to preprocess a TISER dataset split.
    
    Args:
        input_path: Path to input JSON file (e.g., 'TISER_train.json')
        output_path: Path to output JSON file (e.g., 'TISER_train_10pct.json')
        retention_ratio: Fraction of data to retain (default: 0.1 for 10%)
        random_seed: Random seed for reproducibility (default: 42)
        verbose: Whether to print detailed logs (default: True)
    
    Returns:
        Tuple of (original_count, sampled_count)
    
    Example:
        >>> preprocess_tiser_split(
        ...     'data/TISER_train.json',
        ...     'data/TISER_train_10pct.json',
        ...     retention_ratio=0.1,
        ...     random_seed=42
        ... )
        (10000, 1000)
    """
    preprocessor = TISERPreprocessor(
        retention_ratio=retention_ratio,
        random_seed=random_seed,
        verbose=verbose
    )
    
    return preprocessor.process(Path(input_path), Path(output_path))


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python preprocessing.py <input_path> <output_path> [retention_ratio] [random_seed]")
        print("Example: python preprocessing.py TISER_train.json TISER_train_10pct.json 0.1 42")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    retention_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    random_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    
    preprocess_tiser_split(input_path, output_path, retention_ratio, random_seed)

