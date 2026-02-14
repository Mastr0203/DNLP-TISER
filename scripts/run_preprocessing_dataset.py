#!/usr/bin/env python3
"""
TISER dataset preprocessing with hierarchical stratified sampling.

Creates reduced subsets of the TISER dataset while preserving per-dataset and
per-context distribution. Supports single-file, directory, and multi-ratio modes.

Examples:
    python scripts/run_preprocessing_dataset.py --input data/raw/TISER_train_demo.json --output data/processed/TISER_train_demo_10pct.json --ratio 0.1
    python scripts/run_preprocessing_dataset.py --input-dir data/raw --output-dir data/processed --ratio 0.1 --seed 42
    python scripts/run_preprocessing_dataset.py --input data/raw/TISER_train_demo.json --output-prefix data/processed/TISER_train_demo --ratios 0.05 0.1 0.25
"""

import argparse
import sys
from pathlib import Path
from typing import List
import logging
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import PROJECT_ROOT, RAW_DIR, PROCESSED_DIR
from src.data.preprocessing import preprocess_tiser_split

logger = logging.getLogger(__name__)


# ==============================================================================
# UTILITIES
# ==============================================================================

def _pct_tag(r: float) -> str:
    """Format retention ratio as suffix (e.g. 0.1 -> '10pct')."""
    return f"{round(r * 100)}pct"

def _validate_ratio(r: float):
    if r <= 0.0 or r > 1.0:
        raise ValueError(f"--ratio must be in (0, 1], got {r}")

def _resolve_input_path(p: Path) -> Path:
    """Resolve relative input path against RAW_DIR when only a filename is given."""
    if p.is_absolute() or p.parent != Path("."):
        return p
    return RAW_DIR / p

def _resolve_output_path(p: Path) -> Path:
    """Resolve relative output path against PROCESSED_DIR when only a filename is given."""
    if p.is_absolute() or p.parent != Path("."):
        return p
    return PROCESSED_DIR / p

def setup_logging(verbose: bool = True):
    """Configure logging for the script."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# ==============================================================================
# PROCESSING
# ==============================================================================

def process_single_file(
    input_path: Path,
    output_path: Path,
    retention_ratio: float,
    random_seed: int,
    verbose: bool = True
):
    """
    Process a single TISER dataset file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        retention_ratio: Fraction of data to retain
        random_seed: Random seed for reproducibility
        verbose: Whether to show detailed logs
    """
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {input_path.name}")
    logger.info(f"Output: {output_path.name}")
    logger.info(f"Retention Ratio: {retention_ratio*100:.1f}%")
    logger.info(f"Random Seed: {random_seed}")
    logger.info(f"{'='*80}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        original_count, sampled_count = preprocess_tiser_split(
            str(input_path),
            str(output_path),
            retention_ratio=retention_ratio,
            random_seed=random_seed,
            verbose=verbose
        )
        
        logger.info(f"\n✓ Successfully processed {input_path.name}")
        logger.info(f"  Original: {original_count:,} samples")
        logger.info(f"  Sampled: {sampled_count:,} samples")
        logger.info(f"  Saved to: {output_path}\n")
        
    except Exception as e:
        logger.error(f"\n✗ Error processing {input_path.name}: {e}\n")
        raise


def process_directory(
    input_dir: Path,
    output_dir: Path,
    retention_ratio: float,
    random_seed: int,
    pattern: str = "TISER_*.json",
    verbose: bool = True
):
    """
    Process all TISER files in a directory.
    
    Args:
        input_dir: Directory containing input files
        output_dir: Directory for output files
        retention_ratio: Fraction of data to retain
        random_seed: Random seed for reproducibility
        pattern: Glob pattern for finding TISER files
        verbose: Whether to show detailed logs
    """
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    input_files = sorted(input_dir.glob(pattern))
    
    if not input_files:
        logger.error(f"No files matching pattern '{pattern}' found in {input_dir}")
        sys.exit(1)
    
    logger.info(f"\nFound {len(input_files)} file(s) to process:")
    for f in input_files:
        logger.info(f"  - {f.name}")
    logger.info("")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for input_path in input_files:
        stem = input_path.stem
        suffix = f"_{_pct_tag(retention_ratio)}_seed{random_seed}"
        output_path = output_dir / f"{stem}{suffix}.json"
        
        try:
            process_single_file(
                input_path,
                output_path,
                retention_ratio,
                random_seed,
                verbose
            )
            results.append((input_path.name, "SUCCESS"))
        except Exception as e:
            results.append((input_path.name, f"FAILED: {e}"))
    logger.info(f"\n{'='*80}")
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info(f"{'='*80}")
    for filename, status in results:
        logger.info(f"{filename}: {status}")
    logger.info(f"{'='*80}\n")


def process_multiple_ratios(
    input_path: Path,
    output_prefix: Path,
    retention_ratios: List[float],
    random_seed: int,
    verbose: bool = True
):
    """
    Process a single file with multiple retention ratios.
    
    Args:
        input_path: Path to input JSONL file
        output_prefix: Prefix for output files (without extension)
        retention_ratios: List of retention ratios to apply
        random_seed: Random seed for reproducibility
        verbose: Whether to show detailed logs
    """
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    logger.info(f"\nProcessing {input_path.name} with {len(retention_ratios)} retention ratios")
    logger.info(f"Ratios: {', '.join(f'{r*100:.1f}%' for r in retention_ratios)}\n")
    
    # Ensure output directory exists
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    for ratio in retention_ratios:
        suffix = f"_{_pct_tag(ratio)}"
        output_path = Path(str(output_prefix) + suffix + ".json")
        
        try:
            process_single_file(
                input_path,
                output_path,
                ratio,
                random_seed,
                verbose
            )
            results.append((f"{ratio*100:.1f}%", "SUCCESS"))
        except Exception as e:
            results.append((f"{ratio*100:.1f}%", f"FAILED: {e}"))
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("MULTI-RATIO PROCESSING SUMMARY")
    logger.info(f"{'='*80}")
    for ratio_str, status in results:
        logger.info(f"{ratio_str}: {status}")
    logger.info(f"{'='*80}\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess TISER dataset with hierarchical stratified sampling. If neither --input nor --input-dir is provided, the script processes all matching files in data/raw (as defined in config.py)."
        )
    
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--input', '-i',
        type=Path,
        help='Path to input JSONL file'
    )

    input_group.add_argument(
        '--input-dir', '-d',
        type=Path,
        default=RAW_DIR,
        help=f"Directory containing TISER JSONL files (default: {RAW_DIR})"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Path to output JSONL file (required with --input)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROCESSED_DIR,
        help=f"Output directory (default: {PROCESSED_DIR})"
    )

    parser.add_argument(
        '--output-prefix',
        type=Path,
        help='Output file prefix for multiple ratios (without extension)'
    )
    ratio_group = parser.add_mutually_exclusive_group(required=True)
    ratio_group.add_argument(
        '--ratio', '-r',
        type=float,
        help='Retention ratio (e.g., 0.1 for 10%%)'
    )
    ratio_group.add_argument(
        '--ratios',
        type=float,
        nargs='+',
        help='Multiple retention ratios (e.g., 0.05 0.1 0.25)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--pattern', '-p',
        default='TISER_*.json',
        help='File pattern for directory mode (default: TISER_*.json)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed logging'
    )
    
    args = parser.parse_args()
    if args.input is None and args.input_dir is None:
        args.input_dir = RAW_DIR
    if args.ratio is not None:
        _validate_ratio(args.ratio)
    if args.ratios is not None:
        for r in args.ratios:
            _validate_ratio(r)
    setup_logging(verbose=not args.quiet)

    if args.input:
        args.input = _resolve_input_path(args.input)
        if args.output:
            args.output = _resolve_output_path(args.output)
        if args.output_prefix:
            if not (args.output_prefix.is_absolute() or args.output_prefix.parent != Path(".")):
                args.output_prefix = PROCESSED_DIR / args.output_prefix

    if args.input_dir:
        if not args.input_dir.is_absolute():
            args.input_dir = (PROJECT_ROOT / args.input_dir).resolve()
        if args.output_dir and not args.output_dir.is_absolute():
            args.output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    
    if args.input:
        if args.ratios:
            # Multiple ratios mode
            if not args.output_prefix:
                parser.error("--output-prefix is required with --ratios")
            process_multiple_ratios(
                args.input,
                args.output_prefix,
                args.ratios,
                args.seed,
                verbose=not args.quiet
            )
        else:
            # Single file mode
            if not args.output:
                parser.error("--output is required with --input and --ratio")
            process_single_file(
                args.input,
                args.output,
                args.ratio,
                args.seed,
                verbose=not args.quiet
            )
    
    elif args.input_dir:
        # Directory mode
        if args.ratios:
            parser.error("--ratios is not supported with --input-dir (use --ratio)")
        if not args.output_dir:
            parser.error("--output-dir is required with --input-dir")
        process_directory(
            args.input_dir,
            args.output_dir,
            args.ratio,
            args.seed,
            pattern=args.pattern,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()

