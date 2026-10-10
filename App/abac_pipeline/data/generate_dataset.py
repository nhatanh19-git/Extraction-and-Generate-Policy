"""Dataset Generator script.
Parses all ABAC files, generates BIO tags for constraints, splits the data by rule,
and exports to train/val/test CoNLL files.
"""

import sys
from pathlib import Path
import logging

# Ensure backend module can be imported when running standalone
base_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from backend.abac_pipeline.data.abac_parser import ABACDatasetParser
from backend.abac_pipeline.data.bio_generator import BIOConstraintGenerator
from backend.abac_pipeline.data.dataset_splitter import GroupedDatasetSplitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    dataset_dir = base_dir / "ABAC_Labs_dataset"
    output_dir = base_dir / "backend" / "abac_pipeline" / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not dataset_dir.exists() or not any(dataset_dir.glob("*.abac")):
        logger.error(f"Dataset directory '{dataset_dir}' does not exist or contains no .abac files.")
        sys.exit(1)
        
    parser = ABACDatasetParser()
    generator = BIOConstraintGenerator()
    splitter = GroupedDatasetSplitter(val_size=0.15, test_size=0.15, random_state=42)
    
    all_rules = []
    
    # Parse all .abac files
    for abac_file in dataset_dir.glob("*.abac"):
        logger.info(f"Parsing {abac_file.name}...")
        rules = parser.parse_file(abac_file)
        logger.info(f"  Found {len(rules)} rules.")
        
        # Add domain as prefix to rule IDs to ensure global uniqueness across domains
        domain_name = abac_file.stem
        for rule in rules:
            if not rule.rule_id:
                rule.rule_id = "unknown"
            rule.rule_id = f"{domain_name}_{rule.rule_id}"
            
        all_rules.extend(rules)
        
    logger.info(f"Total rules extracted: {len(all_rules)}")
    
    # We need to pass the rules through generate_bio. But generate_bio returns a list of tagged sentences.
    # We want to maintain a 1:1 mapping between rule IDs and their tagged sentences for the splitter.
    # Notice that a single rule might produce multiple sentences if doc.sents splits it.
    
    X = []
    y = [] # Not really used by our generator, we keep it parallel to X for sklearn API
    groups = []
    
    for rule in all_rules:
        # Generate BIO tags for this single rule
        tagged_sents = generator.generate_bio([rule])
        for sent in tagged_sents:
            if sent: # skip empty
                X.append(sent)
                y.append(0) # Dummy target
                groups.append(rule.rule_id)
                
    logger.info(f"Generated {len(X)} annotated sentences.")
    
    # Perform split
    splits = splitter.split(X, y, groups)
    
    # Export
    for split_name, (X_split, _) in splits.items():
        out_file = output_dir / f"{split_name}.conll"
        generator.export_conll(X_split, str(out_file))
        logger.info(f"Exported {split_name} set ({len(X_split)} sentences) to {out_file.name}")
        
    # Also export the full monolithic dataset for backward compatibility if needed
    full_out = output_dir / "abac_constraints_full.conll"
    generator.export_conll(X, str(full_out))
    logger.info(f"Exported full dataset to {full_out.name}")


if __name__ == "__main__":
    main()
