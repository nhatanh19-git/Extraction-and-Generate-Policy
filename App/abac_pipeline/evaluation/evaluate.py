"""Evaluation Module for Cross-Domain Generalization."""

import logging
import sys
from pathlib import Path

# Ensure backend module can be imported when running standalone
base_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from typing import List, Dict, Tuple
from backend.abac_pipeline.data.abac_parser import ABACDatasetParser, ABACPolicyRule
from backend.abac_pipeline.pipeline import ABACPipeline, NLPResult

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, pipeline: ABACPipeline):
        self.pipeline = pipeline
        self.parser = ABACDatasetParser()
        
    def load_dataset(self, data_dir: str) -> Dict[str, List[ABACPolicyRule]]:
        """Load rules grouped by domain (file name)."""
        dataset_dir = Path(data_dir)
        grouped_rules = {}
        
        for abac_file in dataset_dir.glob("*.abac"):
            domain_name = abac_file.stem
            rules = self.parser.parse_file(abac_file)
            grouped_rules[domain_name] = rules
            
        return grouped_rules
        
    def evaluate_leave_one_out(self, data_dir: str):
        """Evaluate the pipeline using Leave-One-Domain-Out strategy."""
        grouped_rules = self.load_dataset(data_dir)
        domains = list(grouped_rules.keys())
        
        logger.info(f"Loaded {len(domains)} domains for evaluation: {domains}")
        
        results = {}
        
        for test_domain in domains:
            logger.info(f"--- Evaluating Domain: {test_domain} ---")
            
            # Since our BiLSTM currently trains on all data in the trainer (Phase 4), 
            # true LODO for the ML component would require retraining per fold. 
            # For this pipeline evaluation, we test the end-to-end extraction capability 
            # on the target domain using the currently loaded models.
            
            test_rules = grouped_rules[test_domain]
            domain_results = self.evaluate_domain(test_rules)
            results[test_domain] = domain_results
            
            logger.info(f"Domain {test_domain} Results: Entities Extracted: {domain_results['entity_count']}, Constraints Extracted: {domain_results['constraint_count']}")
            
        return results

    def evaluate_domain(self, rules: List[ABACPolicyRule]) -> Dict[str, int]:
        """Evaluate the pipeline on a set of rules."""
        metrics = {
            "total_rules": len(rules),
            "processed": 0,
            "entity_count": 0,
            "constraint_count": 0,
            "failed": 0
        }
        
        for rule in rules:
            try:
                result = self.pipeline.process_policy(rule.description)
                metrics["processed"] += 1
                
                if result.entities:
                    metrics["entity_count"] += 1
                    
                if result.constraint_spans:
                    metrics["constraint_count"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to process rule: {rule.description}. Error: {e}")
                metrics["failed"] += 1
                
        return metrics

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    # Initialize pipeline
    try:
        pipeline = ABACPipeline()
        evaluator = Evaluator(pipeline)
        
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        data_dir = base_dir / "ABAC_Labs_dataset"
        
        if data_dir.exists():
            evaluator.evaluate_leave_one_out(str(data_dir))
        else:
            logger.error(f"Data directory {data_dir} not found.")
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        sys.exit(1)
