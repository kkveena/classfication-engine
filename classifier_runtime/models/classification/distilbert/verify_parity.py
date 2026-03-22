"""Parity verification between PyTorch and ONNX model outputs.

Compares predictions from the PyTorch reference wrapper and the
ONNX Runtime wrapper to ensure they match within tolerance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from classifier_runtime.models.classification.distilbert.onnx_model import (
    LocalDistilBertOnnxModel,
)
from classifier_runtime.models.classification.distilbert.pytorch_model import (
    LocalDistilBertModel,
)
from classifier_runtime.models.classification.distilbert.schemas import (
    ParityCheckResult,
)

logger = logging.getLogger(__name__)

# Default test samples for parity checks
DEFAULT_TEST_SAMPLES = [
    "This is a short test.",
    "Please classify this customer inquiry about their recent order status and delivery timeline.",
    "I need help with my account billing.",
    "The product quality was excellent and exceeded my expectations in every way possible. "
    "I would highly recommend this to anyone looking for a reliable solution that delivers "
    "consistent results over time.",
]


def check_single_parity(
    pytorch_result: Dict,
    onnx_result: Dict,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> ParityCheckResult:
    """Compare single prediction outputs from PyTorch and ONNX.

    Args:
        pytorch_result: Output from PyTorch model predict().
        onnx_result: Output from ONNX model predict().
        atol: Absolute tolerance for numerical comparison.
        rtol: Relative tolerance for numerical comparison.

    Returns:
        ParityCheckResult with comparison details.
    """
    text = pytorch_result["text"]

    pt_class = pytorch_result["predicted_class"]
    ox_class = onnx_result["predicted_class"]
    class_match = pt_class == ox_class

    # Label comparison
    label_match = None
    pt_label = pytorch_result.get("predicted_label")
    ox_label = onnx_result.get("predicted_label")
    if pt_label is not None and ox_label is not None:
        label_match = pt_label == ox_label

    # Logit comparison
    pt_logits = np.array(pytorch_result["logits"])
    ox_logits = np.array(onnx_result["logits"])
    logits_close = np.allclose(pt_logits, ox_logits, atol=atol, rtol=rtol)
    max_logit_diff = float(np.max(np.abs(pt_logits - ox_logits)))

    # Probability comparison
    pt_probs = np.array(pytorch_result["probabilities"])
    ox_probs = np.array(onnx_result["probabilities"])
    probs_close = np.allclose(pt_probs, ox_probs, atol=atol, rtol=rtol)
    max_prob_diff = float(np.max(np.abs(pt_probs - ox_probs)))

    passed = class_match and logits_close and probs_close
    if label_match is not None:
        passed = passed and label_match

    return ParityCheckResult(
        text=text,
        pytorch_predicted_class=pt_class,
        onnx_predicted_class=ox_class,
        class_match=class_match,
        label_match=label_match,
        logits_close=logits_close,
        probabilities_close=probs_close,
        max_logit_diff=max_logit_diff,
        max_probability_diff=max_prob_diff,
        passed=passed,
    )


def run_parity_checks(
    pytorch_model_dir: str,
    onnx_model_dir: str,
    test_texts: Optional[List[str]] = None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    batch_size: int = 8,
) -> Dict:
    """Run full parity checks between PyTorch and ONNX models.

    Args:
        pytorch_model_dir: Path to PyTorch model directory.
        onnx_model_dir: Path to ONNX model directory.
        test_texts: List of test texts. Uses defaults if None.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        batch_size: Batch size for batch prediction test.

    Returns:
        Dictionary with check results and summary.
    """
    if test_texts is None:
        test_texts = DEFAULT_TEST_SAMPLES

    logger.info("Loading PyTorch model from: %s", pytorch_model_dir)
    pt_model = LocalDistilBertModel(
        model_path=pytorch_model_dir, device="cpu"
    )

    logger.info("Loading ONNX model from: %s", onnx_model_dir)
    ox_model = LocalDistilBertOnnxModel(model_dir=onnx_model_dir)

    # Single prediction parity
    single_results = []
    for text in test_texts:
        pt_output = pt_model.predict(text)
        ox_output = ox_model.predict(text)
        result = check_single_parity(pt_output, ox_output, atol=atol, rtol=rtol)
        single_results.append(result)

        status = "PASS" if result.passed else "FAIL"
        logger.info(
            "[%s] text='%s...' pt_class=%d ox_class=%d "
            "max_logit_diff=%.6f max_prob_diff=%.6f",
            status,
            text[:40],
            result.pytorch_predicted_class,
            result.onnx_predicted_class,
            result.max_logit_diff,
            result.max_probability_diff,
        )

    # Batch prediction parity
    pt_batch = pt_model.batch_predict(test_texts, batch_size=batch_size)
    ox_batch = ox_model.batch_predict(test_texts, batch_size=batch_size)

    batch_results = []
    for pt_out, ox_out in zip(pt_batch, ox_batch):
        result = check_single_parity(pt_out, ox_out, atol=atol, rtol=rtol)
        batch_results.append(result)

    # Summary
    all_results = single_results + batch_results
    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    failed = total - passed

    summary = {
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "all_passed": failed == 0,
        "single_results": [r.model_dump() for r in single_results],
        "batch_results": [r.model_dump() for r in batch_results],
        "tolerances": {"atol": atol, "rtol": rtol},
    }

    if failed == 0:
        logger.info("All %d parity checks PASSED", total)
    else:
        logger.warning("%d of %d parity checks FAILED", failed, total)

    return summary


def main():
    """CLI entry point for parity verification."""
    parser = argparse.ArgumentParser(
        description="Verify parity between PyTorch and ONNX models"
    )
    parser.add_argument(
        "--pytorch-model-dir",
        type=str,
        required=True,
        help="Path to PyTorch model directory",
    )
    parser.add_argument(
        "--onnx-model-dir",
        type=str,
        required=True,
        help="Path to ONNX model directory",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write results JSON",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    summary = run_parity_checks(
        pytorch_model_dir=args.pytorch_model_dir,
        onnx_model_dir=args.onnx_model_dir,
        atol=args.atol,
        rtol=args.rtol,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results written to: %s", args.output)
    else:
        print(json.dumps(summary, indent=2))

    sys.exit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
