"""Score one record: python examples/scoring.py --model-id scm_... --score-id your_unique_id.

Requires FINIC_API_KEY. The record fields must match your model's training data.
"""

import argparse

from langchain_finic import FinicScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--score-id", required=True)
    args = parser.parse_args()
    with FinicScorer(model_id=args.model_id) as scorer:
        result = (scorer | (lambda result: result.model_dump())).invoke(
            {"score_id": args.score_id, "data": {"amount": 200, "new_device": True}}
        )
        print(result)


if __name__ == "__main__":
    main()
