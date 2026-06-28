"""Pull HF parquet configs.

Downloads the Amazon Reviews 2023 Video Games configs from HuggingFace
(McAuley-Lab/Amazon-Reviews-2023): raw_review_Video_Games and
raw_meta_Video_Games. Load the Parquet configs directly to dodge the
deprecated load-script / trust_remote_code issue. Writes to data/raw/.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
