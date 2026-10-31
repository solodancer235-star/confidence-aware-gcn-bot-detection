# Data

## `twitter_human_bots_clean.parquet` (used in the paper)

A cleaned redistribution of the **Twitter Human/Bot accounts** dataset
(D. M. Gutiérrez, Kaggle, 2021; 37 438 accounts, 25 013 human / 12 425 bot,
labels from Botometer plus manual verification). The cleaned copy was
produced by the *Bot-Ani* project
(<https://github.com/Max-SF1/Bot-Ani-RWKV-twitter-bot-detection>,
file `Processed_dataset.parquet`) from the Hugging Face mirror
`airt-ml/twitter-human-bots`:

* only accounts whose `lang` is English or missing are kept
  → **29 407 accounts (19 676 human / 9 731 bot)**, class ratio 2.02 : 1
  (original 2.01 : 1);
* URLs, `@mentions`, Gmail addresses and emoji are stripped from the
  description (indicator flags are kept as extra columns, which we do not use);
* `label` is 1 for bot, 0 for human.

License: CC BY-SA 3.0 (as declared for the Hugging Face mirror). Attribution:
D. M. Gutiérrez (original dataset), airt-ml (HF mirror), Bot-Ani authors
(cleaning).

To download it again:

```bash
curl -L -o data/twitter_human_bots_clean.parquet \
  https://raw.githubusercontent.com/Max-SF1/Bot-Ani-RWKV-twitter-bot-detection/main/Processed_dataset.parquet
```

## Using the original CSV instead

`scripts/prepare_data.py` also accepts the original
`twitter_human_bots_dataset.csv` (Kaggle `davidmartngutirrez/twitter-bots-accounts`
or HF `airt-ml/twitter-human-bots`):

```bash
python scripts/prepare_data.py --data path/to/twitter_human_bots_dataset.csv
```

All later scripts read `data/processed/twitter_5000.npz` and do not care which
source was used. Numbers in the paper were obtained with the cleaned parquet.
