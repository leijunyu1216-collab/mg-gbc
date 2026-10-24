# Data

The raw data are not redistributed here. Both sources are public.

## UNSW-NB15

Download the four CSV shards and the feature dictionary from

<https://research.unsw.edu.au/projects/unsw-nb15-dataset>

and place them under `data/unsw_nb15/`:

```
data/unsw_nb15/UNSW-NB15_1.csv
data/unsw_nb15/UNSW-NB15_2.csv
data/unsw_nb15/UNSW-NB15_3.csv
data/unsw_nb15/UNSW-NB15_4.csv
data/unsw_nb15/NUSW-NB15_features.csv
```

`experiments/run_unsw_nb15_temporal_compare.py` reads the shards once and
hashes the combined file; the digest is recorded in
`experiments/unsw_nb15_temporal_protocol_v2.json`.

## CIC-IDS2018

Obtain the daily capture files from the Canadian Institute for Cybersecurity
through the AWS Open Data registry,

<https://registry.opendata.aws/cse-cic-ids2018/>,

and place the days you need under `data/cicids2018/`:

```
data/cicids2018/Friday-02-03-2018.csv
data/cicids2018/Friday-23-02-2018.csv
...
```

The two days used in the paper are 2018-02-22 and 2018-02-23. File names,
row counts and SHA-256 digests are recorded in
`experiments/cicids2018_temporal_protocol_v1.json`; the loader checks them
before running.

## Small benchmark sets

Zoo, Iris, Glass, Ecoli, Yeast, Pageblocks, Segment, Covertype, Olivetti faces,
Lymphography and Dermatology are fetched by `src/datasets.py` from the UCI
repository and from OpenML (data set ids 150, 41027, 10, 35). Nothing needs to
be downloaded by hand.
