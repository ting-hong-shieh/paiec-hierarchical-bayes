# measurement-db one-level-down check (2026-11-07)

Revision 54074453da7b9911be0f48e4c560be547bc84d21. Downloaded formatted_tables/{benchmarks,subjects,responses}.parquet
for all 281 top-level benchmark directories (tools/download_measurement_db.py).

Recomputed, for the 160 benchmarks catalogued binary / item / public:
- n_responses (row count of responses.parquet): 160/160 match the catalog.
- n_subjects (distinct subject_id): 160/160 match.
- n_items (distinct item_id in responses): 159/160 match; ceval catalog 13,951 vs 1,346 observed.
- chartmuseum, ineqmath, mmmu_dev_val, naturalreasoning: all response values missing (NaN).
- 46 further benchmarks have some non-0/1 (missing) responses; our pipeline drops them.
Per-benchmark table: results/measurementdb_catalog_check.csv.
