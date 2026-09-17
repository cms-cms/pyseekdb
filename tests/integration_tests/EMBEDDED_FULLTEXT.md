# Embedded full-text stability (seekdb #1382)

Run the same tests natively on Linux x86_64 and macOS arm64 with a selected
`pylibseekdb` wheel and this PySeekDB source checkout:

```sh
SEEKDB_TEST_DATA_ROOT=/absolute/path/on/local-disk/fts-data \
SEEKDB_TEST_ARTIFACT_ROOT=/absolute/path/to/artifacts \
python -m pytest tests/integration_tests/test_embedded_fulltext_stability.py -v -s
```

Use real local disk, not a Linux tmpfs mount that rejects direct I/O. Each case
uses a new subprocess and database directory; it does not reopen an already
warmed database. The two cases participate in the existing `-k embedded` suite.

* `first-pass`: 24,515 deterministic synthetic documents, supplied 8-dimensional
  embeddings (no model/network download), SDK upsert and refresh_index, followed
  immediately by four passes of 300 public `collection.hybrid_search` full-text
  queries (`knn=None`). Pass zero is included in the verdict. No warm-up query,
  sleeps or retries erase an initial failure.
* `scan-freeze`: a separate table with a full-text index and **no vector index**,
  24,515 preloaded documents, eight independently connected scanning threads,
  and one writer inserting 32 rows then executing `ALTER SYSTEM MINOR FREEZE`
  every approximately 120 ms. A barrier starts the 75-second window only when
  all connections are ready. Each scanner must execute successfully and at
  least one freeze must complete; all errors, not just 4016, fail the test.

This preserves the original API failure surface and the accelerated scan/release
trigger separately. It is **not** a byte-identical replay of the original
SciFact/384-dimensional embedding benchmark, and a single passing stress run
does not establish that a probabilistic bug is absent.

`summary.json` records runtime/version, counts, first failure and extracted trace
IDs. `errors.jsonl` preserves up to 200 failures, while the total is unbounded.
The first error immediately triggers bounded database-log context collection;
it does not wait for the pressure window to end. Log scans use a total byte
budget and include tails of rotated files larger than 256 MiB. Inventories
explicitly report scan/output truncation. `last_trace_id()` is best-effort on
the failing connection, not a guaranteed mapping to an SDK internal statement.

`max_syslog_file_count=50` bounds the normal rotated set; it is not a guarantee
of long-term retention. Successful cases remove only their generated database
directory after their worker exits and diagnostic collection completes; failed
cases retain it. Evidence lives outside that directory. CI should archive it
before any later retention cleanup.

For pre/post-fix comparisons, hold this test SHA and bindings source SHA fixed,
record the actual SeekDB REVISION inside each wheel, and change only the kernel
source revision. Old-build failures must retain a failed result. Do not add an
xfail or suppress 4016 to make the comparison green.

## GitHub PR gate and released wheels

The GitHub embedded job builds a **CI-local Linux x86_64 / CPython 3.11** wheel
from the full SeekDB and bindings commits in `.github/embedded-source.json`.
The pinned SeekDB commit contains the #1384 fix; the build verifies that ancestry.
This wheel is not published and does not claim portable manylinux compatibility.
The kernel build uses at most three workers, further limited by available RAM.

The cache key includes both source pins, the build/verification scripts, Ubuntu
22.04 and CPython 3.11. No approximate cache key is accepted. On every restore,
CI verifies the manifest, wheel and embedded binary SHA-256 and full REVISION.
After installation it verifies the actual installed binary again. Integration
tests and examples use `uv run --no-sync` so the lock file cannot silently
replace the tested wheel with the old PyPI release. All embedded tests still run;
neither these two regressions nor database errors are skipped or marked xfail.

The lock file remains unchanged for the other jobs. This source-built gate does
**not** prove that the published wheel is fixed: run 35204502666 used the locked
`pylibseekdb 1.3.0` and failed both new scenarios with explicit 4016 errors (one
first-pass error and 72 scan errors, whose first error was 4016). A released wheel
containing #1384 can replace the temporary source-build path after verification.
Failed full-text reports and bounded database-log excerpts are uploaded as
artifacts; whole database directories are not uploaded.
