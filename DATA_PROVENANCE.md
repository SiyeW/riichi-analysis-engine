# Training data provenance

## Source

The local training corpus was downloaded from [NikkeTryHard/tenhou-to-mjai release v2.0.0](https://github.com/NikkeTryHard/tenhou-to-mjai/releases/tag/v2.0.0), source commit `1c4616eb80c653466dd04debdbf4f6706f4dba52`. The archive sizes and GitHub asset digests match the local files exactly. The same assets also appeared in v1.3.0.

The release licenses its conversion code under Apache License 2.0 and its converted datasets under Creative Commons Attribution 4.0 International. Its data notice states that the original Tenhou logs and gameplay data remain the property of their respective owners and asks users to comply with Tenhou's terms.

| Archive | Games | SHA-256 |
| --- | ---: | --- |
| `2025.zip` | 178,888 | `0274e3f2a3b4ed32cd9af6d0f9895c061c5141e63bb6c90e561f349043b7a771` |
| `2026.zip` | 12,188 | `c37af299d9c382cc45608e6a253a0c966d038335493a55bfcc06d6fdf2674816` |

The release license files and the current source repository at commit `69fb75a51c7efef3212be603227b2a58a9717237` were checked on 2026-08-15.

## Splits used by this project

| Split | Rule | Games | Selection SHA-256 |
| --- | --- | ---: | --- |
| Train | First `1/20` after sorting 2025 filenames by SHA-256 digest | 8,944 | `ae22343e8893a3219fe23ee2ff9f92f3d47095557fc0276682a13411c92f468d` |
| Validation | 2026 filenames in the MD5 bucket `[0.98, 1)`, matching Mortal's dataset preparation rule | 246 | `ec8f86c088c204bde2afc1cddd53ef71203bdf60cbd4815bdfc04aae804834d4` |

The manifests contain local paths and are not committed. Processed shards and raw logs are excluded from the repository.

## Publication checklist

- Preserve the dataset attribution and CC BY 4.0 license link.
- Recheck the source repository license and Tenhou terms at publication time.
- Publish a model card with the training-code revision, split hashes, metrics, weight hash, and weight license.
