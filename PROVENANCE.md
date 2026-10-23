# Provenance of the protocols

Each campaign of the paper ran under a protocol that was fixed before the campaign. This file lists, for each
protocol in this repository, its SHA-256, the date it states, and the commits of the authors' private repository in
which the protocol and the campaign's first results first appear.

Four facts come first.

- Only the two benchmark-suite protocols, of the benchmark suite and of the ramped price, were committed and hashed
  before their runs. The other protocols were committed together with their results. For them, the only record of
  when they were fixed is the date each protocol states.
- The fuel-tank protocol (`fuel-tank/lunar-protocol.json`) lists the configurations, with a note on the arms added
  during the campaign. The fuel-tank records hold no text of the study's predictions, which appear only in the
  manuscript.
- The solvable flow has no protocol file. Its first records appear in the private commit 46fd3f2 (2026-10-03 19:43).
- This repository was assembled on 2026-10-15 and is published as one snapshot of 2026-10-23; its working history is
  kept in the authors' records. The commit hashes below refer to the authors' private repository, except for the
  three protocols of 2026-10-20, which were written in this repository's working history. The study folders moved
  in the private repository on 2026-10-13, so the commits were found with `git log --follow`.

## Protocols and commits

Times are those of the commits, in EEST (UTC+3).

| Study | Protocol (path in this repository) | Date the protocol states | Protocol first committed | First results committed | Protocol committed before the results |
|---|---|---|---|---|---|
| Wear allowance | `wear-allowance/PROTOCOL.md`, `wear-allowance/protocol.json` | fixed 2026-10-02 (`protocol.json` states none) | 4805a77, 2026-10-03 11:35 | 4805a77 | no, same commit |
| Fuel tank | `fuel-tank/lunar-protocol.json` | none (its note dates the added adaptive arms 2026-10-03) | 46fd3f2, 2026-10-03 19:43 | 46fd3f2 | no, same commit |
| The prior on a learned controller | `wear-allowance/PROTOCOL-anchor-v41.md` | fixed 2026-10-08 | 0814f00, 2026-10-08 13:27 | 0814f00 | no, same commit |
| Roles of the regularizers on PPO | `wear-allowance/PROTOCOL-noise-v60.md` | fixed 2026-10-10 | e6b1891, 2026-10-10 20:34 | e6b1891 | no, same commit |
| Benchmark suite | `benchmark-suite/PROTOCOL-safety-v63.md` | fixed 2026-10-12 | 5df3f27, 2026-10-12 13:58 (a draft in 9d52744, 13:52) | 06409b6, 2026-10-12 22:34 | yes |
| Ramped price (mentioned but not reported in the paper) | `benchmark-suite/PROTOCOL-ramp-v65.md` | fixed 2026-10-13 | a29965a, 2026-10-13 16:55 | 1694467, 2026-10-13 21:03 | yes |
| The pull decoupled from Adam (after revision 68) | `wear-allowance/PROTOCOL-decoupled-v68.md` | fixed 2026-10-20 18:01 | this repository, with its results | same commit | no, same commit; the hash file `PROTOCOL-decoupled-v68.sha256` was written at 18:01 before the launch |
| The regularizers under three optimizers (after revision 68) | `wear-allowance/PROTOCOL-optimizers-v68.md` | fixed 2026-10-20 18:27 | this repository, with its results | same commit | no, same commit; the hash file `PROTOCOL-optimizers-v68.sha256` was written at 18:27 before the launch |
| Amendment of the three-optimizer protocol: the revision-60 rows C7-C9, and the application on six seeds, whose records are not in this repository (after revision 68) | `wear-allowance/PROTOCOL-optimizers-v68-AMENDMENT.md` | fixed 2026-10-20 19:38 | this repository, with its results | same commit | no, same commit; the hash file `PROTOCOL-optimizers-v68-AMENDMENT.sha256` was written at 19:38:01, the runs launched at 19:38:13 |

The first results of the benchmark suite in 06409b6 are the campaign's analysed summary (`safety-summary-v63.json`).
Its per-run records entered the private repository in 04bf5ce (2026-10-13 16:26).

The amendment `benchmark-suite/PROTOCOL-safety-v63-AMENDMENT.md` states 2026-10-12, during the campaign. It was first
committed in 548273a (2026-10-12 14:37) with its items 1 and 2, which change only the launcher, before the
campaign's results. Item 3 states that it was added after the campaign, at 22:40, and first appears in 06409b6.
Item 4, a move of the Python runtime dated 2026-10-13, first appears in 082048f (2026-10-13 10:52).

The amendment `wear-allowance/PROTOCOL-optimizers-v68-AMENDMENT.md` was written after the three-optimizer campaign and
its report, and hashed before its own runs. A first hash file (18bc7cf3…) was written at 19:37:46 while the header
stated the intended time 19:45; the header was corrected to the actual time 19:38, the file rehashed at 19:38:01 and
the runs launched at 19:38:13. Nothing else in the file changed between the two hashes.

## SHA-256

Each private original below is unchanged since its first commit, with two exceptions. The benchmark-suite protocol is
unchanged since its hashed version of 5df3f27, and the amendment gained its items 3 and 4 as described above. The hashes recorded at the time (`wear-allowance/PROTOCOL-noise-v60.sha256`,
`benchmark-suite/PROTOCOL-safety-v63.md.sha256` and `benchmark-suite/PROTOCOL-ramp-v65.md.sha256`) equal the
SHA-256 of the private originals.

| Path in this repository | SHA-256 of the private original | Copy here |
|---|---|---|
| `wear-allowance/PROTOCOL.md` | 535fec4c45c1604fc5692504a383e2d7f46307f162fa37cf373bb2fa8c1cea48 | identical |
| `wear-allowance/protocol.json` | aca70b250be09f91ec0396bd8583c03508333a75a8b82635cd63eb430334d136 | identical |
| `fuel-tank/lunar-protocol.json` | ca89738c6fb934b7baac805409956ade085e8795a17d44880c0b598f0313cba3 | identical |
| `wear-allowance/PROTOCOL-anchor-v41.md` | 023f656a22b70af10d4908548b70d2b071cafdc3ef846946660c729918ffb4ee | identical |
| `wear-allowance/PROTOCOL-noise-v60.md` | d26c07a0bfbc305ac989093b45eb15e28bed71e12049eea6cd64d671d7e5c893 | identical |
| `benchmark-suite/PROTOCOL-safety-v63.md` | 11d4a4e55f338fff95fd5e75c05395a4247f3259416ceb7cf7cb77decf230c70 | identical |
| `benchmark-suite/PROTOCOL-safety-v63-AMENDMENT.md` | 20e464a7843b3a39fb64fdca1bdbf6e575c0896c2b770b48eb26fe3ef1fb1740 | identical |
| `benchmark-suite/PROTOCOL-ramp-v65.md` | e484616cefb2dfae38472a4a04f75a8af22f83ab84ba2040f07b70eac38c6f63 | identical |
| `wear-allowance/PROTOCOL-decoupled-v68.md` | 81566cfda5c7fae62f2c63e829b7c2eedb28bc25dd237e5349a4b4263a6587b0 (written in this repository) | the original |
| `wear-allowance/PROTOCOL-optimizers-v68.md` | a138bd9de66b084b17539cb0df42b138c3ff278fbd41008e317afd9fff0284c9 (written in this repository) | the original |
| `wear-allowance/PROTOCOL-optimizers-v68-AMENDMENT.md` | 1ab654475bf5ab4ebe02b5ee12bbb6d096866112475f2ef1d660a2f699fe69b6 (written in this repository) | the original |

## Changes made before publication (2026-10-23)

The published snapshot differs from the working history in two ways.

- The emissions application, which revision 72 of the paper removed, was taken out: its folder `application/`, and
  `wear-allowance/wear-optimizers-v68-application.tex`, the table of its arm in the three-optimizer amendment. Its two
  protocols, published until then as redacted copies, are therefore no longer listed above. The protocols and reports
  of 2026-10-20 still describe its runs. Two notes on the internal revision history, `HISTORY-numerics-readme.md` and
  `FIGURES-v41-REPORT.md`, were also taken out.
- The home folder of the authors' working tree, `/Users/<user>/<synced folder>/Constrained-RL-project`, was replaced
  by `<project>` wherever a run record or log recorded it: 65 result files of the fuel-tank campaign
  (`fuel-tank/lunar/campaign/*/result.json`), three fuel-tank campaign logs, three wear-allowance pilot logs and
  three benchmark-suite logs. Nothing else in these files changed, and no protocol or hashed file contained the path,
  so every hash above still holds.
