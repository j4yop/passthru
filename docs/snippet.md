# Utterance boundary snippet

Passthru splits dictated text into utterances so each one can be scored on its own. Without
a boundary it falls back to sentence splitting, which is a guess. A spoken marker makes the
split reflect what you actually said.

Create it once, in **Wispr Flow -> Snippets**:

| Field | Value |
|---|---|
| Trigger phrase | `end utterance` |
| Expansion | `--- end ---` |

Say the trigger at the end of each dictated spec. Passthru strips it before scoring.

**On bulk import:** Wispr supports importing snippets and dictionary entries from a file, but
only the in-app help documents the exact schema, and the import screen is not reachable
without the relevant plan. Rather than ship a JSON file whose format is a guess and would
fail silently on import, it is written out here. It takes ten seconds to create by hand and
cannot be wrong.

The marker words are configurable in `score.DEFAULT_STRIP` if you prefer different ones.
