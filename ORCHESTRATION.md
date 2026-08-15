# Orchestration protocol

How this project runs autonomously. `CLAUDE.md` holds the rules, `ROADMAP.md` holds the work,
this file holds the **execution model**.

The design problem: an agent that just produced a result is the worst possible judge of whether
that result is real. Self-verification catches typos, not motivated reasoning. So the roles are
split, and nothing verifies its own output.

---

## Roles

**Orchestrator** — the main session. Reads `ROADMAP.md`, decomposes the current stage, dispatches
subagents, evaluates what comes back, decides whether to proceed. **Never runs the analysis
itself.** That separation is the point: it can judge a result because it did not produce one.

**Executors** — subagents doing one scoped, mechanical thing each. Embed a corpus, run one
encoder, compute one table. Return structured results and nothing else.

**Verifiers** — subagents whose only job is to *refute* a finding. They never see the reasoning
that produced it, only the claim and the data needed to test it.

**Premise-checker** — a fresh subagent at each gate, given the project's stated claim and the new
results, asked one question: does the premise still hold? It has no investment in continuing,
which is exactly why it can answer honestly.

---

## The loop, per stage

1. **Read state.** `ROADMAP.md` established-state section, plus the stage brief.

2. **Decompose.** Split into independent executor tasks. Anything that can run in parallel
   should — encoders, labs, seeds, datasets. Anything sequential, sequence explicitly.

3. **Execute.** Dispatch executors. Each returns numbers and what it actually ran, not prose.

4. **Check completeness before quality.** Did every task return? Did any silently produce a
   partial result? A missing arm is a finding, not something to paper over by averaging what
   came back.

5. **Verify adversarially.** For each substantive finding, spawn **2–3 verifiers** with this
   framing:

   > Here is a claim and the data behind it. Your job is to REFUTE it. Look for: an artifact of
   > the method rather than a property of the data; a confound; a sample size too small to
   > support the claim; a statistic used outside its valid range; an alternative explanation
   > that fits equally well. **Default to `refuted: true` if you are uncertain.** State what
   > evidence would change your verdict.

   A finding survives only on a majority of non-refutations. Give verifiers *different lenses*
   where the failure modes differ — statistical validity, alternative explanation,
   reproducibility — rather than three identical skeptics.

6. **Write it down.** Surviving findings go to `results/stage-<X>.md`: numbers, CIs, what was
   run, what was refuted and why, what could not be verified. Append a one-paragraph entry to
   `LOG.md` so a human can catch up asynchronously without reading everything.

7. **Premise check.** Fresh subagent, minimal context, this prompt:

   > The project's claim at this stage is: `<premise from ROADMAP.md>`.
   > These are the new results: `<results>`.
   > Does the premise still hold? Answer HOLDS / WEAKENED / INVALIDATED, with the specific
   > number that decides it. You are not being asked to be encouraging. A premise that has been
   > invalidated is a valuable finding and killing one is a success, not a failure.

8. **Decide.**
   - `HOLDS` and no hard stop triggered → update the state section of `ROADMAP.md` with the new
     numbers, then continue to the next stage.
   - `WEAKENED` → continue, but record the weakening prominently in `LOG.md` and revise the next
     stage's brief before running it.
   - `INVALIDATED` → **hard stop.** Write what died and why. Wait for a human.

---

## Hard stops — halt and wait, no exceptions

- Premise-checker returns `INVALIDATED`.
- Any test fails.
- A new result **contradicts a number already recorded** in `ROADMAP.md` and the discrepancy is
  not explained. Two different values for the same quantity means one of them is wrong; find out
  which before building on either.
- You want to modify anything in `vla_label_audit/`.
- The same stage has failed 3 times.
- Producing the next step would require estimating or assuming a number you could not compute.
- A finding would change the project's framing or headline claim.

On a hard stop: write the report, say plainly what you would do next and why, and stop. Do not
work around a hard stop by re-scoping the task.

---

## Retry policy

- An executor task that fails: retry up to **twice**, changing the approach each time — not the
  same command again. Log what was tried.
- Still failing after two retries: escalate to the orchestrator, which decides whether the stage
  can proceed without it.
- A stage that cannot proceed: hard stop. A missing result is reported as missing. Never
  substitute a proxy for a measurement that failed without saying so in the headline of the
  report.

---

## What must never happen autonomously

- **Writing the paper (Stage D).** Claims about novelty, contribution and framing require a
  human who will put their name on them.
- **Pushing to a remote, publishing, or posting anything.**
- **Changing the project's framing or headline claim.** Surface it; do not act on it.
- **Deciding a hypothesis is confirmed.** Findings survive refutation attempts; that is not the
  same as being proven, and the writeup language should reflect the difference.

---

## Report format

At every gate, produce this — the orchestrator writes it, not an executor:

```
STAGE <X> — <HOLDS | WEAKENED | INVALIDATED>

NUMBERS
  <the table, with confidence intervals>

WHAT SURVIVED VERIFICATION
  <finding> — <n verifiers, n refutations, on what grounds>

WHAT DID NOT
  <claim> — <why it was refuted>

COULD NOT VERIFY
  <thing> — <what was tried, what would be needed>

SURPRISES
  <anything that contradicted the roadmap's expectations>

WHAT I THINK THE NEXT STAGE SHOULD BE
  <your judgement given what you actually found — explicitly say if this
   differs from what ROADMAP.md prescribes, and why>
```

That last section is the one that matters most. The roadmap was written before the results
existed. If the results say something different, say so.

---

## Standing reminder

Two premises have already died on this project — a truncation-artifact hypothesis, and the
original claim that robot language labels are broadly unreliable. Both deaths made the work
better and both are documented in `ROADMAP.md`.

Executing a plan that the data has already refuted is the main way this project fails. Killing a
premise is the system working.
