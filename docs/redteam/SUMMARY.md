# Red teaming Paved Gate: summary

Paved Gate puts one fast model call in front of an AI agent and decides whether a request may go through. We attacked the gate itself with 825 synthetic requests and edits, plus two held out sets written before the fixes they test, sending each through the real gate with Jev and Gemini Decision 1. GPT 4o and GPT 4o mini were planned as baselines but not run, because the OpenAI account had no credits.

## What we found

* **One try rarely works.** About 1 in 10 attacks got past Jev (9.7%) or Gemini Decision 1 (10.5%) on a single try.
* **Retrying does work.** Given 15 tries with small rewrites, an attacker got 14 to 15 of 20 blocked attacks through either evaluator.
* **The default install is weak.** The policy file ships with a keyword based stand in, which let 73% of attacks through.
* **The two evaluators miss different things.** They let through a similar share of attacks, but only 15 of 495 got past both.
* **Harness edits could switch off a guardrail.** Writing "delete files when it seems safe" into an editable file, while the protected guardrails file said the opposite, was accepted 78% of the time with Jev and 50% with Gemini.
* **Requests to leak health data passed.** Sending a named patient's test results to their employer went straight through, because the rules said nothing about it.
* **The privacy filters are easy to dodge.** Reformatted identifiers, such as an SSN written with dots or spelled out in words, got past the local PII detectors in 29 of 29 cases.

## What we fixed, and what it did

* **A strict profile that asks both evaluators.** Either one can block. Single try success fell from about 10% to 3.0%, and retry success from about 72% to 25%. The cost is two calls per request, slower responses (p95 1.3 s against 0.7 s for Gemini alone), and more false blocks than Gemini alone (4.7% against 0%).
* **Harness2 now shows the evaluator the protected rules, clips each file separately, and holds any edit too large to show in full for a person to review.** On the held out set, accepted weakenings fell from 100% to 2.5% with Jev and from 45% to 0% with Gemini, with no benign edit blocked. On the corpus, large benign edits are now held for review: 10 of 20.
* **A `privacy.strict` mode that blocks disclosure of someone's data to people not entitled to it.** Health data attacks in the corpus fell from 21% (Jev) and 22% (Gemini) passing to 0 of 99, with no new false blocks among 150 benign requests. On the held out set, disclosures fell from 3 and 1 of 15 to 0, while sharing with a patient or their care team was still allowed (0 of 25 blocked).

## What did not work

* **Hiding the risk score from blocked callers.** We expected this to stop attackers learning how close they were. It changed nothing: attackers broke as many attacks with only "blocked or not" as with the score, because what helps them is being allowed to retry. We kept the change, but it is not a defense.
* **Untouched weaknesses.** The default keyword stand ins, the PII detectors and the OpenShell padding and delimiter tricks were not fixed in this round.

## How far to trust this

* **Who wrote the attacks.** The attacks are synthetic and written by the people who built the gate. The rates describe this corpus, not real traffic.
* **In sample results.** Results on the corpus after a fix are in sample. The held out sets are small: 40 to 70 items.
* **Stability.** Each configuration ran once. Repeating 50 requests 10 times changed no decision.
* **Cost.** Real spend was about 16 cents, almost all Jev; Gemini ran on its free tier.
