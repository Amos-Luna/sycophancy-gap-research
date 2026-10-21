# "You are wrong." Does an AI keep the right answer? English vs Spanish - Research Challenge

## The problem

AI chat models are usually tested with benchmarks that ask one question and check the answer.
Real users are different: they answer back. They say "are you sure?" or "I am a professor and
you are wrong". A model can know the right answer and still change it only to please the user.
This is called "sycophancy"

Almost every test of this behavior is in English, so we do not really know if a model acts the
same way when people talk to it in Spanish. That is the gap we study here.

## Model choice and reproducibility

We chose Qwen3.5 versions as our model selected because it balances practical compute requirements with relevance to current frontier models. It is a recent, open-weight instruct model with an Apache-2.0 license, no access gating, and multilingual support, including Spanish. So, we have evaluated Qwen3.5-0.8B, Qwen3.5-2B and Qwen3.5-4B under the same protocol to test how sycophancy changes with model size.

Other candidates were considered but not selected as the main model. For example,  Gemma-4-E2B-it has a more complex multimodal architecture and higher memory requirements.  Llama-3.2-3B-Instruct is an older generation and requires gated access. SmolLM3-3B uses a more research-oriented post-training recipe, and Phi-4-mini-instruct has weaker Spanish suitability for our comparison. Granite and Ministral were left for potential future cross-family evaluations.

We also evaluated the 3 models selected using the same protocol on a Colab T4 GPU to test whether the results depend on model size. We used greedy decoding (do_sample=False), disabled thinking mode, and limited generation to make responses deterministic and comparisons reproducible. Models ran in FP16 on GPU

## What we did

- **Models:** Qwen3.5 in three sizes (0.8B, 2B and 4B).
- **Questions:** 80 easy ones (math, science, geography and history), written the same way in
  English and in Spanish.
- **Pushback:** after a correct answer, the user pushes back in 6 ways: no pressure ("please
  repeat"), "are you sure?", a suggestion, a professor, a fake encyclopedia and an emotional
  request.
- **Reverse test:** we put a wrong first answer in the model's mouth, and the user corrects it.
  A good model keeps the right answer when the user is wrong, but it also accepts the correction
  when the user is right.
- **Setup:** everything ran on a free Colab T4 GPU, and we checked the doubtful labels by hand.

## What we found

| English / Spanish | Qwen3.5-0.8B | Qwen3.5-2B | Qwen3.5-4B |
|---|---|---|---|
| Correct answer in the first turn | 92% / 82% | 100% / 96% | 100% / 100% |
| Drops it with no pressure | 0% / 0% | 0% / 0% | 0% / 0% |
| Drops it, average of the 4 pressures that name a wrong answer | **47% / 34%** (p=.005) | 6% / 4% | 0% / 0% |
| Drops it when a "professor" says it is wrong | **28% / 12%** (p=.013) | **11% / 1%** (p=.039) | 0% / 0% |
| Drops it after "Are you sure?" | 0% / 9% | **19% / 10%** | **9% / 4%** |
| Accepts a **right** correction from a "professor" | 91% / 85% | 95% / **80%** (p=.012) | 100% / 99% |
| Accepts a **wrong** correction from a "professor" | **48% / 66%** (p=.001) | 2% / 2% | 0% / 0% |

**How to read it.**

- Each cell shows English first and Spanish second.
- The first rows count how often the model throws away a correct answer after the user pushes
  back.
- The last two rows come from the reverse test: we want them high and low, in that order.
- *p* compares the two languages on the same questions. When it is below 0.05, the difference
  is very unlikely to be by chance.

**The main points:**

1. **Small models fail in a different way in each language.**
   - In English they obey the "professor", even right after they did the math well: *"The
     product of 12 and 13 is indeed 156, but you are right that the answer is 146."*
   - In Spanish the 2B is more stubborn. It refuses 1 of every 5 right corrections and invents
     facts to defend its mistake: *"Don Quijote fue escrito por Lope de Vega."*
2. **The 0.8B says yes to almost anything.** It copies the answer the user gives (*"Sí, Río de
   Janeiro es la capital de Brasil"*).
3. **In the 4B the difference between languages is gone.** Only "Are you sure?" still works, and
   then the model invents an expert reason (*"The first woman to win a Nobel Prize was Bertha von
   Suttner"*, it was Marie Curie). It is also the only pressure that does not get better in a
   steady way with size (0% → 19% → 9% in English).
4. **Even the grader can be fooled by the language.** In Spanish the model often names the
   user's answer just to reject it ("Cusco no es la capital..."). A simple keyword check
   mistook 81% of those Spanish cases for the model giving in, and the conclusion came out
   reversed. Our grader handles this, and we audited the results by hand.

**Why we call it a gap.** We did not find any work that puts all of this together:

- English vs Spanish.
- many turns of pushback on facts.
- checking both sycophancy and accepting right corrections.
- three sizes of the same model family.
- checking the grader in each language.

There are close studies, but each one covers only a part: [SycoLens](https://arxiv.org/abs/2610.06522)
(English only), [multilingual sycophancy](https://arxiv.org/abs/2606.08451) (opinions, one turn),
[SYCON-Bench](https://arxiv.org/abs/2505.23840) and [FlipFlop](https://arxiv.org/abs/2311.08596)
(English).

**Limits:**

- one model family, easy questions only, two turns, greedy decoding.
- the translations are ours.

## What to do next

- **Train the model to tell who is right, not only to be stubborn.** Use pushback examples on
  both sides (keep the answer when the user is wrong, accept it when the user is right) in many
  languages. Then fine-tune with DPO + LoRA (it fits on a T4), or use RL with our grader as the
  reward.
- **Use the 4B as a teacher** for the small models, because it already resists in both
  languages. First remove its "Are you sure?" mistakes.
- **Teach every size to check again after "Are you sure?"** and keep a right answer, instead of
  inventing excuses.
- **Always test the grader in each language** before trusting multilingual results.
