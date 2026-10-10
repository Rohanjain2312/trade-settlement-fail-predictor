<!--
Medium draft, version 1 (2026-10-09). These notes do not show on GitHub and are not part of the article.
- Images: download them from the links in the article and upload them to Medium. Keep the captions.
- Cover image: the resampling chart.
- Tags (5): Machine Learning, Data Science, Imbalanced Data, XGBoost, Fintech.
- Publish as a free story (not member-only), so recruiters can read it.
- Every number comes from the final full run in report/RESULTS.md on the Hugging Face model repo.
-->

# Predicting Trade Settlement Fails: What I’d Do Differently, From SVM + SMOTE to XGBoost + SHAP

*I rebuilt a model I first made at my workplace, this time on 3.1 million synthetic trades where I knew the right answers. Here is what changed, in plain English.*

---

A few years ago, at my workplace, I built a model to spot trades that were likely to fail. It was a support vector machine (SVM) trained with SMOTE, a popular trick for problems where the thing you are looking for is rare. It did its job.

Tools have changed a lot since then, and I kept wondering: if I built it again today, what would I do differently?

So I rebuilt it from scratch. One important note before we start: **every trade in this project is synthetic.** I can't use real trade data, so I wrote code that generates realistic fake trades. That turned out to be a big advantage. Because I wrote the rules that decide which trades fail, I could check whether each model actually found those rules. You can never do that with real data.

Here is the short version of what I'd change:

1. **Skip SMOTE. Use class weights and choose my own cutoff.** SMOTE lost to class weights in every model I tried.
2. **Use XGBoost instead of an SVM**, with logistic regression as a simple baseline to beat.
3. **Turn scores into real probabilities** (calibration), using data the model never trained on.
4. **Explain every alert with SHAP**, so the operations team knows what to fix.
5. **Set the number of daily alerts from the team's capacity**, not from a fixed probability cutoff.

The rest of this article walks through each one. No heavy math, I promise.

## What is a settlement fail?

Think about buying a house. You agree on a price today, but the money and the keys swap hands later, on closing day. If the buyer's money hasn't arrived, or the paperwork has the wrong bank account, closing day slips.

Trades work the same way. When a fund buys shares, the shares and the cash swap hands a day or two later, on the **settlement date**. If either side doesn't arrive on time, the trade **fails to settle**.

Fails are expensive. Money gets stuck, penalties pile up, and every fail lands on the desk of an operations ("ops") team that has to chase it. That team can only check a small number of trades each day.

So the real question isn't "will this trade fail, yes or no?" It's **"which trades should we check first?"** The useful output is a ranked list, riskiest trades at the top, so ops can fix problems before settlement day.

## Why accuracy is the wrong score

In my test period, 4.5% of trades failed. A "model" that simply says "every trade will settle" is right 95.5% of the time. That sounds great, and it is completely useless: it never catches a single fail.

So I used scores that focus on the fails:

- **PR-AUC.** Walk down the ranked list. At each point, ask two questions. Of the trades I flagged so far, how many were real fails (precision)? Of all the fails, how many have I caught (recall)? PR-AUC rolls this into one number between 0 and 1. A random list scores about the fail rate, here 0.045. Higher is better.
- **Recall in the top 2%.** If ops can check 2% of trades a day, what share of all fails is inside that 2%? A random pick would catch about 2%.
- **Brier score.** When the model says "30% chance of failing", does that happen about 30% of the time? Lower is better.

## Building fake data that is still useful

The generator creates about **3.1 million trades** over 24 months, between 300 made-up counterparties (the firms on the other side of each trade) and 5,000 made-up securities. About **3% of trades fail**.

Each trade has 20 features. Here are some of them in plain words:

- **Do the settlement instructions match?** These are the account details that say where the shares and cash should go. If they don't match what's on file, nothing can move.
- **How long did the other side take to confirm the trade?**
- **Do we actually have the shares or cash we owe?** A value below 1 means we're short.
- **How often has this counterparty failed in the last 30 days?**
- **Is the trade cross-border?** More middlemen, more time zones, more things to go wrong.
- **How turbulent is the market today?**

Behind the scenes, a hidden "recipe" turns these features into a chance of failing. Each feature pushes the risk up or down by an amount I chose. Three combinations are extra risky together:

- wrong account details **and** a cross-border trade
- short on shares **and** a security that has been failing a lot lately
- a slow confirmation **and** an instruction sent overnight

To keep it honest, I also added the mess you find in real life: hidden risk that isn't in the data, random noise, near misses (a broken instruction that gets fixed just in time), and a small group of fails (about 1.5%) with no visible cause at all. No model should be able to get everything right.

**What I'd do differently:** test a method on data with a known answer key before trusting it on real data. You learn very quickly whether your model finds real signals or just noise.

## Three ways to cheat by accident

Data leakage is when information from the future, or from the test set, sneaks into training. It's like a student who has seen the exam answers: great practice scores, then a shock on the real exam. Here are the three traps I guarded against.

**1. Features that peek into the future.** "How often has this counterparty failed in the last 30 days?" sounds harmless. But a trade made yesterday may not have settled yet, so nobody knows its outcome. The feature must only count trades whose settlement date is *before* today.

**2. Shuffled splits.** If you shuffle all trades and split them at random, the model trains on trades from next month and gets tested on trades from last month. Real life doesn't work that way. I split by date instead: the first 16 months for training, the next 4 for validation, and the last 4 for testing, with a gap of 5 business days between them so no trade is still settling across a boundary.

**3. SMOTE before the split.** This one deserves a picture.

SMOTE creates fake failed trades so the model sees more examples of the rare class. It picks a real failed trade, finds a similar failed trade, and places a new fake trade somewhere on the line between them. (I used SMOTENC, the version that also handles categories like "asset class".)

![SMOTE draws new fake fails on the line between real fails](https://huggingface.co/rohanjain2312/trade-settlement-fail-predictor/resolve/main/report/smote.png)
*Red dots are real fails. Orange diamonds are fake fails created by SMOTE, each one between two real ones. Synthetic data.*

If you run SMOTE on the full dataset before splitting, some fake trades are built from test trades. The model then trains on near copies of the test set. I measured it: when SMOTE ran first, the typical test fail had a fake fail **about 30 times closer** to it (median distance 0.07 versus 2.19) than when SMOTE ran on training data only.

The fix is to make SMOTE a step inside the model pipeline, so it only runs when the model is trained:

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from sklearn.svm import LinearSVC

pipe = Pipeline([
    ("encode", encoder),   # turn categories into numbers
    ("smote", SMOTENC(categorical_features=cat_cols, random_state=42)),
    ("scale", scaler),     # one-hot encode and scale
    ("model", LinearSVC()),
])

pipe.fit(X_train, y_train)  # SMOTE runs here, on training rows only
pipe.predict(X_test)        # SMOTE never touches test rows
```

Because this is an `imblearn` Pipeline, it also stays safe during cross-validation: SMOTE only ever sees the training part of each fold.

**What I'd do differently:** turn each of these rules into an automated test. In this project, the build fails if a feature uses a future outcome, if a trade straddles a split, or if SMOTE touches anything but training data. Good habits are great. Tests are better.

## The big surprise: SMOTE didn't help

I trained four kinds of models (logistic regression, a linear SVM, a curved "RBF" SVM, and XGBoost), each in three ways:

- **No resampling:** train on the data as it is.
- **Class weights:** tell the model that missing a fail costs much more than a false alarm. No fake data.
- **SMOTE:** add fake fails until the two classes are balanced.

![Test PR-AUC by model and resampling method](https://huggingface.co/rohanjain2312/trade-settlement-fail-predictor/resolve/main/report/resampling.png)
*SMOTE (green) was never the best option for any model. Synthetic data, test period.*

| Model | No resampling | Class weights | SMOTE |
|---|---|---|---|
| Logistic regression | 0.305 | 0.288 | 0.244 |
| Linear SVM | 0.296 | 0.290 | 0.245 |
| RBF SVM | 0.213 | 0.319 | 0.218 |
| XGBoost | 0.333 | 0.332 | 0.189 |

*PR-AUC on the test period. Higher is better; a random list scores about 0.045.*

Class weights beat SMOTE in every model. For XGBoost, SMOTE dropped PR-AUC from 0.333 to 0.189.

So why did SMOTE look so useful back then? Because it changes something else: **how willing the model is to say "fail".**

A linear SVM trained on the raw data, where only 3% of trades fail, almost never says "fail". With its built-in cutoff, it flagged just 0.01% of trades and caught 0.2% of the fails. After SMOTE, the same model flagged 39% of trades and caught 84% of the fails. That looks like a huge improvement.

But SMOTE didn't make the model better at telling risky trades from safe ones. It mostly moved the cutoff. You can get the same effect without inventing any data: keep the model's scores and choose the cutoff yourself. (And no ops team can check 39% of all trades anyway. More on that below.)

This is one dataset, and it is synthetic, so SMOTE may still help elsewhere, for example on very small datasets. But it's worth testing instead of assuming.

**What I'd do differently:** use class weights (or nothing), then choose the cutoff on purpose. Fake data adds training time and leakage risk, and here it bought nothing.

## From SVM to XGBoost

**How an SVM works.** Picture fails and non-fails as dots on a page. An SVM draws the widest possible street between the two groups. Only the dots right at the edge of the street, the "support vectors", decide where the street goes. The RBF version can draw curvy streets instead of straight ones.

The catch is size. A curvy SVM compares trades with each other in pairs, so the work grows roughly with the square of the number of trades: ten times the data is about a hundred times the work. With 3.1 million trades, I had to train the RBF SVM on a 100,000-trade sample on a GPU. Even then, after SMOTE, it kept about 100,000 support vectors, which made scoring new trades slow.

**How XGBoost works.** XGBoost builds hundreds of small decision trees, one after another. Each tree is a short list of yes/no questions: "Do the instructions match? Is it cross-border?" Every new tree focuses on the mistakes the earlier trees still make, like a team of reviewers where each one fixes what the last one missed. It handles categories and missing values on its own and scales to millions of rows.

**The baseline.** I also trained a logistic regression, the simplest model of the bunch. It's fast and easy to read. For example, it says a mismatched instruction multiplies the odds of failing by about 8.7. It was already decent (PR-AUC 0.305), because most of my hidden recipe is a simple sum of effects. What it can't see on its own are combinations, like wrong account details **and** cross-border.

![Precision-recall curves on the test period](https://huggingface.co/rohanjain2312/trade-settlement-fail-predictor/resolve/main/report/pr_curves.png)
*The higher the curve, the better the ranking. XGBoost (red) stays on top. The dotted line is a random list. Synthetic data.*

Results on the test period:

- **XGBoost: PR-AUC 0.333**, about 7 times better than a random list.
- In the riskiest **2%** of trades, XGBoost caught **23%** of all fails. In the riskiest 1%, it caught 14%. A random pick would catch about 2% and 1%.
- Best SVM (RBF with class weights): 0.319. Logistic regression: 0.305.

All models were tuned with Optuna, a tool that searches for good settings automatically. It scored each setting with cross-validation that respects time order, so tuning never peeked at the future either.

**What I'd do differently:** start with XGBoost for table-shaped data like this, and keep logistic regression as the baseline it has to beat.

## Turning scores into probabilities you can trust

An SVM doesn't give a probability. It gives a distance from the street. "2.1 units from the line" means nothing to an ops manager. "A 79% chance of failing" does.

**Calibration** fixes this. I fitted an S-shaped curve (called Platt scaling) that maps SVM scores to probabilities. For example, a score of 0 maps to about 3%, a score of 1 to about 25%, and a score of 2 to about 79%.

Class weights distort probabilities too. Telling the model "fails matter more" makes it overestimate risk. XGBoost with class weights had a Brier score of 0.213 before calibration and **0.035 after**.

Two rules matter most:

- **Fit the calibration on separate validation data**, never on the test set and never on the training set. Otherwise you are grading your own homework.
- **Recheck it over time.** For the riskiest trades in the test period, the SVM's curve predicted about 37%, but only about 27% actually failed. A curve fitted on one period drifts on the next, so check it on new months and refit when needed.

**What I'd do differently:** always calibrate, and keep checking it.

## Explaining every alert with SHAP

A risk score alone isn't enough. When ops see a trade flagged at 60%, the first question is "why?" The answer decides what they do next.

**SHAP** answers that for each trade. It splits the model's score into pieces, one per feature, that add up to the final score. Think of it as a receipt: base risk, plus this much for the mismatched account details, plus this much for being cross-border, minus this much because the shares are already in place.

Because I wrote the hidden recipe, I could check whether SHAP's explanations match the truth:

- SHAP ranked the features in almost the same order as the recipe, with a **rank correlation of 0.96** (1.0 would be a perfect match).
- Out of 190 possible feature pairs, SHAP ranked the **three planted combinations #1, #2, and #3**.

![SHAP importance compared with the planted truth](https://huggingface.co/rohanjain2312/trade-settlement-fail-predictor/resolve/main/report/shap_vs_truth.png)
*Blue: what the model learned (SHAP). Orange: the true recipe. Synthetic data.*

The differences are interesting too:

- **Counterparty fail rate** ranked higher than planted. It also stands in for hidden counterparty risk that isn't in the data, so the model leans on it more.
- **Market turbulence** ranked lower. The test period has a bigger market spike than anything in training, and tree models can't predict beyond the range they've seen.

In practice, the top reasons turn straight into actions: fix the account details, chase the confirmation, find the missing shares.

**What I'd do differently:** ship every alert with its reasons. An SVM gives you a score but no easy "why". XGBoost with SHAP gives you both.

## Set the number of daily alerts, not a fixed cutoff

I chose a cutoff that flagged 2% of trades in the validation period. In the test period, which includes a stressful market spike, the same cutoff flagged **6.8%** of trades. That's more than three times what ops planned for.

A fixed probability cutoff breaks as soon as the market changes. A better rule: each day, flag the **top k trades**, where k is the number the team can actually check.

I also built a small simulator to show the effect. Under assumptions I chose (ops check 2% of trades each day and fix 60% of the would-be fails they reach), ranking trades with XGBoost cut pending fails by about **12%** compared with checking trades in no particular order. This is a what-if on synthetic data, not a measured result, and the demo lets you change every assumption.

**What I'd do differently:** let team capacity set the number of daily alerts, and watch for drift.

## What the model can't do

- **It only partly recognizes new kinds of fails.** When I removed every wrong-account-details case from training, the model caught 27% of them in testing instead of 45%. For short-on-shares cases, it dropped from 35% to 14%. When a new kind of problem shows up, label it and retrain.
- **The data is synthetic.** It shows the method, not real-world performance. Real data always has cases nobody planned for.
- **Some things are out of scope**, like partial settlements and market-wide outages.

## Summary: what I'd do now

| Decision | What I'd do now | Why |
|---|---|---|
| Rare fails | Class weights, then choose the cutoff | SMOTE lost to class weights in every model |
| Model | XGBoost, with logistic regression as the baseline | Best ranking, finds combinations, scales to millions of rows |
| Scores | Calibrate on validation data | Ops need probabilities they can trust |
| Explanations | SHAP reasons for every alert | Tells ops what to fix first |
| Alerts | Daily top k from team capacity | Fixed cutoffs break when markets change |
| Testing | Synthetic data with a known answer key, plus leakage tests | Shows whether the model finds real signals |

## Try it yourself

Everything is open:

- **Live demo:** [Hugging Face Space](https://huggingface.co/spaces/rohanjain2312/trade-settlement-fail-predictor-demo). Move the SVM sliders, watch SMOTE create fake trades, explore SHAP for any trade, and run the ops simulator with your own assumptions.
- **Code:** [GitHub](https://github.com/Rohanjain2312/trade-settlement-fail-predictor)
- **Full results:** [RESULTS.md](https://huggingface.co/rohanjain2312/trade-settlement-fail-predictor/blob/main/report/RESULTS.md)
- **Notebooks:** [generate the data](https://colab.research.google.com/github/Rohanjain2312/trade-settlement-fail-predictor/blob/main/notebooks/01_generate_data.ipynb) and [train and explain the models](https://colab.research.google.com/github/Rohanjain2312/trade-settlement-fail-predictor/blob/main/notebooks/02_train_explain.ipynb) in Colab

Every number in this article can be reproduced from the repo's config files and a fixed random seed.

If you have used SMOTE on a real imbalanced problem, I'd like to hear whether it helped you. Leave a comment.

---

*All data in this article is synthetic and generated by code. The models are a demonstration and must not be used for real settlement decisions.*
