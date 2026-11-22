---
title: "How Tufa Labs Shrank 1 GB of Wikipedia to 105 MB Using Only a Transformer"
lang: en
excerpt: "Tufa Labs compressed 1 GB of Wikipedia into 105.39 MB using a 163M-parameter Transformer—basically Pied Piper from Silicon Valley, except it takes a week."
takeaways:
  - "Tufa Labs' tufazip compressed 1 GB of English Wikipedia (enwik9) down to 105.39 MB losslessly—roughly a 9.5x compression ratio—using a Transformer as its sole predictor."
  - "To avoid shipping hundreds of megabytes of model weights, tufazip starts from a 4.33 MB quantized prior and trains itself identically during both compression and decompression."
  - "Unlike Pied Piper's instant magic in Silicon Valley, tufazip needs two NVIDIA B200 GPUs and around 7.5 days just to decompress the file."
header:
  teaser: https://pbs.twimg.com/profile_banners/1852704072937160704/1791982536
comments: true
share: true
categories:
  - News
tags:
  - ai
  - compression
  - transformer
  - news
  - machine learning
---

Hello everyone! If you ever watched HBO's *Silicon Valley*, you probably remember **Pied Piper**—Richard Hendricks' legendary startup with a "middle-out" compression algorithm so absurdly good that tech billionaires went to war over it. Whenever a researcher announces a wild new compression ratio, the internet immediately asks the same question: *"Wait… is Pied Piper actually real?"*

Well, this week an independent AI research lab called **Tufa Labs** dropped a project called **[tufazip](https://tufalabs.ai/research/tufazip/)**, and honestly, it is the closest thing to a real-life Pied Piper flex we have seen in a while—with one hilarious catch.

## 1. Wait, What Did Tufa Labs Actually Do?

In short, Tufa Labs took **1 billion bytes (~1 GB) of English Wikipedia** (`enwik9`, the classic benchmark file used for Matt Mahoney's Large Text Compression Benchmark and the Hutter Prize) and squeezed it down to **105,388,353 bytes (105.39 MB)**—completely **lossless**. Not a single byte or XML tag goes missing.

What makes `tufazip` stand out isn't just the ~9.5x compression ratio. Most top entries on that leaderboard rely on heavy, hand-crafted text preprocessing rules or hybrid statistical models stitched together over decades. Tufa Labs threw all those manual rules out the window and used a **163-million-parameter decoder-only Transformer as the sole predictor**.

At the time of submission, `tufazip` grabbed the **#2 spot overall** on the 20-year-old Large Text Compression Benchmark and became the **#1 pure neural-network entry**, beating Fabrice Bellard's `NNCP v3.2` (107.26 MB).

If Erlich Bachman were pitching this at TechCrunch Disrupt, he would already be ordering a custom neon sign for the incubator.

## 2. How Do You Compress Text With a Transformer?

At first glance, using an LLM-style Transformer to make a file *smaller* sounds backwards. Aren't neural networks huge? How do you shrink 1 GB of text using a 163M-parameter model without the model itself taking up 600 MB?

Here is the core idea: **compression is just next-token prediction**.

> In information theory, if your model assigns probability $p$ to the next token, an arithmetic coder can encode that token in roughly $-\log_2(p)$ bits. The better your model predicts what comes next, the fewer bits you spend.

**Extra Example:** Imagine texting a close friend who knows you so well they can finish every sentence before you type it. You barely need to send any words because everything you say is predictable. A Transformer does the exact same thing with Wikipedia:

1. **Routine patterns are practically free:** Common English grammar, dates, and Wikipedia XML templates get high probability scores, so arithmetic coding packs them into tiny fractions of a bit.
2. **Plot twists are expensive:** Rare names or unexpected facts get lower probabilities, so they cost more bits to store.

### The Clever Weight-Saving Trick (Synchronized Online Learning)

Under benchmark rules, the final file size has to include **everything** needed to reconstruct the original data—meaning the compressed payload *plus* the decoder program and any saved model weights.

If Tufa Labs saved a trained 163M-parameter model inside the `.zip` archive, their score would be ruined. Instead, they pulled a very neat engineering trick:

- **Tiny Offline Starter (`4.33 MB`):** They trained a smaller 34M-parameter offline model using a "Cycle2" layout (two shared Transformer blocks repeated across 12 steps), quantized it down to just **4.33 MB**, and bundled that with a 4K-word vocabulary.
- **Learning on the Fly:** At runtime, those 2 blocks expand into 12 independent layers (163M parameters total). As `tufazip` reads Wikipedia, it continuously trains itself via **online learning** with replay.
- **Symmetrical Decompression:** Because the decoder starts from the exact same 4.33 MB checkpoint and runs the **exact same deterministic training loop**, it relearns the 163M parameters step-by-step while unpacking the file. The updated weights never have to be transmitted.

Even better, Tufa Labs found a couple of counterintuitive quirks during their experiments: a smaller **4K-word vocabulary** actually beat 8K and 16K vocabularies, and giving Wikipedia's XML markup its own special tokens made compression *worse*. Classic machine learning—sometimes the simpler setup wins.

## 3. The Catch: Pied Piper, If Pied Piper Needed a Data Center and a Vacation

Now, before you uninstall `7-Zip`, `zstd`, or stop saving your datasets in `.parquet`, let's look at the hardware and runtime requirements from the [Tufa Labs GitHub repo](https://github.com/Tufalabs/tufazip):

```bash
# Requirements: Linux x86-64, 2x NVIDIA B200 GPUs, CUDA 13.0, and a lot of patience
uv run --frozen --python 3.12.3 ./tufazip /path/to/enwik9-v2.bin /path/to/enwik9 /path/to/state
```

Here is what the numbers actually look like:

| Metric | Result |
| --- | --- |
| **Original Input (`enwik9`)** | 1,000,000,000 bytes (~1 GB) |
| **Compressed Payload** | 100,958,324 bytes (~100.96 MB) |
| **Decoder + Quantized Weights** | 4,430,029 bytes (~4.43 MB) |
| **Total Ranked Size** | **105,388,353 bytes (~105.39 MB)** |
| **Hardware Required** | **2× NVIDIA B200 GPUs** |
| **Compression Time** | ~631,548 seconds (**~7.3 days**) |
| **Decompression Time** | ~645,748 seconds (**~7.5 days**) |

Not gonna lie, reading those runtimes feels straight out of a Gilfoyle vs. Dinesh argument.

In *Silicon Valley*, Pied Piper's whole magic trick was streaming ultra-compressed 4K video effortlessly on a smartphone. With `tufazip`, decompressing 1 GB takes **seven and a half days** on two enterprise-grade NVIDIA B200 GPUs—which works out to roughly **1.5 KB per second**. That is slower than dial-up internet in the 1990s.

And because arithmetic coding is unforgiving, the entire process has to be **100% deterministic**. If a single floating-point calculation drifts by one tiny bit between encoding and decoding, every prediction after that goes out of sync and the rest of Wikipedia turns into total gibberish. Imagine waiting six days for your `.bin` file to unpack, only for a cosmic ray or a CUDA version mismatch to corrupt the output on day seven.

## In Summary

Tufa Labs isn't trying to sell you a consumer file archiver—and honestly, thank goodness, because nobody wants to wait until next Sunday just to open a ZIP file. What `tufazip` actually proves is how tightly **language modeling and lossless compression are linked**: a pure Transformer, with zero hand-coded linguistic rules, can learn the structure of human knowledge well enough to rival decades of specialized compression engineering.

For everyday data work, stick to `zstd` or Apache Parquet. For AI research, tho, watching a neural network brute-force its way into Pied Piper territory is pretty awesome.
