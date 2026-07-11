"""
Evaluate compression ratio of the tokenizer.

Default (no args): compare our tokenizer against GPT-2 and GPT-4.
With --tokenizer-dirs A B C: rank candidate tokenizer directories by compression
(used as the cheap pre-screen for the BPB sweep; see scripts/eval/tok_sweep.py).
"""

import os
import argparse

from mesosfer.data.tokenizer import get_tokenizer, RustBPETokenizer
from mesosfer.data.dataset import parquets_iter_batched

# Random text I got from a random website this morning
news_text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico’s National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.

While USDA announced a risk-based phased port re-opening strategy for cattle, bison, and equine from Mexico beginning as early as July 7, 2025, this newly reported NWS case raises significant concern about the previously reported information shared by Mexican officials and severely compromises the outlined port reopening schedule of five ports from July 7-September 15. Therefore, in order to protect American livestock and our nation’s food supply, Secretary Rollins has ordered the closure of livestock trade through southern ports of entry effective immediately.

“The United States has promised to be vigilant — and after detecting this new NWS case, we are pausing the planned port reopening’s to further quarantine and target this deadly pest in Mexico. We must see additional progress combatting NWS in Veracruz and other nearby Mexican states in order to reopen livestock ports along the Southern border,” said U.S. Secretary of Agriculture Brooke L. Rollins. “Thanks to the aggressive monitoring by USDA staff in the U.S. and in Mexico, we have been able to take quick and decisive action to respond to the spread of this deadly pest.”
""".strip()

# Random Indonesian text (to test non-English compression)
indonesian_text = r"""
Berita Terkini dari Seluruh Nusantara
Kompas Digital

Kompas Digital hadir sebagai platform berita daring terpercaya yang menyajikan informasi aktual seputar politik, ekonomi, sosial, budaya, dan teknologi di Indonesia maupun mancanegara.

Kami berkomitmen untuk menyampaikan berita secara berimbang dan bertanggung jawab, berlandaskan prinsip jurnalisme yang menjunjung tinggi kebenaran dan kepentingan publik. Setiap laporan disusun berdasarkan fakta yang telah diverifikasi dari berbagai sumber terpercaya.

Dalam era informasi yang bergerak cepat, kami memahami pentingnya kecepatan tanpa mengorbankan akurasi. Oleh karena itu, tim redaksi kami bekerja keras memastikan setiap artikel yang diterbitkan telah melalui proses pengecekan fakta yang ketat.

Kami percaya bahwa masyarakat berhak mendapatkan informasi yang jujur, transparan, dan bebas dari kepentingan golongan tertentu. Dengan semangat kebhinekaan dan persatuan, kami terus berupaya menjadi jembatan informasi bagi seluruh lapisan masyarakat Indonesia.
""".strip()

# Random piece of code
code_text = r"""
class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
""".strip()

math_text = r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amsthm,amssymb}
\usepackage[margin=1in]{geometry}

\newtheorem{theorem}{Theorem}
\newtheorem*{remark}{Remark}

\begin{document}

\begin{center}
{\Large A Cute Identity: The Sum of Cubes is a Square}
\end{center}

\begin{theorem}
For every integer $n \ge 1$,
\[
\sum_{k=1}^{n} k^{3} \;=\; \left(\frac{n(n+1)}{2}\right)^{2}.
\]
\end{theorem}

\begin{proof}[Proof 1 (Induction)]
Let $S(n) = \sum_{k=1}^{n} k^3$. For $n=1$, $S(1)=1=(1\cdot 2/2)^2$, so the base case holds.

Assume $S(n)=\big(\tfrac{n(n+1)}{2}\big)^2$ for some $n\ge 1$.
Then
\[
S(n+1)
= S(n) + (n+1)^3
= \left(\frac{n(n+1)}{2}\right)^2 + (n+1)^3.
\]
Factor out $(n+1)^2$:
\[
S(n+1)
= (n+1)^2\left( \frac{n^2}{4} + (n+1) \right)
= (n+1)^2\left( \frac{n^2 + 4n + 4}{4} \right)
= (n+1)^2\left( \frac{(n+2)^2}{4} \right).
\]
Thus
\[
S(n+1)=\left(\frac{(n+1)(n+2)}{2}\right)^2,
\]
which matches the claimed formula with $n$ replaced by $n+1$. By induction, the identity holds for all $n\ge 1$.
\end{proof}

\begin{proof}[Proof 2 (Algebraic telescoping)]
Recall the binomial identity
\[
(k+1)^4 - k^4 = 4k^3 + 6k^2 + 4k + 1.
\]
Summing both sides from $k=0$ to $n$ telescopes:
\[
(n+1)^4 - 0^4
= \sum_{k=0}^{n}\big(4k^3 + 6k^2 + 4k + 1\big)
= 4\sum_{k=1}^{n}k^3 + 6\sum_{k=1}^{n}k^2 + 4\sum_{k=1}^{n}k + (n+1).
\]
Using the standard sums
\[
\sum_{k=1}^{n}k = \frac{n(n+1)}{2}
\quad\text{and}\quad
\sum_{k=1}^{n}k^2 = \frac{n(n+1)(2n+1)}{6},
\]
solve for $\sum_{k=1}^{n}k^3$ to get
\[
\sum_{k=1}^{n}k^3 = \left(\frac{n(n+1)}{2}\right)^2.
\]
\end{proof}

\begin{remark}
Geometrically, the identity says: ``adding up $1^3,2^3,\dots,n^3$ builds a perfect square’’—namely the square of the $n$th triangular number. This is why one sometimes calls it the \emph{sum-of-cubes is a square} phenomenon.
\end{remark}

\end{document}
""".strip()

science_text = r"""
Photosynthesis is a photochemical energy transduction process in which light-harvesting pigment–protein complexes within the thylakoid membranes of oxygenic phototrophs absorb photons and initiate charge separation at the reaction center, driving the linear electron transport chain from water to NADP⁺ via photosystem II, the cytochrome b₆f complex, and photosystem I, concomitantly generating a trans-thylakoid proton motive force utilized by chloroplastic ATP synthase. The light-dependent reactions produce ATP and NADPH, which fuel the Calvin–Benson–Bassham cycle in the stroma, wherein ribulose-1,5-bisphosphate is carboxylated by ribulose-1,5-bisphosphate carboxylase/oxygenase (RuBisCO) to form 3-phosphoglycerate, subsequently reduced and regenerated through a series of enzymatic steps, enabling net assimilation of CO₂ into triose phosphates and ultimately carbohydrates. This process is tightly regulated by photoprotective mechanisms, redox feedback, and metabolite flux, representing a central biochemical pathway coupling solar energy capture to the biosphere’s primary productivity.
""".strip()

# Cybersecurity narrative (incident response, MITRE ATT&CK, CVE references)
cybersec_text = r"""
At 14:32 UTC, the SOC observed CVE-2024-3094-style supply chain backdoor activity originating from the build pipeline for service api-prod-east. The initial indicator was anomalous ELF section padding in libcompress.so.5 detected by YARA rule MAL_SupplyChain_Liblzma_3094. Within minutes, the EDR (CrowdStrike Falcon) flagged a suspicious sshd child process spawning /usr/bin/python3 with base64-encoded arguments, consistent with MITRE ATT&CK T1059.006 (Python) and T1027.013 (encrypted/encoded files). Containment actions: isolated host i-0a3f9b7c via VPC NACL, snapshotted volume vol-0e8a1b2c3d4e5f6 for forensics, revoked IAM role arn:aws:iam::482719364105:role/api-prod-east-svc, and rotated all KMS keys touched in the last 24h. The threat hunting team correlated this with prior CISA KEV entries and confirmed lateral movement via SSM Session Manager (T1021.004), suggesting a hands-on-keyboard intrusion rather than fully automated malware. Recommended SigmaHQ rule: detect ssm:StartSession from non-jumphost source IPs in CloudTrail. Final assessment: high-confidence intrusion, defense evasion phase, no exfiltration confirmed but credential access likely.
""".strip()

# Suricata-style alert in narrative form (typical SOC log analysis)
soc_log_text = r"""
Between 03:14:22 and 03:14:48 UTC, internal host 10.10.24.57 generated 4 Suricata alerts (signature_id 2030011, "ET MALWARE Possible C2 Beaconing Activity", severity 1) targeting external IP 185.193.126.44 on TCP/443. The flow analysis showed periodic small-byte transfers (1244 bytes outbound, 1987 bytes inbound) with sub-second TLS handshakes, consistent with command-and-control beaconing rather than legitimate traffic. JA3 fingerprint 72a589da586844d7f0818ce684948eea matched known Cobalt Strike profiles. Concurrent DNS TXT queries for ajd82j3k2k2k.cmd-sync-storage.com (algorithmically generated subdomain pattern) returned base64-encoded answers — strong indicator of DNS tunneling exfiltration (MITRE ATT&CK T1071.004). Action taken: blackholed destination IP at perimeter firewall, queued endpoint for memory acquisition.
""".strip()

def build_eval_texts():
    """Assemble the (name, text) domain probes plus, if available, one batch of
    train/val parquet text. Guarded so missing shards don't break inline evaluation."""
    # The tokenizer was trained on data from earlier shards, so it has seen train data.
    try:
        train_docs = next(parquets_iter_batched(split="train"))
        train_text = "\n".join(train_docs)
    except (StopIteration, FileNotFoundError, OSError) as e:
        print(f"Warning: could not load train parquet shards ({type(e).__name__}: {e}); skipping fwe-train.")
        train_text = ""
    try:
        val_docs = next(parquets_iter_batched(split="val"))
        val_text = "\n".join(val_docs)
    except (StopIteration, FileNotFoundError, OSError) as e:
        print(f"Warning: could not load val parquet shards ({type(e).__name__}: {e}); skipping fwe-val.")
        val_text = ""

    all_text = [
        ("news", news_text),
        ("indonesian", indonesian_text),
        ("code", code_text),
        ("math", math_text),
        ("science", science_text),
        ("cybersec", cybersec_text),
        ("soc-log", soc_log_text),
    ]
    if train_text:
        all_text.append(("fwe-train", train_text))
    if val_text:
        all_text.append(("fwe-val", val_text))
    return all_text


def measure_compression(tokenizer, all_text):
    """Return {text_name: {bytes, tokens, ratio}} for one tokenizer. Asserts lossless round-trip."""
    results = {}
    for name, text in all_text:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text
        encoded_bytes = text.encode('utf-8')
        results[name] = {
            'bytes': len(encoded_bytes),
            'tokens': len(encoded),
            'ratio': len(encoded_bytes) / len(encoded),
        }
    return results


def overall_ratio(results):
    """Byte-weighted overall bytes/token (higher = better compression)."""
    total_bytes = sum(r['bytes'] for r in results.values())
    total_tokens = sum(r['tokens'] for r in results.values())
    return total_bytes / total_tokens


# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def print_comparison(baseline_name, baseline_results, ours_results, all_text):
    """Print comparison table between baseline tokenizer and ours."""
    print(f"\nComparison with {baseline_name}:")
    print("=" * 95)
    print(f"{'Text Type':<10} {'Bytes':<8} {baseline_name:<15} {'Ours':<15} {'Relative':<12} {'Better':<10}")
    print(f"{'':10} {'':8} {'Tokens':<7} {'Ratio':<7} {'Tokens':<7} {'Ratio':<7} {'Diff %':<12}")
    print("-" * 95)

    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]

        # Calculate relative difference (positive means ours is better, negative means worse)
        # Using tokens: fewer tokens is better, so we calculate (baseline_tokens - ours_tokens) / baseline_tokens
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100

        # Determine which has better compression (higher ratio = better)
        if baseline_data['ratio'] > ours_data['ratio']:
            baseline_color, ours_color = GREEN, RED
            better = baseline_name
            diff_color = RED
        elif ours_data['ratio'] > baseline_data['ratio']:
            baseline_color, ours_color = RED, GREEN
            better = "Ours"
            diff_color = GREEN
        else:
            baseline_color, ours_color = "", ""
            better = "Tie"
            diff_color = ""

        print(f"{name:<10} {baseline_data['bytes']:<8} "
              f"{baseline_color}{baseline_data['tokens']:<7}{RESET} "
              f"{baseline_color}{baseline_data['ratio']:<7.2f}{RESET} "
              f"{ours_color}{ours_data['tokens']:<7}{RESET} "
              f"{ours_color}{ours_data['ratio']:<7.2f}{RESET} "
              f"{diff_color}{relative_diff:+7.1f}%{RESET}     "
              f"{better:<10}")

def _run_default():
    """Original behavior: compare our tokenizer against GPT-2 and GPT-4."""
    all_text = build_eval_texts()
    tokenizer_results = {}
    vocab_sizes = {}
    for tokenizer_name in ["gpt2", "gpt4", "ours"]:
        if tokenizer_name == "gpt2":
            tokenizer = RustBPETokenizer.from_pretrained("gpt2")       # gpt-2 base model tokenizer
        elif tokenizer_name == "gpt4":
            tokenizer = RustBPETokenizer.from_pretrained("cl100k_base")  # gpt-4 base model tokenizer
        else:
            tokenizer = get_tokenizer()
        vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
        tokenizer_results[tokenizer_name] = measure_compression(tokenizer, all_text)

    print(f"\nVocab sizes:")
    print(f"GPT-2: {vocab_sizes['gpt2']}")
    print(f"GPT-4: {vocab_sizes['gpt4']}")
    print(f"Ours: {vocab_sizes['ours']}")

    print_comparison("GPT-2", tokenizer_results['gpt2'], tokenizer_results['ours'], all_text)
    print_comparison("GPT-4", tokenizer_results['gpt4'], tokenizer_results['ours'], all_text)

    # Log to report
    from mesosfer.utils.report import get_report
    lines = []
    for baseline_name in ["GPT-2", "GPT-4"]:
        baseline_key = baseline_name.lower().replace('-', '')
        baseline_results = tokenizer_results[baseline_key]
        ours_results = tokenizer_results['ours']
        lines.append(f"### Comparison with {baseline_name}")
        lines.append("")
        lines.append("| Text Type | Bytes | " + baseline_name + " Tokens | " + baseline_name + " Ratio | Ours Tokens | Ours Ratio | Relative Diff % |")
        lines.append("|-----------|-------|--------------|--------------|-------------|------------|-----------------|")
        for name, text in all_text:
            baseline_data = baseline_results[name]
            ours_data = ours_results[name]
            relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100
            lines.append(f"| {name} | {baseline_data['bytes']} | {baseline_data['tokens']} | {baseline_data['ratio']:.2f} | {ours_data['tokens']} | {ours_data['ratio']:.2f} | {relative_diff:+.1f}% |")
        lines.append("")
    report_markdown = "\n".join(lines)
    get_report().log(section="Tokenizer evaluation", data=[report_markdown])


def _run_candidates(tokenizer_dirs):
    """Rank candidate tokenizer directories by compression. Returns rows sorted best-first.
    Used as the cheap pre-screen for the BPB sweep."""
    all_text = build_eval_texts()
    rows = []
    for d in tokenizer_dirs:
        tokenizer = RustBPETokenizer.from_directory(d)
        results = measure_compression(tokenizer, all_text)
        rows.append({
            'name': os.path.basename(os.path.normpath(d)),
            'dir': d,
            'vocab': tokenizer.get_vocab_size(),
            'overall': overall_ratio(results),
            'results': results,
        })
    rows.sort(key=lambda r: r['overall'], reverse=True)  # best compression first

    print("\nCandidate tokenizer compression (byte-weighted bytes/token; higher = better):")
    print("=" * 72)
    print(f"{'Rank':<5}{'Candidate':<28}{'Vocab':<10}{'Overall':<12}")
    print("-" * 72)
    for i, r in enumerate(rows, 1):
        color = GREEN if i == 1 else ""
        print(f"{i:<5}{color}{r['name']:<28}{RESET}{r['vocab']:<10}{r['overall']:<12.4f}")

    domain_names = [n for n, _ in all_text]
    print("\nPer-domain ratio:")
    header = f"{'Candidate':<28}" + "".join(f"{n[:9]:<10}" for n in domain_names)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['name']:<28}" + "".join(f"{r['results'][n]['ratio']:<10.2f}" for n in domain_names))

    from mesosfer.utils.report import get_report
    lines = ["### Candidate tokenizer compression pre-screen", "",
             "| Rank | Candidate | Vocab | Overall bytes/token |",
             "|------|-----------|-------|---------------------|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['name']} | {r['vocab']} | {r['overall']:.4f} |")
    get_report().log(section="Tokenizer evaluation", data=["\n".join(lines)])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate tokenizer compression ratio")
    parser.add_argument("--tokenizer-dirs", nargs="+", default=None,
                        help="Rank these candidate tokenizer directories by compression (BPB sweep pre-screen). "
                             "If omitted, compare our default tokenizer against GPT-2 and GPT-4.")
    args = parser.parse_args()
    if args.tokenizer_dirs:
        _run_candidates(args.tokenizer_dirs)
    else:
        _run_default()


if __name__ == "__main__":
    main()
