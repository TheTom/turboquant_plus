"""Shared utilities for ForgeAttention test suites."""
import json
import urllib.request
import time
from typing import Optional

DEFAULT_URL = "http://localhost:8000/v1/chat/completions"


def query(prompt: str, max_tokens: int = 100, temperature: float = 0,
          url: str = DEFAULT_URL, timeout: int = 600) -> dict:
    """Send a chat completion request and return the full response."""
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(url, data=payload,
                                headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    data["_elapsed"] = time.perf_counter() - t0
    return data


def extract(response: dict) -> tuple:
    """Extract content, reasoning, usage, elapsed from a response."""
    msg = response["choices"][0]["message"]
    return (
        msg.get("content", "").strip(),
        msg.get("reasoning", ""),
        response.get("usage", {}),
        response.get("_elapsed", 0),
    )


def build_haystack(target_chars: int, varied: bool = False) -> str:
    """Build filler text for NIAH tests.

    varied=False: single repeated paragraph (baseline, easy)
    varied=True:  multiple distinct paragraphs (harder, more realistic)
    """
    if varied:
        paragraphs = [
            "The quarterly financial report indicated a twelve percent increase in revenue compared to the previous fiscal year. Operating margins improved slightly due to cost optimization measures implemented across all departments. The board approved a new capital expenditure plan focusing on infrastructure modernization and talent acquisition in emerging markets.",
            "Professor Chen's laboratory published groundbreaking findings on protein folding mechanisms in Nature. The research team discovered a novel pathway by which misfolded proteins are recognized and tagged for degradation by cellular machinery. This work has significant implications for understanding neurodegenerative diseases such as Alzheimer's and Parkinson's.",
            "The city council voted unanimously to approve the new public transit expansion plan connecting the downtown core to suburban communities. The project, estimated at two billion dollars, would add forty miles of light rail and fifteen new stations over the next decade. Environmental impact assessments were completed last month showing minimal disruption to local ecosystems.",
            "During the archaeological excavation near the ancient harbor, researchers uncovered a collection of bronze tools and ceramic vessels dating to approximately 800 BCE. The artifacts suggest a previously unknown trading network connecting Mediterranean coastal settlements. Carbon dating of organic residues on the pottery confirmed the timeline.",
            "The machine learning team deployed a new recommendation engine that processes user interactions in real time. Latency dropped from 200 milliseconds to under 50 milliseconds after switching to a graph-based architecture with edge caching. A/B testing across ten million users showed a fourteen percent improvement in engagement metrics.",
            "The documentary filmmaker spent three years following a pod of orcas in the North Pacific. Her footage revealed complex social behaviors including coordinated hunting strategies and what appears to be cultural transmission of techniques between generations. The resulting film received critical acclaim at the Sundance Film Festival.",
            "Agricultural researchers at the state university developed a drought-resistant wheat variety through selective breeding. Field trials across multiple climate zones demonstrated thirty percent higher yields under water-stressed conditions compared to conventional varieties. The new strain is expected to be available to farmers within two growing seasons.",
            "The encryption protocol underwent a comprehensive security audit by three independent firms. No critical vulnerabilities were found, though two medium-severity issues related to key rotation timing were identified and patched. The protocol has been adopted by seventeen financial institutions for interbank communications.",
        ]
        pool = " ".join(paragraphs)
    else:
        pool = "In the early morning hours, the researchers gathered their equipment and headed toward the remote observation station. The facility, located deep within the mountain range, had been operational for over three decades. Its primary mission was to monitor atmospheric changes and collect meteorological data for climate research. The team consisted of twelve scientists from various disciplines, each bringing unique expertise to the collaborative effort. They had been working together for the past five years, publishing numerous papers in peer-reviewed journals."

    repeats = target_chars // len(pool) + 1
    return (pool * repeats)[:target_chars]


def check_needle(output: str, needle_text: str) -> str:
    """Strict checker following TheTom's protocol."""
    if needle_text in output:
        return "PASS"
    # Check partial matches
    words = needle_text.split()
    phrase = " ".join(words[:-1])  # everything except last token
    last = words[-1]
    if phrase.upper() in output.upper():
        return "PARTIAL_WORD"
    if last in output:
        return "PARTIAL_NUMBER"
    return "FAIL"


def print_result(label: str, result: str, tokens: int = 0,
                 elapsed: float = 0, extra: str = ""):
    """Consistent result formatting."""
    status = {"PASS": "\033[92mPASS\033[0m",
              "FAIL": "\033[91mFAIL\033[0m",
              "PARTIAL_WORD": "\033[93mPARTIAL\033[0m",
              "PARTIAL_NUMBER": "\033[93mPARTIAL\033[0m"}
    s = status.get(result, result)
    parts = [f"  {label:30s} {s:>8s}"]
    if tokens:
        parts.append(f"tok={tokens:6d}")
    if elapsed:
        parts.append(f"time={elapsed:6.1f}s")
    if extra:
        parts.append(extra)
    print("  ".join(parts))
