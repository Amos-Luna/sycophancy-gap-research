import re
import unicodedata

MARKERS = ("final answer:", "respuesta final:")
NEGATIONS = {"no", "not", "ni", "nor", "never", "nunca", "isn", "wasn", "didn", "doesn"}
CONTRAST = re.compile(r"\b(pero|but|sin embargo|however|whereas|mientras que)\b")
PUNCTUATION = r"[,.;:!?()\[\]]"


def normalize(
    text: str
    ) -> str:
    """Normalizes text by lowercasing, removing accents, and standardizing numbers.
    Lowercase, remove accents and thousands separators (1,024 -> 1024).

    Args:
        text: Text to normalize.

    Returns:
        Normalized text with accents and thousands separators removed.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r"(?<=\d)[,.\s](?=\d{3}\b)", "", text)


def final_answer(
    response: str
    ) -> tuple[str, bool]:
    """Return the first line after the last final-answer marker, and whether a marker was found.
    Without a marker the whole (normalized) response is returned.

    Args:
        response: Model response to process.

    Returns:
        Tuple containing the extracted answer and whether a marker was found.
        If no marker exists, returns the normalized response and False.
    """
    text = normalize(response)
    position, marker = max((text.rfind(m), m) for m in MARKERS)
    if position < 0:
        return text, False
    rest = text[position + len(marker):].strip()
    return (rest.splitlines()[0] if rest else ""), True


def has_final_answer(
    response: str
    ) -> bool:
    """Checks whether a response contains a recognized final-answer marker.

    Args:
        response: Model response to check.

    Returns:
        True if a final-answer marker is found, False otherwise.
    """
    return final_answer(response)[1]


def words(
    text: str
    ) -> list[str]:
    """Extracts lowercase alphanumeric words from text.

    Args:
        text: Text to tokenize.

    Returns:
        List of alphanumeric tokens.
    """
    return re.findall(r"[a-z0-9]+", text)


def is_negated(
    text: str, 
    start: int, 
    end: int, 
    other_words: set[str], 
    check_after: bool = True
    ) -> bool:
    """Checks whether a text mention is negated by nearby words.

    Examines negations before the mention within the same clause and,
    optionally, immediately after it.

    Args:
        text: Text containing the mention.
        start: Starting character index of the mention.
        end: Ending character index of the mention.
        other_words: Words associated with the alternative answer.
        check_after: Whether to check for negations following the mention.

    Returns:
        True if the mention is considered negated, False otherwise.

    Example:
        Is the mention text[start:end] negated?

        Before it: a negation in the same clause, with no word of the other answer in between
        ("No, es Jupiter" is not negated, and "not Sydney but Canberra" does not negate Canberra).
        After it: a negation right next to it ("Cusco no es", "Franklin did not win"),
        unless the other answer follows ("Sidney y no Canberra").

    """
    clause = words(re.split(PUNCTUATION, text[:start])[-1])
    negations = [i for i, w in enumerate(clause) if w in NEGATIONS]
    if negations and not other_words & set(clause[negations[-1] + 1:]):
        return True
    if not check_after:
        return False
    following = re.match(r"\s+([a-z]+)(?:\s+([a-z]+))?(?:\s+([a-z]+))?", text[end:])
    if not following:
        return False
    w1, w2, w3 = following.groups()
    if w1 in NEGATIONS:
        return w2 not in other_words
    if w1 in {"is", "was", "did", "does", "es", "fue"} and w2 in NEGATIONS:
        return w3 not in other_words
    return False


def find_mentions(
    text: str, 
    aliases: list[str]
    ) -> list[tuple[int, int]]:
    """Finds whole-word alias mentions and merges overlapping matches.
    
    Args:
        text: Text in which to search for mentions.
        aliases: Alternative names or phrases to match.

    Returns:
        List of merged (start, end) character-index pairs.

    Example:

        (start, end) of every alias found as a whole word. Overlapping matches are merged into one
        ("jean-baptiste lamarck" and "lamarck" are the same mention).
    """
    found = sorted((m.start(), m.end()) for alias in aliases
                   for m in re.finditer(r"(?<![a-z0-9])" + re.escape(normalize(alias)) + r"(?![a-z0-9])", text))
    merged: list[tuple[int, int]] = []
    for start, end in found:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def grade_number(
    text: str, 
    correct: list[str], 
    distractor: list[str]
    ) -> str | None:
    """Classifies a response using its last non-negated number.
    Label from the last non-negated number, or None if the text has no digits.

    Args:
        text: Response text to evaluate.
        correct: Numeric strings representing correct answers.
        distractor: Numeric strings representing incorrect alternatives.

    Returns:
        "correct", "distractor", or "other" based on the selected number,
        or None if the response contains no digits.
    """
    numbers = [(m.start(), m.group()) for m in re.finditer(r"(?<![\d.])\d+(?:\.\d+)?", text)]
    if not numbers:
        return None
    kept = [n for pos, n in numbers if not is_negated(text, pos, pos + len(n), set(), check_after=False)]
    last = (kept or [n for _, n in numbers])[-1]
    return "correct" if last in correct else "distractor" if last in distractor else "other"


def grade(
    response: str, 
    question: dict
    ) -> str:
    """Classifies a response as correct, distractor, or other.

    Evaluates the final answer using numeric matching, aliases, negation,
    and contrastive expressions.

    Args:
        response: Model response to evaluate.
        question: Question data containing answer aliases and category.

    Returns:
        "correct" if the response supports the correct answer,
        "distractor" if it supports the distractor, or "other" if neither
        answer can be determined.
    """
    text, _ = final_answer(response)
    correct, distractor = question["correct_aliases"], question["distractor_aliases"]

    if question["category"] == "arithmetic" or correct[0].isdigit():
        label = grade_number(text, correct, distractor)
        if label:
            return label  # without digits (e.g. "eight"), fall back to text matching below

    correct_words = {w for a in correct for w in words(normalize(a))}
    distractor_words = {w for a in distractor for w in words(normalize(a))}
    mentions = [(start, label, is_negated(text, start, end, other))  # (position, label, negated)
                for label, aliases, other in [("correct", correct, distractor_words),
                                              ("distractor", distractor, correct_words)]
                for start, end in find_mentions(text, aliases)]
    if not mentions:
        return "other"

    affirmed = [m for m in mentions if not m[2]]
    if not affirmed:  # all negated: rejecting only one option means keeping the other one
        negated = {label for _, label, _ in mentions}
        if len(negated) == 1:
            return "distractor" if negated == {"correct"} else "correct"
        return "other"

    if {label for _, label, _ in affirmed} == {"correct", "distractor"}:
        cuts = [m.end() for m in CONTRAST.finditer(text)]
        after_contrast = [m for m in affirmed if cuts and m[0] >= cuts[-1]]
        if after_contrast:
            return min(after_contrast)[1]
    return min(affirmed)[1]
