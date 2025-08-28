import re
from typing import List, Tuple, Optional

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Try to load spaCy English model with NER; fallback to blank if unavailable
try:
    import en_core_web_sm  # type: ignore
    nlp = en_core_web_sm.load()  # has NER
except Exception:
    nlp = spacy.blank("en")  # minimal pipeline without NER

_sentiment_analyzer = SentimentIntensityAnalyzer()


def clean_text(text: str) -> str:
    if not text:
        return ""
    # Keep letters, numbers, common punctuation; collapse whitespace
    text = re.sub(r"[\r\t]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_sentiment(text: str) -> str:
    if not text:
        return "neutral"
    vs = _sentiment_analyzer.polarity_scores(text)
    compound = vs.get("compound", 0.0)
    if compound >= 0.2:
        return "positive"
    if compound <= -0.2:
        return "negative"
    return "neutral"


def bias_flags(text: str) -> List[str]:
    flags: List[str] = []
    if not text:
        return flags
    patterns = {
        "absolutes": r"\b(always|never|everyone|no one|all|none)\b",
        "superlatives": r"\b(best|worst|greatest|smallest|only|must)\b",
        "hedging": r"\b(might|could|possibly|perhaps|likely)\b",
        "subjective": r"\b(awesome|terrible|amazing|disaster|brilliant)\b",
    }
    for name, pat in patterns.items():
        if re.search(pat, text, flags=re.IGNORECASE):
            flags.append(name)
    return flags


def choose_wh(np_text: str) -> str:
    doc = nlp(np_text)
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    if ents:
        label = ents[0][1]
        if label == 'PERSON':
            return 'Who'
        if label in ('GPE', 'LOC', 'FAC'):
            return 'Where'
        if label in ('DATE', 'TIME'):
            return 'When'
    if re.search(r'(process|method|technique|way|approach|system|strategy)', np_text, re.I):
        return 'How'
    return 'What'


def generate_varied_questions(
    sent: str,
    slide_kws: Optional[List[str]] = None,
    slide_topics: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    sent_clean = clean_text(sent).rstrip('.!?')
    qs: List[Tuple[str, str]] = []

    # RULE 1: Very short text (titles / headings)
    if len(sent_clean.split()) <= 2 and sent_clean:
        title = sent_clean.title()
        qs.extend([
            ("definition", f"What is {title}?"),
            ("importance", f"Why is {title} important?"),
            ("components", f"What are the main components of {title}?"),
            ("application", f"How is {title} applied in practice?"),
            ("example", f"Give an example where {title} is critical."),
            ("challenges", f"What are common challenges in implementing {title}?"),
            ("prediction", f"How might {title} evolve in the next 5 years?"),
            ("evaluation", f"Do you agree that {title} is important? Why or why not?")
        ])
        return qs

    # RULE 2: Detect actual human names → team questions
    people = [ent.text for ent in nlp(sent_clean).ents if ent.label_ == "PERSON"]
    if len(people) > 1:  # multiple people detected
        qs.extend([
            ("factual", "Who are the members mentioned here?"),
            ("factual", "Which team do these members belong to?"),
            ("factual", "What roles might these members play?"),
            ("process", "How can this team organise tasks effectively?"),
            ("challenge", "What challenges could this team face?"),
            ("evaluation", "How can the performance of this team be evaluated?")
        ])
        return qs

    # RULE 3: Quantitative values (percentages, ₹, Cr, large numbers)
    # Avoid low-signal single digits like "1" without units
    numbers = re.findall(r'\d+%|₹[\d,]+|\d+ ?Cr|\d+(?:,\d{3})+|\d{2,}', sent_clean)
    if numbers:
        for num in numbers:
            qs.extend([
                ("quant", f"What does the figure '{num}' represent?"),
                ("trend", f"How has '{num}' changed over time?"),
                ("cause", f"What factors contribute to '{num}'?"),
                ("impact", f"What is the business impact of '{num}'?"),
                ("forecast", f"How might '{num}' change in the future?"),
                ("comparison", f"How does '{num}' compare to industry benchmarks?")
            ])

    # RULE 4: Noun phrase driven questions
    doc = nlp(sent_clean)
    nps = [chunk.text for chunk in doc.noun_chunks] if hasattr(doc, "noun_chunks") else []
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    sentiment = get_sentiment(sent_clean)
    flags = bias_flags(sent_clean)

    if nps:
        target = max(nps, key=lambda x: len(x.split()))
        wh = choose_wh(target)
        # Replace one occurrence of target with WH word
        pattern = r'\b' + re.escape(target) + r'\b'
        replaced = re.sub(pattern, wh, sent_clean, count=1, flags=re.I)
        if replaced == sent_clean:
            replaced = sent_clean.replace(target, wh, 1)

        if replaced:
            qs.append(("factual", replaced[0].upper() + replaced[1:] + "?"))
        qs.append(("conceptual", f"Why is {target} important?"))
        qs.append(("process", f"How does {target} work in this context?"))
        qs.append(("application", f"How can {target} be applied in practice?"))
        qs.append(("benefit", f"What are the benefits of {target}?"))
        qs.append(("challenge", f"What challenges are associated with {target}?"))
        qs.append(("cause_effect", f"What factors influence {target}?"))
        qs.append(("impact", f"What impact does {target} have on strategy?"))
        qs.append(("example", f"Can you give an example of {target}?"))
        qs.append(("evaluation", f"How would you evaluate the success of {target}?"))
        qs.append(("improve", f"How would you improve {target}?"))

        # comparative if multiple noun phrases
        if len(nps) > 1:
            qs.append(("compare", f"What is the difference between {nps[0]} and {nps[1]}?"))
            qs.append(("relation", f"How are {nps[0]} and {nps[1]} related?"))

    # RULE 5: Keyword-driven questions
    if slide_kws:
        for kw in slide_kws[:5]:
            qs.append(("keyword", f"Why is '{kw}' important in this slide?"))
            qs.append(("keyword_app", f"How is '{kw}' applied in the context of this slide?"))
            qs.append(("keyword_future", f"How might '{kw}' evolve in future business practices?"))

    # RULE 6: Sentiment-driven questions
    if sentiment == 'positive':
        qs.append(("opportunity", "What opportunities can be derived from this positive trend?"))
    elif sentiment == 'negative':
        qs.append(("challenge", "What challenges are reflected in this negative outlook?"))
    else:
        qs.append(("neutral_check", "Does this information indicate stability or neutrality?"))

    # RULE 7: Topic-driven questions
    if slide_topics:
        topic_str = ", ".join([str(t) for t in slide_topics])
        qs.append(("topic", f"Which topic does this slide most relate to: {topic_str}?"))
        qs.append(("topic_analysis", f"How does this slide connect with the identified topics?"))

    # RULE 8: Bias flags
    if flags:
        qs.append(("bias_flag", f"Note: This sentence contains possible flags: {flags}"))

    # Deduplicate while preserving order
    seen = set()
    final: List[Tuple[str, str]] = []
    for typ, q in qs:
        if q not in seen:
            final.append((typ, q))
            seen.add(q)
    return final
