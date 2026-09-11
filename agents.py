"""Agent node functions for the AI Radar graph.

Each function takes the RadarState and returns a partial state update.
Code is in English; the briefing itself is generated in Dutch.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from state import Category, CriticVerdict, RadarState, ScoutItem, ScoutReport

load_dotenv()

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
CACHE_DIR = Path(os.getenv("RADAR_CACHE_DIR", ".cache"))
CACHE_ENABLED = os.getenv("RADAR_CACHE", "1") != "0"


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
    )


@lru_cache(maxsize=1)
def get_tavily() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# ---------------------------------------------------------------------------
# Search cache (one JSON file per query per day)
# ---------------------------------------------------------------------------


def _cache_path(query: str, max_results: int, day: str, region: str) -> Path:
    key = hashlib.sha256(
        f"{day}|{max_results}|{region}|{query}".encode()
    ).hexdigest()[:16]
    return CACHE_DIR / day / f"{key}.json"


def _cache_read(path: Path) -> dict | None:
    if not CACHE_ENABLED or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cache_write(path: Path, payload: dict) -> None:
    if not CACHE_ENABLED:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # Caching is best-effort; never fail the run because of it.


# Region profiles: how to steer Tavily for Dutch vs. international news.
REGIONS: dict[str, dict] = {
    "nl": {
        "label": "Nederland",
        "country": "netherlands",
        "language": "nl",
        # Boost (not filter) Dutch outlets so good NL sources rank higher.
        "include_domains": [
            "nos.nl", "nu.nl", "nrc.nl", "volkskrant.nl", "fd.nl", "tweakers.net",
            "computable.nl", "agconnect.nl", "emerce.nl", "rijksoverheid.nl",
            "autoriteitpersoonsgegevens.nl", "ictmagazine.nl",
        ],
        "include_domains_mode": "boost",
    },
    "int": {
        "label": "Internationaal",
        "country": None,
        "language": "en",
        "include_domains": None,
        "include_domains_mode": None,
    },
}


def tavily_search(
    query: str,
    max_results: int = 5,
    day: str | None = None,
    region: str = "int",
) -> dict:
    """Single Tavily search, cached per calendar day and region.

    Running the graph twice on the same day (e.g. after a critic tweak or a
    crash) reuses the stored results instead of spending Tavily credits.
    """
    day = day or date.today().isoformat()
    path = _cache_path(query, max_results, day, region)

    cached = _cache_read(path)
    if cached is not None:
        return cached

    profile = REGIONS[region]
    kwargs = {
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "topic": "news",
        "days": 7,
        "country": profile["country"],
        "language": profile["language"],
        "include_domains": profile["include_domains"],
        "include_domains_mode": profile["include_domains_mode"],
    }
    response = get_tavily().search(**{k: v for k, v in kwargs.items() if v is not None})
    _cache_write(path, response)
    return response


def web_search(
    queries: list[str],
    max_results: int = 5,
    day: str | None = None,
    region: str = "int",
) -> list[dict]:
    """Run several Tavily searches and return de-duplicated raw results."""
    seen: set[str] = set()
    results: list[dict] = []
    for query in queries:
        response = tavily_search(query, max_results=max_results, day=day, region=region)
        for hit in response.get("results", []):
            url = hit.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            results.append(
                {
                    "title": hit.get("title", ""),
                    "source_url": url,
                    "content": (hit.get("content") or "")[:1200],
                    "date": hit.get("published_date"),
                    "region": region,
                }
            )
    return results


# ---------------------------------------------------------------------------
# Generic scout
# ---------------------------------------------------------------------------

SCOUT_SYSTEM = """You are a research scout for a weekly AI briefing.
You receive raw web search results. Select the most relevant, recent and
credible items for your category, remove duplicates and marketing fluff,
and return structured findings. Summaries may be in English; the final
briefing will be translated later. Give an honest hype_score (1 = sober,
5 = unverified marketing) and name the concrete evidence for each item."""


def run_scout(
    category: Category,
    focus: str,
    queries: list[str],
    queries_nl: list[str] | None = None,
    max_items: int = 8,
    day: str | None = None,
) -> list[ScoutItem]:
    """Search internationally (English) and in the Netherlands (Dutch)."""
    raw = web_search(queries, day=day, region="int")
    if queries_nl:
        raw += web_search(queries_nl, day=day, region="nl")
    if not raw:
        return []

    llm = get_llm().with_structured_output(ScoutReport)
    prompt = (
        f"Category: {category}\n"
        f"Focus: {focus}\n"
        f"Return at most {max_items} items. Use category='{category}' for every item.\n"
        "Copy the 'region' field from each result ('nl' = Dutch news, 'int' = international).\n"
        "Include at least 2 Dutch ('nl') items if the results contain any that are relevant.\n\n"
        f"Search results (JSON):\n{json.dumps(raw, ensure_ascii=False, indent=2)}"
    )
    report: ScoutReport = llm.invoke(
        [SystemMessage(content=SCOUT_SYSTEM), HumanMessage(content=prompt)]
    )
    # Force the category in case the model drifts.
    return [item.model_copy(update={"category": category}) for item in report.items]


# ---------------------------------------------------------------------------
# Scout nodes
# ---------------------------------------------------------------------------


def scout_frontier(state: RadarState) -> RadarState:
    """Frontier research: new models, papers, benchmarks, capabilities."""
    items = run_scout(
        category="frontier",
        focus="Frontier AI research and model releases: new foundation models, "
        "papers, benchmark results, capability jumps, alignment and safety research.",
        queries=[
            "new AI model release this week",
            "frontier LLM benchmark results",
            "AI research paper breakthrough",
        ],
        queries_nl=[
            "nieuw AI-model onderzoek",
            "Nederlandse universiteit AI onderzoek doorbraak",
            "GPT-NL taalmodel",
        ],
        day=state.get("date"),
    )
    return {"frontier_items": items}


def scout_productie(state: RadarState) -> RadarState:
    """Production AI: tooling, infrastructure, MLOps, developer platforms."""
    items = run_scout(
        category="productie",
        focus="AI in production: developer tooling, agent frameworks, inference "
        "infrastructure, evaluation/observability, pricing and API changes.",
        queries=[
            "AI agent framework release",
            "LLM inference infrastructure news",
            "AI developer platform update",
        ],
        queries_nl=[
            "AI-agents softwareontwikkeling Nederland",
            "generatieve AI tooling bedrijven",
            "AI infrastructuur datacenter Nederland",
        ],
        day=state.get("date"),
    )
    return {"productie_items": items}


def scout_cases(state: RadarState) -> RadarState:
    """Real-world business cases and adoption stories."""
    items = run_scout(
        category="cases",
        focus="Concrete enterprise and public-sector AI use cases with measurable "
        "results, adoption studies, and lessons learned from deployments.",
        queries=[
            "enterprise AI case study results",
            "company deploys AI agents productivity",
            "AI adoption survey enterprise",
        ],
        queries_nl=[
            "bedrijf zet AI in resultaten",
            "AI overheid gemeente pilot",
            "AI in de zorg ziekenhuis",
        ],
        day=state.get("date"),
    )
    return {"case_items": items}


def scout_hype(state: RadarState) -> RadarState:
    """Hype, controversy, regulation and critical voices."""
    items = run_scout(
        category="hype",
        focus="Hype versus reality: overblown claims, controversies, regulation "
        "(EU AI Act etc.), lawsuits, safety incidents, critical analyses.",
        queries=[
            "AI hype criticism",
            "AI regulation news EU AI Act",
            "AI controversy lawsuit",
        ],
        queries_nl=[
            "AI-verordening Autoriteit Persoonsgegevens",
            "kritiek op AI hype",
            "AI rechtszaak privacy Nederland",
        ],
        day=state.get("date"),
    )
    return {"hype_items": items}


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

SYNTH_SYSTEM = """Je bent de hoofdredacteur van "AI Radar", een wekelijkse
Nederlandstalige briefing voor technisch geïnteresseerde beslissers.

Schrijf de volledige briefing in het Nederlands, in Markdown, met deze structuur:

# AI Radar – {date}

## TL;DR
Drie tot vijf bullets met het belangrijkste van de week.

## Frontier
## Productie
## Cases uit de praktijk
## Hype & kritiek

Per sectie eerst een kopje **Nederland** (bronnen met region 'nl'), daarna
**Buitenland** (region 'int'). Als er voor Nederland niets is, schrijf dat
kort op. Korte alinea's of bullets, elke bewering met een bronlink in de
vorm [titel](url). Wees concreet, vermijd wollige taal, geef duiding
("waarom doet dit ertoe?"). Sluit af met:

## Wat betekent dit voor jou
Twee tot drie praktische aanbevelingen.

Gebruik alleen de aangeleverde bronnen. Verzin geen feiten of links."""


def _items_block(label: str, items: list[ScoutItem]) -> str:
    if not items:
        return f"### {label}\n(geen bevindingen)\n"
    lines = [f"### {label}"]
    for it in sorted(items, key=lambda x: x.hype_score):
        pub = f" ({it.date})" if it.date else ""
        lines.append(
            f"- [{it.region}] {it.title}{pub} | hype {it.hype_score}/5\n  {it.source_url}\n  {it.summary}\n  Bewijs: {it.evidence}"
        )
    return "\n".join(lines) + "\n"


def synthesizer(state: RadarState) -> RadarState:
    """Combine all scout findings into a Dutch briefing."""
    today = state.get("date") or date.today().isoformat()
    sources = "\n".join(
        [
            _items_block("Frontier", state.get("frontier_items", [])),
            _items_block("Productie", state.get("productie_items", [])),
            _items_block("Cases", state.get("case_items", [])),
            _items_block("Hype", state.get("hype_items", [])),
        ]
    )

    messages = [
        SystemMessage(content=SYNTH_SYSTEM.replace("{date}", today)),
        HumanMessage(content=f"Bronnen:\n\n{sources}"),
    ]

    critique = state.get("critique")
    previous = state.get("draft_briefing")
    if critique and previous:
        messages.append(
            HumanMessage(
                content=(
                    "Dit was je vorige versie:\n\n"
                    f"{previous}\n\n"
                    "De critic gaf deze feedback. Verwerk die volledig en lever "
                    f"een verbeterde versie:\n\n{critique}"
                )
            )
        )

    draft = get_llm(temperature=0.4).invoke(messages).content
    return {
        "draft_briefing": draft,
        # First draft is retry 0; every revision after critic feedback counts +1.
        "retries": state.get("retries", 0) + (1 if state.get("critique") else 0),
    }


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = """Je bent een kritische eindredacteur. Beoordeel de briefing op:

1. Feitelijkheid: staat er iets in dat niet door de bronnen wordt gedekt?
2. Volledigheid: ontbreken belangrijke bevindingen met hoge relevantie?
3. Structuur: zijn alle verplichte secties aanwezig?
4. Taal: is alles in correct, helder Nederlands? Geen anglicismen waar een
   Nederlands woord bestaat (behalve vaste vaktermen).
5. Duiding: wordt uitgelegd waarom iets ertoe doet?

Keur alleen goed als er geen substantiële problemen zijn. Geef anders
concrete, puntsgewijze feedback in het Nederlands."""


def critic(state: RadarState) -> RadarState:
    """Review the draft; approve or return feedback for another round."""
    all_items = (
        state.get("frontier_items", [])
        + state.get("productie_items", [])
        + state.get("case_items", [])
        + state.get("hype_items", [])
    )
    sources = "\n".join(f"- {it.title} — {it.source_url}: {it.summary}" for it in all_items)

    llm = get_llm().with_structured_output(CriticVerdict)
    verdict: CriticVerdict = llm.invoke(
        [
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(
                content=(
                    f"Beschikbare bronnen:\n{sources}\n\n"
                    f"Briefing:\n\n{state['draft_briefing']}"
                )
            ),
        ]
    )

    # Stop looping after MAX_RETRIES revisions regardless of verdict.
    forced = state.get("retries", 0) >= MAX_RETRIES
    return {
        "approved": verdict.approved or forced,
        "critique": verdict.feedback,
    }


def finalize(state: RadarState) -> RadarState:
    """Promote the approved draft to the final briefing."""
    return {"final_briefing": state["draft_briefing"]}


# ---------------------------------------------------------------------------
# Rewriter: turns the factual briefing into a publish-ready Dutch article
# ---------------------------------------------------------------------------

STYLE_DIR = Path(__file__).parent / "style"

REWRITER_SYSTEM = """Je bent een ervaren Nederlandse schrijver van duidende
artikelen over digitale technologie, AI en regelgeving. Je herschrijft een
feitelijke nieuwsbriefing tot één samenhangend artikel voor een zakelijk
publiek: ondernemers, managers en beslissers zonder technische achtergrond.

Stijlregels (volg het meegeleverde voorbeeldartikel zo dicht mogelijk):
- Kop (H1): een bewering die de rode draad vat, eventueel met dubbele punt.
- Lead: één vetgedrukte alinea van twee zinnen die de kern en de strekking geeft.
- Korte alinea's van één tot drie zinnen. Zet een enkele zin los als eigen
  alinea wanneer die nadruk verdient.
- Tussenkoppen (H2) zijn stellingen, geen labels. Dus niet "Frontier" maar
  "Nieuwe modellen komen sneller dan organisaties kunnen bijhouden".
- Groepeer per thema of rode draad, niet per bron of per categorie van de
  briefing. Laat onbelangrijke items weg.
- Rustig, uitleggend, analyserend. Gebruik retorische vragen en formuleringen
  als "De onderliggende gedachte is eenvoudig:" of "De boodschap is duidelijk:".
- Maak per thema de vertaalslag naar organisaties: wat betekent dit voor hen?
- Eén vetgedrukte kernzin per sectie mag; verder spaarzaam met vet en cursief.
- Geen bullets, geen genummerde lijsten, geen links en geen bronvermeldingen in
  de lopende tekst. Noem bronnen alleen als naam wanneer dat inhoudelijk
  relevant is ("De Europese Commissie...", "Toezichthouder ACM...").
- Geen hype, geen superlatieven, geen marketingtaal. Beweringen uit zwakke of
  commerciële bronnen formuleer je als claim ("volgens het bedrijf zelf").
- Nederlandse termen waar die bestaan; Engelse vaktermen alleen met korte uitleg.
- Sluit af met een reflecterende sectie die de ontwikkelingen samenbrengt en
  eindigt met een vetgedrukte slotzin.
- Lengte: 700 tot 1100 woorden.

Gebruik uitsluitend feiten uit de briefing. Verzin niets. Sluit het artikel af
met een aparte sectie "## Bronnen" met een kale lijst van de gebruikte
bronlinks (titel – url); die sectie is voor de redactie en mag later worden
verwijderd."""


def load_style_example() -> str:
    """Concatenate all Markdown files in ./style as few-shot style examples."""
    if not STYLE_DIR.exists():
        return ""
    parts = []
    for path in sorted(STYLE_DIR.glob("*.md")):
        parts.append(f"<voorbeeld bestand=\"{path.name}\">\n{path.read_text(encoding='utf-8')}\n</voorbeeld>")
    return "\n\n".join(parts)


def rewriter(state: RadarState) -> RadarState:
    """Rewrite the approved briefing into an article in the house style."""
    example = load_style_example()
    messages = [SystemMessage(content=REWRITER_SYSTEM)]
    if example:
        messages.append(
            HumanMessage(
                content=(
                    "Dit zijn voorbeeldartikelen in de gewenste stijl. Neem de toon, "
                    "opbouw en alinea-lengte over, niet de inhoud:\n\n" + example
                )
            )
        )
    messages.append(
        HumanMessage(
            content=(
                f"Herschrijf deze briefing van {state.get('date', '')} tot een "
                f"artikel in die stijl:\n\n{state['final_briefing']}"
            )
        )
    )
    article = get_llm(temperature=0.5).invoke(messages).content
    return {"article": article}
