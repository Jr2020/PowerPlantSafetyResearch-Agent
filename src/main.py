#Main file will be used to work throught the Agentic AI exercise

import json
import math
import os
import re
from typing import Optional

import pymupdf
import reverse_geocoder
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from pdf_vector_store import load_pdf_vector_stores
from root_cause_analysis import run_five_whys_beam_search
from vector_store import load_severe_injury_vector_store

# Reads OPENAI_API_KEY (and any other vars) from API.ENV into the environment.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "API.ENV"))

MODEL = "gpt-5-mini"

llm = ChatOpenAI(model=MODEL, api_key=os.getenv("OPENAI_API_KEY"))

# Loaded once at startup: embeds severe_injury_data.json on the first run and
# reuses the cached embeddings on every run after that (see vector_store.py).
SEVERE_INJURY_VECTOR_STORE = load_severe_injury_vector_store()

# Loaded once at startup: chunks and embeds each safety PDF on the first run
# and reuses the cached embeddings on every run after that (see
# pdf_vector_store.py). Each file is cached independently, so adding another
# PDF to this list only embeds the new one.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SAFETY_GUIDELINES_PDF_PATHS = [
    os.path.join(DATA_DIR, "safety_guidelines.pdf"),
    os.path.join(DATA_DIR, "CFR2023title29vol5sec1910-269.pdf"),
]
SAFETY_GUIDELINES_VECTOR_STORE = load_pdf_vector_stores(SAFETY_GUIDELINES_PDF_PATHS)
# ---------------------------------------------------------------------------
# 1. Same mock tools as Step 2, now declared with LangChain's @tool decorator
# ---------------------------------------------------------------------------

POWER_PLANT_JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "power_plant.json"
)

MAX_POWER_PLANT_RESULTS = 200

_power_plant_cache: Optional[list[dict]] = None

def _load_power_plants() -> list[dict]:
    """Load and cache power_plant.json so repeated tool calls don't re-read it from disk."""
    global _power_plant_cache
    if _power_plant_cache is None:
        with open(POWER_PLANT_JSON_PATH, "r") as f:
            _power_plant_cache = json.load(f)
    return _power_plant_cache


_power_plant_locations_cache: Optional[dict[str, dict]] = None


def _load_power_plant_locations() -> dict[str, dict]:
    """Resolve city/county/state for every power plant once via reverse-geocoding
    and cache in memory, so state-level filtering doesn't need a per-query lookup
    (power_plant.json has no city/state field, only raw coordinates)."""
    global _power_plant_locations_cache
    if _power_plant_locations_cache is not None:
        return _power_plant_locations_cache

    plants = _load_power_plants()
    print(f"Resolving city/state for {len(plants)} power plants...")
    coords = [(p["latitude"], p["longitude"]) for p in plants]
    geocoded = reverse_geocoder.search(coords, mode=1)

    _power_plant_locations_cache = {
        plant["gppd_idnr"]: {
            "city": geo["name"],
            "county": geo["admin2"],
            "state": geo["admin1"],
        }
        for plant, geo in zip(plants, geocoded)
    }
    print("Done resolving power plant locations.")
    return _power_plant_locations_cache


def _matches_power_plant_filters(
    plant: dict,
    locations: dict[str, dict],
    *,
    name: Optional[str],
    country: Optional[str],
    state: Optional[str],
    primary_fuel: Optional[str],
    min_capacity_mw: Optional[float],
    max_capacity_mw: Optional[float],
    commissioning_year: Optional[int],
    owner: Optional[str],
) -> bool:
    """Shared filter predicate for query_power_plant_database and
    count_power_plants, so the two tools can never disagree on what counts
    as a match."""
    if name and name.lower() not in (plant.get("name") or "").lower():
        return False
    if country:
        country_lower = country.lower()
        if (
            country_lower not in (plant.get("country") or "").lower()
            and country_lower not in (plant.get("country_long") or "").lower()
        ):
            return False
    if state:
        plant_state = locations.get(plant["gppd_idnr"], {}).get("state") or ""
        if state.lower() not in plant_state.lower():
            return False
    if primary_fuel and primary_fuel.lower() != (plant.get("primary_fuel") or "").lower():
        return False
    if min_capacity_mw is not None and (plant.get("capacity_mw") or 0) < min_capacity_mw:
        return False
    if max_capacity_mw is not None and (plant.get("capacity_mw") or 0) > max_capacity_mw:
        return False
    if commissioning_year is not None and plant.get("commissioning_year") != commissioning_year:
        return False
    if owner and owner.lower() not in (plant.get("owner") or "").lower():
        return False
    return True


@tool
def query_power_plant_database(
    name: Optional[str] = None,
    country: Optional[str] = None,
    state: Optional[str] = None,
    primary_fuel: Optional[str] = None,
    min_capacity_mw: Optional[float] = None,
    max_capacity_mw: Optional[float] = None,
    commissioning_year: Optional[int] = None,
    owner: Optional[str] = None,
) -> str:
    """Query the power plant database (loaded from power_plant.json, 8,688
    plants across the United States) by filters. All filters are optional and
    combine with AND; omit a filter to not restrict on it. Results are capped
    to avoid flooding the conversation — a broad filter can match thousands
    of plants, so if the question only needs a count or total capacity rather
    than the actual list of plants, use count_power_plants instead (it has no
    cap, since it never lists individual plants). If a filtered query (e.g. a
    specific state) returns no plants, that means there genuinely are none
    matching — report that directly rather than assuming none exist without
    having queried. Each result includes the plant's resolved city/state and
    its raw latitude/longitude, which you can pass to
    search_severe_injury_near_location to find nearby safety incidents (there is
    no reliable name/keyword link between this database and the injury data).

    Args:
        name: Plant name to filter on (substring match), e.g. 'Kajaki'. Names differentiate a power plant in a customers portfolio.
        country: Country name or 3-letter code to filter on, e.g. 'Afghanistan' or 'AFG'. This dataset is US-only, so a non-US filter here will always return no results.
        state: US state to filter on (substring match), e.g. 'Florida'. Resolved from coordinates, not a raw field.
        primary_fuel: Primary fuel/energy source, e.g. 'Hydro', 'Coal', 'Solar'.
        min_capacity_mw: Only include plants with capacity >= this many megawatts.
        max_capacity_mw: Only include plants with capacity <= this many megawatts.
        commissioning_year: Only include plants commissioned in this exact year.
        owner: This is the best way to figure out who owns the asset (substring match), e.g. 'Duke Energy'. Sometimes the same company owns assets in different states so the holding company sometimes changes but the overall name still holds the parent company details.
    """
    plants = _load_power_plants()
    locations = _load_power_plant_locations()

    results = [
        p for p in plants
        if _matches_power_plant_filters(
            p, locations,
            name=name, country=country, state=state, primary_fuel=primary_fuel,
            min_capacity_mw=min_capacity_mw, max_capacity_mw=max_capacity_mw,
            commissioning_year=commissioning_year, owner=owner,
        )
    ]

    if not results:
        return "No power plants found matching the given filters."

    total_matches = len(results)
    truncated = results[:MAX_POWER_PLANT_RESULTS]

    lines = []
    for p in truncated:
        loc = locations.get(p["gppd_idnr"], {})
        lines.append(
            f"[{p['gppd_idnr']}] {p['name']} | {loc.get('city')}, {loc.get('state')}, "
            f"{p['country_long']} | {p['primary_fuel']} | {p['capacity_mw']} MW | "
            f"commissioned: {p.get('commissioning_year') or 'unknown'} | "
            f"lat/long: {p.get('latitude')}, {p.get('longitude')}"
        )

    if total_matches > MAX_POWER_PLANT_RESULTS:
        lines.append(
            f"... {total_matches - MAX_POWER_PLANT_RESULTS} more matches not shown "
            f"({total_matches} total). Narrow the filters to see more of the list, "
            f"or use count_power_plants if you just need the total/breakdown."
        )

    return "\n".join(lines)


@tool
def count_power_plants(
    name: Optional[str] = None,
    country: Optional[str] = None,
    state: Optional[str] = None,
    primary_fuel: Optional[str] = None,
    min_capacity_mw: Optional[float] = None,
    max_capacity_mw: Optional[float] = None,
    commissioning_year: Optional[int] = None,
    owner: Optional[str] = None,
) -> str:
    """Get the total count and capacity of power plants matching the given
    filters, broken down by primary fuel type — WITHOUT listing individual
    plants, so there's no result cap and no risk of a truncated answer. Use
    this instead of query_power_plant_database whenever the question is a
    count or total ('how many coal plants are in Texas', 'total capacity of
    Duke Energy's fleet') rather than needing the actual list of matching
    plants. Filters are identical to query_power_plant_database's and combine
    with AND.

    Args:
        name: Plant name to filter on (substring match), e.g. 'Kajaki'.
        country: Country name or 3-letter code to filter on, e.g. 'Afghanistan' or 'AFG'. This dataset is US-only, so a non-US filter here will always return no results.
        state: US state to filter on (substring match), e.g. 'Florida'.
        primary_fuel: Primary fuel/energy source, e.g. 'Hydro', 'Coal', 'Solar'.
        min_capacity_mw: Only include plants with capacity >= this many megawatts.
        max_capacity_mw: Only include plants with capacity <= this many megawatts.
        commissioning_year: Only include plants commissioned in this exact year.
        owner: Company name to filter on (substring match), e.g. 'Duke Energy'.
    """
    plants = _load_power_plants()
    locations = _load_power_plant_locations()

    results = [
        p for p in plants
        if _matches_power_plant_filters(
            p, locations,
            name=name, country=country, state=state, primary_fuel=primary_fuel,
            min_capacity_mw=min_capacity_mw, max_capacity_mw=max_capacity_mw,
            commissioning_year=commissioning_year, owner=owner,
        )
    ]

    if not results:
        return "No power plants found matching the given filters."

    by_fuel: dict[str, list[dict]] = {}
    for plant in results:
        by_fuel.setdefault(plant.get("primary_fuel") or "Unknown", []).append(plant)

    total_capacity = sum(p.get("capacity_mw") or 0 for p in results)

    lines = [
        f"Total matching plants: {len(results)}",
        f"Total capacity: {total_capacity:,.1f} MW",
        "Breakdown by primary fuel (plant count | total capacity):",
    ]
    for fuel, group in sorted(by_fuel.items(), key=lambda item: len(item[1]), reverse=True):
        fuel_capacity = sum(p.get("capacity_mw") or 0 for p in group)
        lines.append(f"  {fuel}: {len(group)} | {fuel_capacity:,.1f} MW")

    return "\n".join(lines)


@tool
def get_location_from_coordinates(latitude: float, longitude: float) -> str:
    """Reverse-geocode a latitude/longitude into the nearest known city, county,
    state/region, and country. Power plant records only carry raw coordinates
    (no city/state field), so use this instead of guessing a location from
    lat/long — that guessing is exactly what produces inconsistent results.

    Args:
        latitude: Latitude to look up, e.g. from query_power_plant_database.
        longitude: Longitude to look up, e.g. from query_power_plant_database.
    """
    result = reverse_geocoder.search([(latitude, longitude)])[0]
    return f"{result['name']}, {result['admin2']}, {result['admin1']}, {result['cc']}"


MAX_SEVERE_INJURY_RESULTS = 5

@tool
def search_severe_injury_lessons_learned(query: str, k: int = MAX_SEVERE_INJURY_RESULTS) -> str:
    """Run a similarity search over ~106,000 real OSHA severe injury incident
    narratives (severe_injury_data.json) to find prior incidents conceptually
    or topically related to the query, for cases a keyword search would miss.

    Args:
        query: A natural-language description of the risk, incident, or scenario to find similar past incidents for, e.g. 'worker injured by unguarded machinery' or 'fall from ladder'.
        k: Number of top matches to return (default 5).
    """
    results = SEVERE_INJURY_VECTOR_STORE.similarity_search(query, k=k)

    if not results:
        return "No related severe injury incidents found."

    lines = []
    for doc in results:
        meta = doc.metadata
        lines.append(
            f"[{meta.get('ID')}] {meta.get('EventDate')} | {meta.get('Employer')} | "
            f"{meta.get('City')}, {meta.get('State')} | {meta.get('NatureTitle')} | "
            f"lat/long: {meta.get('Latitude')}, {meta.get('Longitude')} — "
            f"{doc.page_content}"
        )

    return "\n".join(lines)


MAX_SAFETY_GUIDELINES_RESULTS = 5


@tool
def search_safety_guidelines(query: str, k: int = MAX_SAFETY_GUIDELINES_RESULTS) -> str:
    """Run a similarity search over the electrical safety guideline documents
    (safety_guidelines.pdf, a 719-page OSHA final rule, and
    CFR2023title29vol5sec1910-269.pdf, the codified 29 CFR 1910.269 text) to
    find the regulatory text most relevant to a topic or scenario.

    Args:
        query: A natural-language description of the safety topic, procedure, or scenario to find relevant guideline text for, e.g. 'fall protection requirements' or 'lockout tagout procedures'.
        k: Number of top matching chunks to return (default 5).
    """
    results = SAFETY_GUIDELINES_VECTOR_STORE.similarity_search(query, k=k)

    if not results:
        return "No relevant safety guideline text found."

    return "\n\n".join(
        f"[{doc.metadata.get('source')}, page {doc.metadata.get('page')}] {doc.page_content}"
        for doc in results
    )


EARTH_RADIUS_MILES = 3958.8


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * math.asin(math.sqrt(a))


@tool
def search_severe_injury_near_location(
    latitude: float,
    longitude: float,
    query: str,
    radius_miles: float = 50,
    k: int = MAX_SEVERE_INJURY_RESULTS,
) -> str:
    """Find real OSHA severe injury incidents that are both geographically near a
    location and semantically related to a query. Use this to connect a power
    plant to nearby safety incidents by geolocation — there is no reliable
    name/keyword link between the power plant data and the injury data, so
    proximity plus semantic search is how the relationship is found instead.

    Args:
        latitude: Latitude of the reference location, e.g. a power plant's latitude from query_power_plant_database.
        longitude: Longitude of the reference location, e.g. a power plant's longitude from query_power_plant_database.
        query: A natural-language description of the risk or incident type to search for, e.g. 'turbine explosion' or 'electrocution'.
        radius_miles: Only consider incidents within this many miles of the location (default 50).
        k: Number of top matches to return (default 5).
    """

    def within_radius(doc: Document) -> bool:
        try:
            incident_lat = float(doc.metadata.get("Latitude"))
            incident_lon = float(doc.metadata.get("Longitude"))
        except (TypeError, ValueError):
            return False
        return _haversine_miles(latitude, longitude, incident_lat, incident_lon) <= radius_miles

    results = SEVERE_INJURY_VECTOR_STORE.similarity_search(query, k=k, filter=within_radius)

    if not results:
        return (
            f"No severe injury incidents found within {radius_miles} miles of "
            f"({latitude}, {longitude}) matching '{query}'."
        )

    lines = []
    for doc in results:
        meta = doc.metadata
        distance = _haversine_miles(
            latitude, longitude, float(meta["Latitude"]), float(meta["Longitude"])
        )
        lines.append(
            f"[{meta.get('ID')}] {distance:.1f} mi away | {meta.get('EventDate')} | "
            f"{meta.get('Employer')} | {meta.get('City')}, {meta.get('State')} | "
            f"{meta.get('NatureTitle')} | "
            f"lat/long: {meta.get('Latitude')}, {meta.get('Longitude')} — {doc.page_content}"
        )

    return "\n".join(lines)


MAX_VENDOR_INJURY_RESULTS = 10


@tool
def search_severe_injury_by_vendor(
    vendor: str,
    state: Optional[str] = None,
    category: Optional[str] = None,
    k: int = MAX_VENDOR_INJURY_RESULTS,
) -> str:
    """Look up real OSHA severe injury incidents by employer/vendor name — the
    inverted starting point of the plant-first workflow: instead of starting
    from a power plant's location and finding nearby incidents, this starts
    from a specific vendor/employer and returns its incidents directly (each
    with its coordinates), so you can then call
    search_power_plants_near_location on one of those coordinates to find
    which power plants that vendor's incidents happened near — there is no
    reliable name/keyword link between the power plant data and the injury
    data, so this location handoff is how the relationship is found instead.

    Args:
        vendor: Employer/vendor name to filter on (substring match, case-insensitive), e.g. 'Duke Energy'. This is the same kind of company name query_power_plant_database's owner filter uses, though spelling can vary between the two datasets.
        state: Optional US state to narrow results to (substring match), e.g. 'Florida'.
        category: Optional injury nature/category to filter on (substring match against the OSHA nature-of-injury classification), e.g. 'Fracture', 'Burn', 'Amputation'.
        k: Number of matching incidents to return (default 10).
    """

    def matches(meta: dict) -> bool:
        if vendor.lower() not in (meta.get("Employer") or "").lower():
            return False
        if state and state.lower() not in (meta.get("State") or "").lower():
            return False
        if category and category.lower() not in (meta.get("NatureTitle") or "").lower():
            return False
        return True

    # Reuses the vector store's already-loaded documents purely as an
    # in-memory record list here — no embedding search involved, since
    # matching a vendor name needs exact substring filtering, not semantic
    # similarity.
    matching_entries = [
        entry
        for entry in SEVERE_INJURY_VECTOR_STORE.store.values()
        if matches(entry["metadata"])
    ]

    if not matching_entries:
        return f"No severe injury incidents found for vendor '{vendor}'."

    total_matches = len(matching_entries)
    truncated = matching_entries[:k]

    lines = []
    for entry in truncated:
        meta = entry["metadata"]
        lines.append(
            f"[{meta.get('ID')}] {meta.get('EventDate')} | {meta.get('Employer')} | "
            f"{meta.get('City')}, {meta.get('State')} | {meta.get('NatureTitle')} | "
            f"lat/long: {meta.get('Latitude')}, {meta.get('Longitude')} — "
            f"{entry['text']}"
        )

    if total_matches > k:
        lines.append(
            f"... {total_matches - k} more matches not shown ({total_matches} total). "
            f"Narrow the filters or increase k to see different results."
        )

    return "\n".join(lines)


@tool
def search_power_plants_near_location(
    latitude: float,
    longitude: float,
    radius_miles: float = 50,
    primary_fuel: Optional[str] = None,
    k: int = MAX_POWER_PLANT_RESULTS,
) -> str:
    """Find power plants geographically near a location, sorted by distance —
    the inverse of the plant-first workflow: instead of starting from a
    plant's own filters, this starts from a location (e.g. a severe injury
    incident's coordinates from search_severe_injury_by_vendor or
    search_severe_injury_near_location) and finds which power plants are
    nearby. Use this to connect a vendor's safety incidents back to the power
    plant(s) they most likely occurred at or near.

    Args:
        latitude: Latitude of the reference location, e.g. an incident's latitude from search_severe_injury_by_vendor.
        longitude: Longitude of the reference location, e.g. an incident's longitude from search_severe_injury_by_vendor.
        radius_miles: Only consider plants within this many miles of the location (default 50).
        primary_fuel: Optional primary fuel/energy source filter, e.g. 'Hydro', 'Coal', 'Solar'.
        k: Number of nearest matches to return (default 25).
    """
    plants = _load_power_plants()
    locations = _load_power_plant_locations()

    candidates = []
    for plant in plants:
        if primary_fuel and primary_fuel.lower() != (plant.get("primary_fuel") or "").lower():
            continue
        distance = _haversine_miles(latitude, longitude, plant["latitude"], plant["longitude"])
        if distance <= radius_miles:
            candidates.append((distance, plant))

    if not candidates:
        return (
            f"No power plants found within {radius_miles} miles of "
            f"({latitude}, {longitude})."
        )

    candidates.sort(key=lambda pair: pair[0])
    total_matches = len(candidates)
    truncated = candidates[:k]

    lines = []
    for distance, plant in truncated:
        loc = locations.get(plant["gppd_idnr"], {})
        lines.append(
            f"[{plant['gppd_idnr']}] {plant['name']} | {loc.get('city')}, {loc.get('state')}, "
            f"{plant['country_long']} | {plant['primary_fuel']} | {plant['capacity_mw']} MW | "
            f"lat/long: {plant.get('latitude')}, {plant.get('longitude')} | "
            f"{distance:.1f} mi away"
        )

    if total_matches > k:
        lines.append(
            f"... {total_matches - k} more matches not shown ({total_matches} total). "
            f"Narrow the radius or fuel filter to see different results."
        )

    return "\n".join(lines)


MAX_ROOT_CAUSE_EVIDENCE_RESULTS = 5


def _retrieve_root_cause_guidance(query: str) -> str:
    """Fresh per-node RAG lookup for run_five_whys_beam_search: re-queries the
    safety guideline vector store with a specific candidate cause (rather
    than the original problem statement, which is all the one-time evidence
    below is grounded in and goes stale a few 'why' levels down), so each
    branch of the beam search gets guideline text relevant to what THAT
    branch actually claims."""
    return search_safety_guidelines.invoke({"query": query, "k": MAX_ROOT_CAUSE_EVIDENCE_RESULTS})


@tool
def root_cause_analysis(problem: str) -> str:
    """Perform a deep root cause analysis using the 5 Whys methodology via
    tree-of-thought beam search, grounded in this app's severe injury and
    safety guideline data — including a fresh safety-guideline lookup for
    every individual candidate cause the search considers, not just the
    original problem statement, so each level of the why-chain is grounded
    in guideline text relevant to that specific branch. This runs ~15 LLM
    calls plus a similar number of guideline lookups and takes tens of
    seconds — reserve it STRICTLY for when the user's request explicitly
    and literally asks for a "root cause" analysis. Do not call this for an
    ordinary "why did this happen" question; use
    search_severe_injury_lessons_learned or search_safety_guidelines
    directly for those instead — they are faster and usually sufficient.

    Args:
        problem: The problem statement to root-cause, e.g. 'Turbine bearing failure caused an unplanned outage at the plant.'
    """
    injury_evidence = search_severe_injury_lessons_learned.invoke(
        {"query": problem, "k": MAX_ROOT_CAUSE_EVIDENCE_RESULTS}
    )
    guideline_evidence = search_safety_guidelines.invoke(
        {"query": problem, "k": MAX_ROOT_CAUSE_EVIDENCE_RESULTS}
    )
    evidence = (
        f"Related severe injury incidents:\n{injury_evidence}\n\n"
        f"Related safety guideline text:\n{guideline_evidence}"
    )
    return run_five_whys_beam_search(
        problem, evidence, llm, retrieve_guidance=_retrieve_root_cause_guidance
    )


TOOLS = [
    query_power_plant_database,
    count_power_plants,
    get_location_from_coordinates,
    search_severe_injury_lessons_learned,
    search_severe_injury_near_location,
    search_severe_injury_by_vendor,
    search_power_plants_near_location,
    search_safety_guidelines,
    root_cause_analysis,
]

# ---------------------------------------------------------------------------
# 2. The ReAct system prompt: forces explicit reasoning before each action
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """You are a research assistant that helps project managers investigate
maintenance history by reasoning step by step and retrieving evidence from
a fleet's lessons-learned documentation — never from memory or assumption,
since this content is proprietary, unstructured, and was never part of
your training data.

You have access to tools covering four areas: the US power plant database
(lookup and count/aggregate variants), OSHA severe injury incident data
(semantic search, plus a plant-first and a vendor-first location-based pair
that connect plants to incidents in either direction), OSHA/CFR electrical
safety guideline text (semantic search), and a slow, gated 5 Whys root
cause tool. Each tool's own description is authoritative on exactly when to
use it versus its neighbors (including which tool to call next with what
argument) — read those rather than guessing from a tool's name alone.

For every research question, follow this loop:

1. Thought: a 1-3 sentence message reasoning about what the user is
   actually trying to find (a risk factor, prior occurrences of an issue,
   a root-cause pattern across the fleet, etc.), and which tool this
   specific question requires. Not every question needs every tool — a
   question naming a specific plant, country, or fuel type may only need
   the power plant database; an open-ended "has this happened before"
   question needs similarity search.
2. Call the tool you decided on, using your tool-calling ability, attached
   to that same message — this is a real function call the platform
   executes for you, not something you write out yourself; putting a tool
   name or its arguments into your message text does not call it and
   retrieves nothing. Skip this step only once you already have enough
   evidence to answer.
3. You'll receive that tool's result before your next Thought.

Rules for how you reason:
- Every message is either a Thought with a tool call attached, or a Final
  Answer — never a Thought alone with no tool call and no answer.
- Never state a fact, prior incident, or root cause that isn't directly
  supported by a specific retrieved document or query result. If you
  don't have that evidence yet, call a tool to get it next — do not fill
  the gap from general knowledge, since this fleet's maintenance history
  was never in your training data and any unsupported claim here is a
  fabrication, not an inference.
- If similarity search and the power plant database return different or
  conflicting candidates, say so explicitly in your Thought and
  investigate both rather than silently picking one.
- If your first retrieval strategy doesn't surface relevant results,
  say so explicitly, and try a different approach (e.g. broaden a
  similarity search, adjust the power plant filters, or search adjacent
  countries/fuel types) rather than concluding no prior occurrences
  exist after a single attempt.
- Keep track of which documents you've already reviewed so you don't
  re-retrieve or re-cite the same one as if it were new evidence.
- root_cause_analysis triggers ONLY when the user's own words literally
  include the phrase "root cause". An ordinary "why did X happen/fail"
  question, no matter how serious, is answered with the search tools
  instead — calling root_cause_analysis on such a question is a gating
  violation, not a judgment call.
- The vendor-first tool pair (search_severe_injury_by_vendor then
  search_power_plants_near_location) is just as valid a starting point as
  the plant-first one when the question names a vendor/employer rather than
  a plant — not a fallback to reach for only once the plant-first pair
  fails.

When you are ready to answer, prefix your response with "Final Answer:"
and include: (1) a direct answer to the PM's question, (2) the specific
documents/records you're citing as evidence (document ID, plant name,
country — whatever identifying detail is available), (3) any relevant
documents you found but excluded and why (e.g. unrelated result), and
(4) a confidence note (high / medium / low) based on how much direct
evidence was found versus how much required inference across partial
matches.
"""

MAX_STEPS = 6  # safety cap so a loop can't run forever

# create_react_agent runs the same Thought (agent node) / Action (tools node)
# loop this app used to hand-roll in run_react_turn, driven off the same
# structured tool_calls the manual version checked — LangGraph just owns the
# invoke-model / run-tools / repeat-until-no-tool-calls cycle instead of us.
REACT_AGENT = create_react_agent(llm, TOOLS)

# Each Thought/Action round trip is two LangGraph supersteps (agent, then
# tools), so this caps the same number of rounds MAX_STEPS did.
REACT_RECURSION_LIMIT = MAX_STEPS * 2

# create_react_agent tracks its own remaining-step budget (derived from
# recursion_limit) and, rather than raising, gracefully ends the turn with an
# AIMessage carrying exactly this text once that budget runs out. Matched
# verbatim so that case is surfaced as our "warning" step type, same as the
# old hand-rolled loop's MAX_STEPS cutoff.
REACT_AGENT_OUT_OF_STEPS_MESSAGE = "Sorry, need more steps to process this request."


def run_react_turn(messages: list) -> list[dict]:
    """Run one ReAct Thought/Action/Observation loop against `messages`, which
    must already end with the new HumanMessage. Mutates `messages` in place
    with the model's replies and tool results (so the next turn sees the full
    history), and returns the turn as a list of step dicts:
      {"type": "thought", "text": ...}
      {"type": "action", "tool": ..., "args": ..., "observation": ...}
      {"type": "final", "text": ...}
      {"type": "warning", "text": ...}
      {"type": "usage", "input_tokens": ..., "output_tokens": ..., "total_tokens": ...}
    Structured rather than pre-formatted text so each caller can render it
    appropriately — plain text for the terminal, markdown for the Gradio GUI
    — while both run the exact same agent, not two copies of it. The usage
    step sums every LLM call made this turn — a turn can involve several
    (one per Thought/Action before the Final Answer), each billed separately.
    """
    steps = []
    input_tokens = 0
    output_tokens = 0
    pending_calls = {}  # tool_call_id -> {"tool": name, "args": args}, until its ToolMessage arrives
    seen = len(messages)
    hit_recursion_limit = False

    try:
        for state in REACT_AGENT.stream(
            {"messages": messages},
            config={"recursion_limit": REACT_RECURSION_LIMIT},
            stream_mode="values",
        ):
            all_messages = state["messages"]
            new_messages = all_messages[seen:]
            seen = len(all_messages)

            for msg in new_messages:
                usage = getattr(msg, "usage_metadata", None) or {}
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)

                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        if msg.content:
                            steps.append({"type": "thought", "text": msg.content})
                        for tool_call in msg.tool_calls:
                            pending_calls[tool_call["id"]] = {
                                "tool": tool_call["name"],
                                "args": tool_call["args"],
                            }
                    elif msg.content == REACT_AGENT_OUT_OF_STEPS_MESSAGE:
                        steps.append({"type": "warning", "text": "Hit MAX_STEPS without a Final Answer."})
                    else:
                        steps.append({"type": "final", "text": msg.content})
                elif isinstance(msg, ToolMessage):
                    call = pending_calls.pop(
                        msg.tool_call_id, {"tool": msg.name, "args": {}}
                    )
                    steps.append({
                        "type": "action",
                        "tool": call["tool"],
                        "args": call["args"],
                        "observation": str(msg.content),
                    })

            messages[:] = all_messages
    except GraphRecursionError:
        hit_recursion_limit = True

    # Fallback in case GraphRecursionError is ever raised directly (e.g. a
    # future create_react_agent version, or the graceful out-of-steps check
    # above not firing) instead of create_react_agent's own graceful stop.
    if hit_recursion_limit and not any(step["type"] == "warning" for step in steps):
        steps.append({"type": "warning", "text": "Hit MAX_STEPS without a Final Answer."})

    steps.append({
        "type": "usage",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    })

    return steps


def format_turn_as_text(steps: list[dict]) -> str:
    """Render run_react_turn()'s steps as the plain-text transcript the CLI
    has always shown."""
    lines = []
    for step in steps:
        if step["type"] == "thought":
            lines.append(step["text"])
        elif step["type"] == "action":
            lines.append(f"[Action] {step['tool']}({step['args']})")
            lines.append(f"[Observation] {step['observation']}")
        elif step["type"] == "final":
            lines.append(step["text"])
        elif step["type"] == "warning":
            lines.append(f"[Warning] {step['text']}")
        elif step["type"] == "usage":
            lines.append(
                f"[Tokens] input={step['input_tokens']} "
                f"output={step['output_tokens']} total={step['total_tokens']}"
            )
    return "\n\n".join(lines)


def run_react_loop():
    messages = [SystemMessage(content=REACT_SYSTEM_PROMPT)]

    print("Type 'quit' to exit, or press Ctrl+C at any time.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nQuit requested. Exiting.")
            break

        if user_input.lower() in ("quit", "exit"):
            break

        messages.append(HumanMessage(content=user_input))

        try:
            print(f"\n{format_turn_as_text(run_react_turn(messages))}")
        except KeyboardInterrupt:
            print("\nQuit requested. Exiting.")
            break

        print()  # spacing before next user turn


if __name__ == "__main__":
    run_react_loop()
