================================================================================
 POWER PLANT SAFETY RESEARCH ASSISTANT - README
================================================================================

This app is a ReAct-style LLM agent (LangChain + LangGraph) that answers
questions about US power plants, OSHA severe-injury incidents, and OSHA/CFR
electrical safety guidelines, and can run a "5 Whys" root cause analysis
grounded in that data. It can be used two ways:

  1. A command-line chat loop        -> src/main.py
  2. A browser-based chat GUI (Gradio) with interactive maps -> src/gradio_app.py

Both entry points share the exact same agent logic (tools, system prompt,
Thought/Action/Observation loop) defined in src/main.py, so they behave
identically -- the GUI just adds maps and a nicer chat window on top.

--------------------------------------------------------------------------------
 1. PREREQUISITES
--------------------------------------------------------------------------------

- Python 3.13 (already installed on this machine at /Library/Frameworks/
  Python.framework/Versions/3.13/bin/python3)
- An OpenAI API key (used both for the chat model and for building the
  embeddings used in vector search)

--------------------------------------------------------------------------------
 2. ONE-TIME SETUP
--------------------------------------------------------------------------------

Step 1 - Install dependencies (from the project root):

    pip install -r requirements.txt

Step 2 - Add your secrets file:

    Create/edit src/API.ENV (this file is gitignored, so it stays local) with:

        OPENAI_API_KEY=sk-...your key...

        # Optional - only needed if you want LangSmith tracing:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=ls__...your key...
        LANGCHAIN_PROJECT=your-project-name

    src/main.py loads this file automatically on startup via python-dotenv,
    so both the CLI and the GUI pick these values up.

--------------------------------------------------------------------------------
 3. HOW TO RUN THE APP
--------------------------------------------------------------------------------

Run everything from the project root ("Capstone Project" folder).

  OPTION A - Command-line chat (terminal only):

      python src/main.py

      - Type your question at the "You:" prompt and press Enter.
      - Type "quit" or "exit" (or press Ctrl+C) to stop.

  OPTION B - Browser GUI with maps (recommended):

      python src/gradio_app.py

      - The very first time you run this (cold start), it will print a
        loading log in the browser while it builds the vector store caches
        (see section 4) -- this can take several minutes because it has to
        embed ~106,000 injury records plus two large PDFs.
      - Every run after that reuses the cached embeddings from the data/
        folder and starts in a few seconds.
      - See section 5 below for how to actually open the page once it
        starts.

  Either way, the very first run will also print progress like:

      Checking for cached severe injury narratives embeddings...
      No cached embeddings found for severe injury narratives -- embedding
      106000 chunks with 'text-embedding-3-small' now...

  This is normal and only happens once (or again if you delete the cache
  folders under data/).

--------------------------------------------------------------------------------
 4. HOW THE JSON FILES LOAD INTO THE VECTOR STORE
--------------------------------------------------------------------------------

There are two datasets that get turned into a searchable "vector store"
(an in-memory semantic search index), and one JSON dataset that is queried
directly without any embedding at all:

  A) data/severe_injury_data.json  (~106,000 OSHA severe injury records)
     -> loaded by src/vector_store.py
     -> Each JSON record's "Final Narrative" text becomes one Document; every
        other field (ID, EventDate, Employer, City, State, Latitude,
        Longitude, NatureTitle, etc.) is kept as that Document's metadata.
     -> src/embedding_cache.py sends the documents to OpenAI's
        text-embedding-3-small model in batches of 500 and stores the
        resulting vectors.
     -> Caching: on the FIRST run, the embeddings are written to
            data/severe_injury_embeddings_cache/vectors.npy   (the vectors)
            data/severe_injury_embeddings_cache/records.json  (text + metadata)
        On every run AFTER that, these two files are loaded straight back
        into memory instead of re-calling OpenAI -- this is what makes
        startup fast after the first time.
     -> Used by tools like search_severe_injury_lessons_learned,
        search_severe_injury_near_location, and search_severe_injury_by_vendor.

  B) The safety guideline PDFs (data/safety_guidelines.pdf and
     data/CFR2023title29vol5sec1910-269.pdf)
     -> loaded by src/pdf_vector_store.py (not JSON, but cached the same way)
     -> Each PDF page's text is split into ~1000-character overlapping
        chunks, embedded, and cached under
            data/pdf_embeddings_cache/<pdf name>__cs1000_ov200/
        keyed by filename + chunk size, so each PDF is cached independently
        and adding a new PDF only embeds that new file.
     -> Used by search_safety_guidelines and by root_cause_analysis.

  C) data/power_plant.json  (8,688 US power plants)
     -> loaded directly by src/main.py's _load_power_plants() -- this one is
        NOT embedded into a vector store. It's just read into memory once
        and filtered/counted directly (by name, state, fuel type, capacity,
        owner, etc.) via query_power_plant_database and count_power_plants,
        since exact filtering doesn't need semantic search.
     -> Plant coordinates are also reverse-geocoded once at startup (into
        city/county/state) and cached in memory, so plants can be linked to
        nearby severe-injury incidents by location.

In short: the two big text datasets (injury narratives, safety PDFs) are
embedded once and cached to disk as vectors.npy + records.json so future
runs skip the expensive OpenAI embedding step; the power plant JSON is just
loaded and filtered directly, no embeddings involved.

--------------------------------------------------------------------------------
 5. HOW TO OPEN THE GUI IN YOUR BROWSER
--------------------------------------------------------------------------------

After running:

    python src/gradio_app.py

Gradio will print something like this in the terminal:

    Running on local URL:  http://127.0.0.1:7860

To open the app:

  - Hold Cmd (Mac) and click the http://127.0.0.1:7860 link directly in the
    terminal (most terminals, including VS Code's, turn it into a clickable
    link) -- this opens it in your default browser.
  - If clicking doesn't work, just copy the URL and paste it into your
    browser's address bar manually.
  - The page will first show a "Starting up" screen with a live log while
    the backend loads (see section 3); once loading finishes it
    automatically switches to the chat interface -- no need to refresh.

To stop the app, either press Ctrl+C in the terminal, or click the
"Quit" button in the top-right corner of the page.

--------------------------------------------------------------------------------
 6. TROUBLESHOOTING
--------------------------------------------------------------------------------

- "OPENAI_API_KEY" errors: make sure src/API.ENV exists and contains a
  valid key, and that you're running the app from the project root.
- First run is slow / seems stuck: this is expected -- check the terminal
  (CLI) or the "Startup log" box (GUI) for embedding progress messages.
- Want to force a rebuild of the embeddings: delete the relevant cache
  folder under data/ (severe_injury_embeddings_cache/ or
  pdf_embeddings_cache/) and re-run -- it will be rebuilt automatically.
- Root cause analysis is expensive: the root_cause_analysis tool only runs
  when your question explicitly and literally includes the phrase "root
  cause" (see the REACT_SYSTEM_PROMPT gating rule in src/main.py) -- an
  ordinary "why did this happen" question is answered with the regular
  search tools instead. When it does run, it performs a 5 Whys tree-of-
  thought beam search (~15 LLM calls plus a similar number of safety-
  guideline lookups) and uses roughly 50,000 tokens to complete, so expect
  it to take noticeably longer and cost more than a normal question.
