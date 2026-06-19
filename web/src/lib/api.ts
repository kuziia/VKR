export type Topic = {
  id: string
  display_name: string
  keywords: string[]
  description: string
}
export type Subfield = { id: string; display_name: string; topics: Topic[] }
export type Field = { id: string; display_name: string; subfields: Subfield[] }
export type Domain = { id: string; display_name: string; fields: Field[] }
export type Taxonomy = { domains: Domain[] }

export type Level = 'all' | 'domain' | 'field' | 'subfield' | 'topic'
export type Granularity = 'month' | 'quarter' | 'year'
export type Source = 'openalex' | 'openaire'

export type TrendPoint = { period: string; count: number }
export type TrendsResponse = {
  source: Source
  level: Level
  id: string | null
  label: string | null
  granularity: Granularity
  country: string | null
  lang: string | null
  from: string
  to: string
  points: TrendPoint[]
  total: number
  notes: string | null
}

export type WorkRef = {
  id: string | null
  display_name: string | null
}
export type TopCitedItem = {
  openalex_id: string
  doi: string | null
  title: string
  publication_year: number | null
  publication_date: string | null
  language: string | null
  cited_by_count: number
  authors: string[]
  primary_topic: {
    id: string | null
    display_name: string | null
    subfield: WorkRef
    field: WorkRef
    domain: WorkRef
  }
  open_access: {
    is_oa: boolean
    oa_status: string | null
    landing_page_url: string | null
    pdf_url: string | null
  }
}
export type TopCitedResponse = {
  source: Source
  level: Level
  id: string | null
  label: string | null
  country: string | null
  from: string | null
  to: string | null
  limit: number
  items: TopCitedItem[]
  notes: string | null
}

export type Author = { name: string; openalex_id: string | null; affiliations: string[] }
export type Article = {
  openalex_id: string
  doi: string | null
  title: string
  abstract: string | null
  publication_year: number | null
  publication_date: string | null
  language: string | null
  type: string | null
  cited_by_count: number
  referenced_works_count: number
  authors: Author[]
  venue: { display_name: string | null; type: string | null }
  primary_topic: {
    id: string | null
    display_name: string | null
    subfield: WorkRef
    field: WorkRef
    domain: WorkRef
  }
  open_access: {
    is_oa: boolean
    oa_status: string | null
    landing_page_url: string | null
    pdf_url: string | null
  }
  ids: { doi: string | null; arxiv: string | null; openalex: string | null }
  keywords: string[]
  concepts: { display_name: string; level: number | null }[]
  counts_by_year: { year: number; cited_by_count: number }[]
}
export type OAStatus = {
  openalex_id: string
  is_oa: boolean
  oa_status: string | null
  sources: { kind: string; url: string; label: string }[]
}
export type SummaryResponse = {
  summary_md: string
  source: 'pdf' | 'abstract' | 'none'
  pdf_kind: string | null
  pdf_url: string | null
  oa_status: string | null
}

export type AuthorRef = {
  openalex_id: string
  display_name: string
}
export type AuthorProfile = {
  openalex_id: string
  orcid: string | null
  display_name: string
  alternatives: string[]
  works_count: number
  cited_by_count: number
  h_index: number | null
  i10_index: number | null
  mean_citedness: number | null
  last_known_institutions: {
    id: string | null
    display_name: string
    country_code: string | null
    type: string | null
  }[]
  counts_by_year: { year: number; works_count: number; cited_by_count: number }[]
  topics: { id: string | null; display_name: string; count: number | null }[]
}
export type AuthorWork = {
  openalex_id: string
  doi: string | null
  title: string
  publication_year: number | null
  publication_date: string | null
  language: string | null
  cited_by_count: number
  authors: string[]
  primary_topic: {
    id: string | null
    display_name: string | null
    field: WorkRef
  }
  open_access: {
    is_oa: boolean
    oa_status: string | null
    landing_page_url: string | null
    pdf_url: string | null
  }
}
export type AuthorWorksResponse = {
  author_id: string
  sort: string
  limit: number
  items: AuthorWork[]
}

export type GraphNode = {
  id: string
  title: string
  year: number | null
  cited_by_count: number
  depth: number
}
export type GraphEdge = {
  source: string
  target: string
  kind: 'refs' | 'cites'
}
export type CitationGraph = {
  root_id: string
  depth: number
  fanout: number
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export type CentroidPaper = {
  doc_id: string | null
  openalex_id: string | null
  title: string | null
  doi: string | null
  primary_topic: string | null
  primary_subfield: string | null
  primary_field: string | null
  similarity: number | null
  cluster_size: number | null
}
export type BertrendTopic = {
  topic_id: number
  signal: 'noise' | 'weak' | 'strong' | 'emerging' | 'dying'
  total_docs: number
  peak_count: number
  first_window: string
  last_window: string
  words: string[]
  history: { period: string; count: number }[]
  centroid: CentroidPaper
}
export type BertrendResponse = {
  available: boolean
  reason?: string
  from_window?: string
  to_window?: string
  windows?: { period: string; n_docs: number; n_topics: number; n_outliers: number }[]
  signal_counts?: Record<string, number>
  emerging?: BertrendTopic[]
  strong?: BertrendTopic[]
}

export type CoverageResponse = {
  from: string
  to: string
  country: string | null
  lang: string
  openalex_count: number
  openaire_count: number
  openaire_supported: boolean
  openaire_error: string | null
}

export type SearchItem = {
  openalex_id: string
  title: string
  abstract_snippet: string | null
  publication_year: number | null
  language: string | null
  cited_by_count: number
  authors: string[]
  primary_topic: {
    id: string | null
    display_name: string | null
    subfield: WorkRef
    field: WorkRef
    domain: WorkRef
  }
  open_access: {
    is_oa: boolean
    oa_status: string | null
    landing_page_url: string | null
    pdf_url: string | null
  }
  relevance_score: number | null
}
export type SearchResponse = {
  source: Source
  query: string
  country: string | null
  total: number
  pipeline: string
  notes: string
  items: SearchItem[]
}

export type ByFieldItem = { id: string; display_name: string; count: number }
export type ByFieldResponse = {
  source: Source
  country: string | null
  from: string | null
  to: string | null
  domain_id: string | null
  items: ByFieldItem[]
  supported: boolean
  notes: string | null
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) {
    let detail = ''
    try {
      const j = await r.json()
      detail = (j as { detail?: string }).detail ?? ''
    } catch {
      // ignore
    }
    throw new Error(detail || `${r.status} ${r.statusText} ${path}`)
  }
  return r.json() as Promise<T>
}

async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!r.ok) {
    let detail = ''
    try {
      const j = await r.json()
      detail = (j as { detail?: string }).detail ?? ''
    } catch {
      // ignore
    }
    const err = new Error(detail || `${r.status} ${r.statusText} ${path}`)
    ;(err as Error & { status?: number }).status = r.status
    throw err
  }
  return r.json() as Promise<T>
}

function qs(params: Record<string, string | number | undefined | null>): string {
  const out = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === '') continue
    out.set(k, String(v))
  }
  const s = out.toString()
  return s ? `?${s}` : ''
}

export const api = {
  health: () => get<{ status: string }>('/healthz'),
  taxonomy: () => get<Taxonomy>('/api/taxonomy'),
  trends: (p: {
    level: Level
    id?: string | null
    from: string
    to: string
    granularity: Granularity
    country?: string | null
    lang?: string
    source?: Source
  }) => get<TrendsResponse>(`/api/dashboard/trends${qs(p)}`),
  topCited: (p: {
    level: Level
    id?: string | null
    from?: string | null
    to?: string | null
    limit?: number
    country?: string | null
    lang?: string
    source?: Source
  }) => get<TopCitedResponse>(`/api/dashboard/top-cited${qs(p)}`),
  byField: (p: {
    from?: string | null
    to?: string | null
    domain_id?: string | null
    limit?: number
    country?: string | null
    lang?: string
    source?: Source
  }) => get<ByFieldResponse>(`/api/dashboard/by-field${qs(p)}`),
  coverage: (p: {
    from: string
    to: string
    country?: string | null
  }) => get<CoverageResponse>(`/api/dashboard/coverage${qs(p)}`),
  bertrend: () => get<BertrendResponse>(`/api/dashboard/bertrend`),
  search: (p: {
    q: string
    top_k?: number
    country?: string | null
    source?: Source
  }) => get<SearchResponse>(`/api/search${qs(p)}`),
  article: (id: string) => get<Article>(`/api/articles/${encodeURIComponent(id)}`),
  oaStatus: (id: string) => get<OAStatus>(`/api/articles/${encodeURIComponent(id)}/oa-status`),
  summary: (id: string) => postJson<SummaryResponse>(`/api/articles/${encodeURIComponent(id)}/summary`),
  citationGraph: (id: string, p: { depth?: number; fanout?: number } = {}) =>
    get<CitationGraph>(`/api/articles/${encodeURIComponent(id)}/citation-graph${qs(p)}`),
  author: (id: string) => get<AuthorProfile>(`/api/authors/${encodeURIComponent(id)}`),
  authorWorks: (id: string, p: { sort?: string; limit?: number } = {}) =>
    get<AuthorWorksResponse>(`/api/authors/${encodeURIComponent(id)}/works${qs(p)}`),
  interpretTrends: (
    body: {
      trends: TrendsResponse | undefined
      top: TopCitedResponse | undefined
      by_field: ByFieldResponse | undefined
      bertrend?: BertrendResponse | undefined
    },
    handlers: {
      onChunk: (text: string) => void
      onError: (msg: string) => void
      onDone: () => void
      signal?: AbortSignal
    },
  ) => streamSSE('/api/agent/interpret-trends', body, handlers),
}

async function streamSSE(
  path: string,
  body: unknown,
  handlers: {
    onChunk: (text: string) => void
    onError: (msg: string) => void
    onDone: () => void
    signal?: AbortSignal
  },
): Promise<void> {
  let r: Response
  try {
    r = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(body),
      signal: handlers.signal,
    })
  } catch (e) {
    handlers.onError((e as Error).message)
    handlers.onDone()
    return
  }
  if (!r.ok || !r.body) {
    let detail = ''
    try {
      const j = await r.json()
      detail = (j as { detail?: string }).detail ?? ''
    } catch {
      // ignore
    }
    handlers.onError(detail || `${r.status} ${r.statusText}`)
    handlers.onDone()
    return
  }

  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let done = false
  let event = 'message'

  while (!done) {
    const { value, done: streamDone } = await reader.read()
    done = streamDone
    if (value) buffer += decoder.decode(value, { stream: true })

    let idx
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      event = 'message'
      let dataLine = ''
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLine = line.slice(5).trim()
      }
      if (!dataLine) continue
      try {
        const parsed = JSON.parse(dataLine)
        if (event === 'error' && parsed.error) handlers.onError(parsed.error)
        else if (event === 'done') {
          handlers.onDone()
          return
        } else if (parsed.chunk) handlers.onChunk(parsed.chunk)
      } catch {
        // ignore malformed line
      }
    }
  }
  handlers.onDone()
}
