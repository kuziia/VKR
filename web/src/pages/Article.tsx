import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api, type Article, type SummaryResponse } from '../lib/api'
import Masthead from '../components/Masthead'
import NavStrip from '../components/NavStrip'

export default function ArticlePage() {
  const { id = '' } = useParams<{ id: string }>()
  const article = useQuery({
    queryKey: ['article', id],
    queryFn: () => api.article(id),
    enabled: !!id,
  })
  const oa = useQuery({
    queryKey: ['oa-status', id],
    queryFn: () => api.oaStatus(id),
    enabled: !!id,
  })

  const [summaryError, setSummaryError] = useState<string | null>(null)
  const summary = useMutation<SummaryResponse, Error>({
    mutationFn: () => api.summary(id),
    onError: (e) => setSummaryError(e.message),
    onSuccess: () => setSummaryError(null),
  })

  return (
    <div className="max-w-[1340px] mx-auto px-9 pt-6 pb-16">
      <Masthead />
      <NavStrip />

      <div className="text-[11px] italic font-serif text-text-muted mb-4">
        <Link to="/" className="hover:text-accent">
          ← Главная
        </Link>
        {' · '}
        <span className="font-mono not-italic">{id}</span>
      </div>

      {article.isLoading && (
        <div className="py-16 text-center text-text-dim italic font-serif">
          загрузка статьи…
        </div>
      )}
      {article.isError && (
        <div className="py-12 text-loss italic font-serif">
          Не удалось загрузить статью: {(article.error as Error).message}
        </div>
      )}
      {article.data && (
        <ArticleBody
          art={article.data}
          oaSources={oa.data?.sources ?? []}
          summaryState={summary}
          summaryError={summaryError}
          onSummary={() => summary.mutate()}
        />
      )}
    </div>
  )
}

type SummaryState = ReturnType<typeof useMutation<SummaryResponse, Error>>

function ArticleBody({
  art,
  oaSources,
  summaryState,
  summaryError,
  onSummary,
}: {
  art: Article
  oaSources: { kind: string; url: string; label: string }[]
  summaryState: SummaryState
  summaryError: string | null
  onSummary: () => void
}) {
  const summary = summaryState.data
  const isLoading = summaryState.isPending

  return (
    <section className="grid grid-cols-[2fr_1fr] gap-10 pb-8 border-b border-border">
      <div>
        <div className="text-[10px] tracking-[0.14em] uppercase text-accent font-bold mb-2.5">
          {art.primary_topic.field.display_name ?? '—'} ·{' '}
          {art.primary_topic.display_name ?? art.type ?? '—'}
        </div>
        <h1 className="font-serif text-[44px] font-semibold tracking-[-0.018em] leading-[1.05] mb-4">
          {art.title}
        </h1>

        {art.authors.length > 0 && (
          <div className="font-serif italic text-[16px] text-text-muted mb-3 leading-snug">
            {art.authors.map((a, i) => (
              <span key={`${a.openalex_id ?? a.name}-${i}`}>
                {i > 0 && ', '}
                {a.openalex_id ? (
                  <Link
                    to={`/author/${a.openalex_id}`}
                    className="text-text-muted hover:text-accent"
                  >
                    {a.name}
                  </Link>
                ) : (
                  a.name
                )}
              </span>
            ))}
          </div>
        )}

        <div className="text-[12px] font-mono text-text-muted mb-5 flex flex-wrap gap-x-4 gap-y-1">
          <span>{art.publication_date ?? art.publication_year ?? '—'}</span>
          {art.venue.display_name && (
            <span className="not-italic">{art.venue.display_name}</span>
          )}
          {art.language && (
            <span className="uppercase">lang: {art.language}</span>
          )}
          {art.doi && (
            <a
              href={art.doi.startsWith('http') ? art.doi : `https://doi.org/${art.doi}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent hover:underline"
            >
              {art.doi.replace(/^https?:\/\/(dx\.)?doi\.org\//, '')}
            </a>
          )}
        </div>

        <div className="flex flex-wrap gap-2 mb-7">
          {oaSources.map((s) => (
            <a
              key={`${s.kind}:${s.url}`}
              href={s.url}
              target="_blank"
              rel="noopener noreferrer"
              className="px-3 py-1.5 text-[11px] font-mono font-semibold border border-border bg-surface text-text hover:border-accent hover:text-accent uppercase tracking-wide"
            >
              {s.label}
            </a>
          ))}
          <button
            type="button"
            onClick={onSummary}
            disabled={isLoading}
            className="px-3 py-1.5 text-[11px] font-mono font-semibold border border-accent bg-accent text-white hover:bg-text hover:border-text uppercase tracking-wide disabled:opacity-50"
          >
            {isLoading ? 'генерация…' : summary ? 'обновить саммари' : 'саммари ии'}
          </button>
          <Link
            to={`/article/${art.openalex_id}/graph`}
            className="px-3 py-1.5 text-[11px] font-mono font-semibold border border-text bg-surface text-text hover:bg-text hover:text-white uppercase tracking-wide"
          >
            граф цитирования
          </Link>
        </div>

        {art.abstract && (
          <>
            <h2 className="font-serif text-[18px] font-bold mb-2 pb-1 border-b border-border">
              Abstract
            </h2>
            <p className="font-serif text-[16px] leading-relaxed mb-7 text-text">
              {art.abstract}
            </p>
          </>
        )}

        <h2 className="font-serif text-[18px] font-bold mb-2 pb-1 border-b border-border">
          Саммари ИИ-агента
        </h2>
        {summaryError && (
          <div className="font-serif italic text-loss text-[14px] mb-3">
            Не удалось получить саммари: {summaryError}
          </div>
        )}
        {!summary && !summaryError && !isLoading && (
          <div className="font-serif italic text-text-muted text-[14px]">
            Нажмите «Саммари ИИ» — отправим статью в Claude. Если PDF доступен,
            будет суммаризация полного текста; иначе — по аннотации.
          </div>
        )}
        {isLoading && (
          <div className="font-serif italic text-text-muted text-[14px]">
            Генерируем саммари… Это может занять до минуты.
          </div>
        )}
        {summary && (
          <div>
            <div className="text-[10px] uppercase tracking-[0.08em] text-text-muted font-bold mb-2">
              источник: {summary.source}
              {summary.pdf_kind ? ` · ${summary.pdf_kind}` : ''}
            </div>
            <div className="prose prose-sm max-w-none font-serif text-[15px] leading-relaxed [&>ul]:list-disc [&>ul]:pl-6 [&>ol]:list-decimal [&>ol]:pl-6 [&_strong]:font-semibold [&>h1]:font-serif [&>h2]:font-serif [&>h3]:font-serif [&>h2]:mt-4 [&>h3]:mt-3 [&>p]:mb-3 [&>ul]:mb-3">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {summary.summary_md}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </div>

      <aside className="border-l border-border pl-8 space-y-6 self-start">
        <SidebarBlock label="Цитирований" value={art.cited_by_count.toLocaleString('ru-RU')} />
        <SidebarBlock
          label="Ссылается на"
          value={
            art.referenced_works_count
              ? art.referenced_works_count.toLocaleString('ru-RU')
              : '—'
          }
        />
        <SidebarBlock
          label="Open Access"
          value={art.open_access.is_oa ? (art.open_access.oa_status ?? 'oa') : '—'}
          accent={art.open_access.is_oa ? 'profit' : 'dim'}
        />

        <div>
          <div className="text-[10px] uppercase tracking-[0.08em] font-bold text-text-muted mb-2">
            Таксономия
          </div>
          <ul className="font-serif text-[13px] space-y-1.5">
            {art.primary_topic.domain.display_name && (
              <li>
                <span className="text-[10px] font-mono uppercase text-text-dim mr-1.5">
                  domain
                </span>
                {art.primary_topic.domain.display_name}
              </li>
            )}
            {art.primary_topic.field.display_name && (
              <li>
                <span className="text-[10px] font-mono uppercase text-text-dim mr-1.5">
                  field
                </span>
                {art.primary_topic.field.display_name}
              </li>
            )}
            {art.primary_topic.subfield.display_name && (
              <li>
                <span className="text-[10px] font-mono uppercase text-text-dim mr-1.5">
                  subf
                </span>
                {art.primary_topic.subfield.display_name}
              </li>
            )}
            {art.primary_topic.display_name && (
              <li>
                <span className="text-[10px] font-mono uppercase text-text-dim mr-1.5">
                  topic
                </span>
                {art.primary_topic.display_name}
              </li>
            )}
          </ul>
        </div>

        {art.keywords.length > 0 && (
          <div>
            <div className="text-[10px] uppercase tracking-[0.08em] font-bold text-text-muted mb-2">
              Ключевые слова
            </div>
            <div className="flex flex-wrap gap-1.5">
              {art.keywords.slice(0, 10).map((k) => (
                <span
                  key={k}
                  className="px-2 py-0.5 text-[11px] font-mono border border-border bg-surface text-text-muted"
                >
                  {k}
                </span>
              ))}
            </div>
          </div>
        )}

        {art.authors.length > 0 && (
          <div>
            <div className="text-[10px] uppercase tracking-[0.08em] font-bold text-text-muted mb-2">
              Авторы ({art.authors.length})
            </div>
            <ul className="font-serif text-[13px] space-y-1.5">
              {art.authors.slice(0, 12).map((a, i) => (
                <li key={`${a.openalex_id ?? a.name}-${i}`}>
                  {a.openalex_id ? (
                    <Link
                      to={`/author/${a.openalex_id}`}
                      className="font-semibold hover:text-accent"
                    >
                      {a.name}
                    </Link>
                  ) : (
                    <span className="font-semibold">{a.name}</span>
                  )}
                  {a.affiliations.length > 0 && (
                    <span className="text-text-muted italic">
                      {' '}
                      — {a.affiliations[0]}
                    </span>
                  )}
                </li>
              ))}
              {art.authors.length > 12 && (
                <li className="text-text-dim italic">
                  и ещё {art.authors.length - 12}
                </li>
              )}
            </ul>
          </div>
        )}
      </aside>
    </section>
  )
}

function SidebarBlock({
  label,
  value,
  accent = 'text',
}: {
  label: string
  value: string
  accent?: 'text' | 'profit' | 'dim'
}) {
  const color =
    accent === 'profit'
      ? 'text-profit'
      : accent === 'dim'
        ? 'text-text-dim'
        : 'text-text'
  return (
    <div className="border-t border-border pt-1.5">
      <div className="text-[10px] uppercase tracking-[0.08em] font-bold text-text-muted mb-0.5">
        {label}
      </div>
      <div className={`font-mono font-semibold text-base tnum ${color}`}>
        {value}
      </div>
    </div>
  )
}
