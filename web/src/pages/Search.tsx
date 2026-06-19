import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api, type SearchItem, type Source } from '../lib/api'
import Masthead from '../components/Masthead'
import NavStrip from '../components/NavStrip'
import SearchBox from '../components/SearchBox'
import CountryPicker from '../components/CountryPicker'
import SourcePicker from '../components/SourcePicker'

export default function SearchPage() {
  const [params, setParams] = useSearchParams()
  const q = (params.get('q') || '').trim()
  const country = params.get('country') || 'all'
  const source = ((params.get('source') as Source) || 'openalex')

  const [topK] = useState(20)

  const enabled = q.length > 0
  const search = useQuery({
    queryKey: ['search', source, q, country, topK],
    queryFn: () =>
      api.search({
        q,
        top_k: topK,
        country: country === 'all' ? null : country,
        source,
      }),
    enabled,
    staleTime: 5 * 60_000,
  })

  const onCountry = (next: string) => {
    const p = new URLSearchParams(params)
    if (next === 'all') p.delete('country')
    else p.set('country', next)
    setParams(p, { replace: true })
  }
  const onSource = (next: Source) => {
    const p = new URLSearchParams(params)
    p.set('source', next)
    setParams(p, { replace: true })
  }

  const totalLabel = useMemo(() => {
    if (!search.data) return null
    const t = search.data.total
    return t.toLocaleString('ru-RU')
  }, [search.data])

  return (
    <div className="max-w-[1340px] mx-auto px-9 pt-6 pb-16">
      <Masthead />
      <NavStrip />

      <section className="grid grid-cols-[2fr_1fr] gap-10 pb-8 border-b border-border mb-8">
        <div>
          <div className="text-[10px] tracking-[0.14em] uppercase text-accent font-bold mb-2.5">
            Поиск · {source === 'openaire' ? 'OpenAIRE' : 'OpenAlex'} live
          </div>
          <h1 className="font-serif text-[44px] font-semibold tracking-[-0.02em] leading-[1.05] mb-4 break-words">
            {q ? <>«{q}»</> : 'Поиск'}
          </h1>

          <div className="mb-3">
            <SearchBox initial={q} />
          </div>

          {q && (
            <p className="font-serif italic text-text-muted text-[14px]">
              {search.isLoading
                ? 'ищем…'
                : search.isError
                  ? 'ошибка запроса'
                  : totalLabel
                    ? `Найдено ${totalLabel} работ; показано топ-${search.data?.items.length ?? 0}.`
                    : '—'}
            </p>
          )}
        </div>

        <aside className="border-l border-border pl-8 space-y-5 self-start">
          <div>
            <h3 className="font-serif text-[13px] font-bold uppercase tracking-[0.06em] mb-3 text-accent">
              Источник
            </h3>
            <SourcePicker value={source} onChange={onSource} />
          </div>
          <div>
            <h3 className="font-serif text-[13px] font-bold uppercase tracking-[0.06em] mb-3 text-accent">
              Страна авторов
            </h3>
            <CountryPicker value={country} onChange={onCountry} />
          </div>
          <div>
            <h3 className="font-serif text-[13px] font-bold uppercase tracking-[0.06em] mb-2 text-accent">
              О ранжировании
            </h3>
            <PipelineNote
              pipeline={search.data?.pipeline}
              notes={search.data?.notes}
            />
          </div>
        </aside>
      </section>

      {!q ? (
        <div className="text-text-dim italic font-serif">
          Введите запрос, чтобы получить результаты.
        </div>
      ) : (
        <ResultsBlock
          items={search.data?.items}
          isLoading={search.isLoading || search.isFetching}
          isError={search.isError}
        />
      )}

      <footer className="text-[11px] italic text-text-muted font-serif text-center mt-10 pt-4 border-t border-border">
        Данные: OpenAlex + OpenAIRE (live, кэш 1ч).
      </footer>
    </div>
  )
}

function PipelineNote({
  pipeline,
  notes,
}: {
  pipeline?: string
  notes?: string
}) {
  if (!pipeline) {
    return (
      <p className="text-[12px] italic font-serif text-text-muted leading-snug">
        Введи запрос — справа появится описание ранжирования и параметры
        выбранного источника.
      </p>
    )
  }
  return (
    <div className="text-[12px] font-serif text-text-muted leading-snug">
      <div className="font-mono text-[10px] uppercase tracking-[0.06em] text-text-dim mb-1.5">
        pipeline: {pipeline}
      </div>
      <p className="italic">{notes}</p>
    </div>
  )
}

function ResultsBlock({
  items,
  isLoading,
  isError,
}: {
  items: SearchItem[] | undefined
  isLoading: boolean
  isError: boolean
}) {
  if (isError) {
    return (
      <div className="text-loss italic font-serif py-8">Ошибка поиска.</div>
    )
  }
  if (isLoading) {
    return (
      <div className="grid place-items-center text-text-dim italic font-serif py-12 border border-dashed border-border">
        загрузка…
      </div>
    )
  }
  if (!items || items.length === 0) {
    return (
      <div className="text-text-dim italic font-serif py-8">
        Ничего не найдено. Попробуй другую формулировку или смени язык.
      </div>
    )
  }
  return (
    <table className="w-full text-[13px]">
      <thead>
        <tr className="border-b border-text">
          <th className="text-left py-2 px-2.5 text-[9px] uppercase tracking-[0.1em] text-text-muted font-bold">
            Title / Authors
          </th>
          <th className="text-right py-2 px-2.5 text-[9px] uppercase tracking-[0.1em] text-text-muted font-bold">
            Year
          </th>
          <th className="text-right py-2 px-2.5 text-[9px] uppercase tracking-[0.1em] text-text-muted font-bold">
            Cit.
          </th>
          <th className="text-right py-2 px-2.5 text-[9px] uppercase tracking-[0.1em] text-text-muted font-bold">
            Score
          </th>
          <th className="text-center py-2 px-2.5 text-[9px] uppercase tracking-[0.1em] text-text-muted font-bold">
            OA / Lang
          </th>
        </tr>
      </thead>
      <tbody>
        {items.map((it) => {
          const isLinkable = !!it.openalex_id && !it.openalex_id.startsWith('oaire:')
          return (
          <tr key={it.openalex_id} className="dotted-rule hover:bg-surface-2 align-top">
            <td className="py-3 px-2.5">
              {isLinkable ? (
                <Link
                  to={`/article/${it.openalex_id}`}
                  className="font-serif font-semibold leading-snug hover:text-accent block"
                >
                  {it.title}
                </Link>
              ) : (
                <span className="font-serif font-semibold leading-snug block">
                  {it.title}
                </span>
              )}
              {it.authors.length > 0 && (
                <div className="text-[11px] italic text-text-muted mt-0.5">
                  {it.authors.slice(0, 4).join(', ')}
                  {it.authors.length > 4 && ` +${it.authors.length - 4}`}
                </div>
              )}
              {it.abstract_snippet && (
                <div className="text-[12px] text-text-muted mt-1.5 leading-snug max-w-[640px]">
                  {it.abstract_snippet}
                </div>
              )}
              {it.primary_topic.field.display_name && (
                <div className="text-[10px] uppercase tracking-[0.06em] text-text-dim mt-1.5">
                  {it.primary_topic.field.display_name}
                  {it.primary_topic.display_name && ` · ${it.primary_topic.display_name}`}
                </div>
              )}
            </td>
            <td className="py-3 px-2.5 text-right font-mono">
              {it.publication_year ?? '—'}
            </td>
            <td className="py-3 px-2.5 text-right font-mono font-semibold">
              {it.cited_by_count.toLocaleString('ru-RU')}
            </td>
            <td className="py-3 px-2.5 text-right font-mono text-text-muted">
              {it.relevance_score !== null
                ? it.relevance_score.toFixed(1)
                : '—'}
            </td>
            <td className="py-3 px-2.5 text-center">
              <div className="flex flex-col items-center gap-0.5">
                {it.open_access.is_oa && (
                  <span className="inline-block px-1.5 py-px text-[9px] font-bold uppercase tracking-[0.06em] text-profit border border-profit">
                    {it.open_access.oa_status ?? 'oa'}
                  </span>
                )}
                {it.language && (
                  <span className="font-mono text-[10px] uppercase text-text-dim">
                    {it.language}
                  </span>
                )}
              </div>
            </td>
          </tr>
          )
        })}
      </tbody>
    </table>
  )
}
