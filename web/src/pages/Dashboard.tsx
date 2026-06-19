import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type Granularity, type Source } from '../lib/api'
import { periodFor, type PresetKey } from '../lib/period'
import { selectionToValue } from '../components/CascadeFilter'
import Masthead from '../components/Masthead'
import NavStrip from '../components/NavStrip'
import Hero from '../components/Hero'
import FilterBar from '../components/FilterBar'
import MetricsRow from '../components/MetricsRow'
import LineChart from '../components/LineChart'
import TopCitedTable from '../components/TopCitedTable'
import SymBars from '../components/SymBars'
import InterpretBlock from '../components/InterpretBlock'
import CoverageBlock from '../components/CoverageBlock'
import BertrendBlock from '../components/BertrendBlock'

export default function Dashboard() {
  const [preset, setPreset] = useState<PresetKey>('1Y')
  const [granOverride, setGranOverride] = useState<Granularity | null>(null)
  const [country, setCountry] = useState<string>('all')
  const [lang, setLang] = useState<string>('all')
  const [source, setSource] = useState<Source>('openalex')
  const [domain, setDomainSel] = useState<string | null>(null)
  const [field, setFieldSel] = useState<string | null>(null)
  const [subfield, setSubfieldSel] = useState<string | null>(null)
  const [topic, setTopicSel] = useState<string | null>(null)

  const period = useMemo(() => periodFor(preset), [preset])

  const sel = useMemo(() => {
    if (source === 'openaire') return { level: 'all' as const, id: null }
    return selectionToValue({ domain, field, subfield, topic })
  }, [source, domain, field, subfield, topic])

  const tax = useQuery({ queryKey: ['taxonomy'], queryFn: api.taxonomy })

  const countryParam = country === 'all' ? null : country
  const langParam = source === 'openaire' ? 'all' : lang
  // OpenAIRE supports only year-level. Otherwise: user override → preset default.
  const granularity: Granularity =
    source === 'openaire' ? 'year' : (granOverride ?? period.granularity)

  const trends = useQuery({
    queryKey: ['trends', source, sel.level, sel.id, period.from, period.to, granularity, country, langParam],
    queryFn: () =>
      api.trends({
        level: sel.level,
        id: sel.id,
        from: period.from,
        to: period.to,
        granularity,
        country: countryParam,
        lang: langParam,
        source,
      }),
  })

  const top = useQuery({
    queryKey: ['top', source, sel.level, sel.id, period.from, period.to, country, langParam],
    queryFn: () =>
      api.topCited({
        level: sel.level,
        id: sel.id,
        from: period.from,
        to: period.to,
        limit: 10,
        country: countryParam,
        lang: langParam,
        source,
      }),
  })

  const symEnabled =
    source === 'openalex' && (sel.level === 'all' || sel.level === 'domain')
  const byField = useQuery({
    enabled: symEnabled,
    queryKey: ['by-field', source, period.from, period.to, sel.level === 'domain' ? sel.id : null, country, langParam],
    queryFn: () =>
      api.byField({
        from: period.from,
        to: period.to,
        domain_id: sel.level === 'domain' ? sel.id : null,
        limit: 8,
        country: countryParam,
        lang: langParam,
        source,
      }),
  })

  const bertrend = useQuery({
    queryKey: ['bertrend'],
    queryFn: api.bertrend,
    staleTime: 60 * 60_000,
  })

  const allFailed =
    !trends.isLoading && !top.isLoading && trends.isError && top.isError

  return (
    <div className="max-w-[1340px] mx-auto px-9 pt-6 pb-16">
      <Masthead />
      <NavStrip />

      {allFailed && (
        <div className="border border-loss bg-surface px-3 py-2 mb-5 font-serif text-[13px] text-loss">
          <strong className="not-italic">Бэкенд не отвечает.</strong>{' '}
          <span className="italic text-text-muted">
            Проверь, поднят ли uvicorn на 127.0.0.1:8088.
          </span>
        </div>
      )}

      <Hero
        trends={trends.data}
        trendsLoading={trends.isLoading || trends.isFetching}
        source={source}
      />

      <FilterBar
        source={source}
        onSource={setSource}
        taxonomy={tax.data}
        taxLoading={tax.isLoading}
        taxError={tax.isError}
        domain={domain}
        field={field}
        subfield={subfield}
        topic={topic}
        onCascade={(n) => {
          setDomainSel(n.domain)
          setFieldSel(n.field)
          setSubfieldSel(n.subfield)
          setTopicSel(n.topic)
        }}
        preset={preset}
        onPreset={(p) => {
          setPreset(p)
          setGranOverride(null)  // reset to preset default
        }}
        granularity={granularity}
        onGranularity={(g) => setGranOverride(g)}
        granularityLocked={source === 'openaire'}
        country={country}
        onCountry={setCountry}
        lang={lang}
        onLang={setLang}
      />

      <MetricsRow
        points={trends.data?.points}
        granularity={granularity}
        isLoading={trends.isLoading || trends.isFetching}
      />

      <div className="mb-7">
        <CoverageBlock from={period.from} to={period.to} country={country} />
      </div>

      <section className="grid grid-cols-[1.45fr_1fr] gap-8 mb-10 items-stretch">
        <div className="flex flex-col gap-7">
          <div>
            <div className="flex items-baseline justify-between pb-2 mb-3 border-b-2 border-text">
              <h2 className="font-serif text-[22px] font-semibold tracking-[-0.015em]">
                График динамики
              </h2>
              <div className="text-[11px] italic text-text-muted font-serif">
                {trends.isFetching
                  ? 'обновление…'
                  : trends.isError
                    ? <span className="text-loss">ошибка</span>
                    : `${trends.data?.from ?? '—'} → ${trends.data?.to ?? '—'}`}
              </div>
            </div>
            {trends.isLoading ? (
              <div className="h-[340px] grid place-items-center text-text-dim italic font-serif text-sm border border-dashed border-border">
                загрузка…
              </div>
            ) : (
              <LineChart points={trends.data?.points ?? []} height={340} />
            )}
            {trends.data?.notes && (
              <div className="mt-2 text-[11px] italic font-serif text-text-muted leading-snug">
                {trends.data.notes}
              </div>
            )}
          </div>

          <div>
            <div className="flex items-baseline justify-between pb-2 mb-3 border-b-2 border-text">
              <h2 className="font-serif text-[22px] font-semibold tracking-[-0.015em]">
                По полям
              </h2>
              <div className="text-[11px] italic text-text-muted font-serif">
                {source === 'openaire'
                  ? 'недоступно для OpenAIRE'
                  : symEnabled
                    ? sel.level === 'domain'
                      ? 'распределение в выбранном домене · top-8'
                      : 'все домены · top-8'
                    : 'недоступно для уровня ниже домена'}
              </div>
            </div>
            {source === 'openaire' ? (
              <div className="text-text-dim italic font-serif text-[13px]">
                OpenAIRE не предоставляет агрегацию по полям. Переключи источник на OpenAlex.
              </div>
            ) : symEnabled ? (
              <SymBars items={byField.data?.items} isLoading={byField.isLoading || byField.isFetching} />
            ) : (
              <div className="text-text-dim italic font-serif text-[13px]">
                Выбери уровень <span className="font-mono">domain</span> или ниже для распределения по полям.
              </div>
            )}
          </div>
        </div>

        <div className="flex flex-col min-h-0">
          <div className="flex items-baseline justify-between pb-2 mb-3 border-b-2 border-text">
            <h2 className="font-serif text-[22px] font-semibold tracking-[-0.015em]">
              Популярные статьи
            </h2>
            <div className="text-[11px] italic text-text-muted font-serif">
              top-{top.data?.items?.length ?? 0}
            </div>
          </div>
          <div className="list-scroll overflow-auto pr-1" style={{ maxHeight: 880 }}>
            <TopCitedTable
              items={top.data?.items}
              isLoading={top.isLoading || top.isFetching}
              isError={top.isError}
            />
          </div>
          {top.data?.notes && (
            <div className="mt-2 text-[11px] italic font-serif text-text-muted leading-snug">
              {top.data.notes}
            </div>
          )}
        </div>
      </section>

      <section className="mb-10">
        <BertrendBlock data={bertrend.data} isLoading={bertrend.isLoading} />
      </section>

      {/* BERTrend намеренно НЕ передаём в interpret: его корпус — отдельный
          русскоязычный срез, не связанный с фильтрами дашборда (OpenAlex
          country/lang/taxonomy). Передача путала агента — он пытался
          объяснить несвязанные emerging-темы. См. BertrendBlock выше как
          самостоятельный сигнал. */}
      <InterpretBlock
        trends={trends.data}
        top={top.data}
        byField={byField.data}
      />

      <footer className="text-[11px] italic text-text-muted font-serif text-center mt-8 pt-4 border-t border-border">
        Данные: OpenAlex + OpenAIRE (live, кэш 24ч). LLM-комментарии —
        Claude Haiku 4.5.
      </footer>
    </div>
  )
}
