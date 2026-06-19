import type { Source, TrendsResponse } from '../lib/api'

type Props = {
  trends: TrendsResponse | undefined
  trendsLoading: boolean
  source: Source
}

export default function Hero({ trends, trendsLoading, source }: Props) {
  const total = trends?.total
  const label = trends?.label ?? '— все домены —'
  const granularity = trends?.granularity ?? '—'

  return (
    <section className="border-b border-border pb-6 mb-5">
      <div className="text-[10px] tracking-[0.14em] uppercase text-accent font-bold mb-2">
        Динамика публикаций · {source === 'openaire' ? 'OpenAIRE' : 'OpenAlex'} live
      </div>
      <div className="grid grid-cols-[auto_1fr_auto] gap-8 items-end">
        <h1 className="font-serif text-[64px] font-semibold tracking-[-0.02em] leading-none tnum">
          {trendsLoading || total === undefined ? '—' : total.toLocaleString('ru-RU')}
        </h1>
        <p className="font-serif text-[16px] italic text-text-muted leading-snug max-w-[560px] pb-1.5">
          Узел: <strong className="text-text not-italic">{label}</strong>. Период:{' '}
          <span className="font-mono not-italic text-text">
            {trends?.from ?? '—'} → {trends?.to ?? '—'}
          </span>
          . Гранулярность{' '}
          <span className="font-mono not-italic text-text">{granularity}</span>
          {trends?.country && (
            <>
              , страна{' '}
              <span className="font-mono not-italic text-text uppercase">{trends.country}</span>
            </>
          )}
          {trends?.lang && trends.lang !== 'all' && (
            <>
              , язык{' '}
              <span className="font-mono not-italic text-text uppercase">{trends.lang}</span>
            </>
          )}
          .
        </p>
        <div className="border-l border-border pl-5 pb-1">
          <div className="text-[10px] uppercase tracking-[0.08em] font-semibold text-text-muted mb-0.5">
            точек
          </div>
          <div className="font-mono font-semibold text-[18px] tnum">
            {trends?.points.length ?? '—'}
          </div>
        </div>
      </div>
    </section>
  )
}
