import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { BertrendResponse, BertrendTopic, CentroidPaper } from '../lib/api'

type Props = {
  data: BertrendResponse | undefined
  isLoading: boolean
}

export default function BertrendBlock({ data, isLoading }: Props) {
  const [tab, setTab] = useState<'emerging' | 'strong'>('emerging')

  if (isLoading) {
    return (
      <div className="border border-dashed border-border h-[180px] grid place-items-center text-text-dim italic font-serif">
        BERTrend загружается…
      </div>
    )
  }
  if (!data || !data.available) {
    return (
      <div className="border border-dashed border-border p-4 text-text-dim italic font-serif text-[14px]">
        BERTrend-данные недоступны: {data?.reason ?? 'нет файла bertrend.db'}.
      </div>
    )
  }

  const topics: BertrendTopic[] =
    (tab === 'emerging' ? data.emerging : data.strong) ?? []
  const sigs = data.signal_counts ?? {}

  return (
    <div>
      <div className="flex items-baseline justify-between pb-2 mb-3 border-b-2 border-text">
        <h2 className="font-serif text-[22px] font-semibold tracking-[-0.015em]">
          BERTrend · тематические сигналы
        </h2>
        <div className="text-[11px] italic text-text-muted font-serif">
          {data.from_window} → {data.to_window} · {data.windows?.length ?? 0} окон
        </div>
      </div>

      <div className="flex items-center gap-4 mb-4 flex-wrap">
        <div className="flex gap-1.5">
          {(['emerging', 'strong'] as const).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setTab(s)}
              className={
                'px-2.5 py-1 text-[12px] font-mono font-semibold border tracking-wide uppercase ' +
                (s === tab
                  ? 'bg-accent text-white border-accent'
                  : 'bg-surface text-text-muted border-border hover:text-text')
              }
            >
              {s} · {sigs[s] ?? 0}
            </button>
          ))}
        </div>
        <div className="text-[11px] italic font-serif text-text-muted">
          всего: emerging {sigs.emerging ?? 0} · strong {sigs.strong ?? 0} ·
          weak {sigs.weak ?? 0} · dying {sigs.dying ?? 0} · noise {sigs.noise ?? 0}
        </div>
      </div>

      <table className="w-full text-[13px]">
        <thead>
          <tr className="border-b border-text">
            <th className="text-left py-2 px-2.5 text-[9px] uppercase tracking-[0.1em] text-text-muted font-bold w-[40px]">
              #
            </th>
            <th className="text-left py-2 px-2.5 text-[9px] uppercase tracking-[0.1em] text-text-muted font-bold">
              Top-words
            </th>
            <th className="text-left py-2 px-2.5 text-[9px] uppercase tracking-[0.1em] text-text-muted font-bold w-[320px]">
              OpenAlex (центроид кластера)
            </th>
            <th className="text-right py-2 px-2.5 text-[9px] uppercase tracking-[0.1em] text-text-muted font-bold">
              Docs
            </th>
            <th className="text-right py-2 px-2.5 text-[9px] uppercase tracking-[0.1em] text-text-muted font-bold">
              Peak
            </th>
            <th className="text-left py-2 px-2.5 text-[9px] uppercase tracking-[0.1em] text-text-muted font-bold w-[180px]">
              Sparkline
            </th>
          </tr>
        </thead>
        <tbody>
          {topics.map((t) => (
            <tr key={t.topic_id} className="dotted-rule align-top hover:bg-surface-2">
              <td className="py-2.5 px-2.5 font-mono text-text-dim">
                {t.topic_id}
              </td>
              <td className="py-2.5 px-2.5">
                <div className="font-serif font-semibold leading-snug">
                  {t.words.slice(0, 4).join(' · ')}
                </div>
                {t.words.length > 4 && (
                  <div className="text-[11px] italic text-text-muted mt-0.5">
                    {t.words.slice(4, 8).join(', ')}
                  </div>
                )}
                <div className="text-[10px] uppercase tracking-[0.06em] text-text-dim mt-1">
                  окна {t.first_window} → {t.last_window}
                </div>
              </td>
              <td className="py-2.5 px-2.5">
                <CentroidCell centroid={t.centroid} />
              </td>
              <td className="py-2.5 px-2.5 text-right font-mono font-semibold">
                {t.total_docs}
              </td>
              <td className="py-2.5 px-2.5 text-right font-mono">
                {t.peak_count}
              </td>
              <td className="py-2.5 px-2.5">
                <Sparkline history={t.history} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function CentroidCell({ centroid }: { centroid: CentroidPaper }) {
  if (!centroid || !centroid.primary_topic) {
    return <span className="text-text-dim italic font-serif text-[11px]">—</span>
  }
  const sim = centroid.similarity ?? 0
  return (
    <div className="space-y-1">
      <div className="font-serif font-semibold text-[12.5px] leading-snug">
        {centroid.primary_topic}
      </div>
      <div className="text-[10px] uppercase tracking-[0.06em] text-text-dim leading-snug">
        {centroid.primary_subfield ?? '—'}
        {centroid.primary_field ? ` · ${centroid.primary_field}` : ''}
      </div>
      {centroid.title && centroid.openalex_id && (
        <Link
          to={`/article/${centroid.openalex_id}`}
          className="block font-serif italic text-[11px] text-text-muted hover:text-accent leading-snug"
          title={centroid.title}
        >
          ↗ {centroid.title.length > 60 ? centroid.title.slice(0, 57) + '…' : centroid.title}
        </Link>
      )}
      <div className="font-mono text-[10px] text-text-dim tabular-nums pt-0.5">
        sim {sim.toFixed(2)} · n={centroid.cluster_size ?? '—'}
      </div>
    </div>
  )
}

function Sparkline({
  history,
}: {
  history: { period: string; count: number }[]
}) {
  if (history.length === 0) {
    return <span className="text-text-dim italic font-serif text-[11px]">—</span>
  }
  const w = 180
  const h = 36
  const padX = 2
  const padY = 4
  const max = Math.max(1, ...history.map((p) => p.count))
  const stepX = (w - padX * 2) / Math.max(1, history.length - 1)
  const x = (i: number) => padX + i * stepX
  const y = (v: number) => padY + (h - padY * 2) * (1 - v / max)

  const path = history
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(p.count).toFixed(1)}`)
    .join(' ')
  const area =
    `M ${x(0)} ${h - padY} ` +
    history.map((p, i) => `L ${x(i).toFixed(1)} ${y(p.count).toFixed(1)}`).join(' ') +
    ` L ${x(history.length - 1).toFixed(1)} ${h - padY} Z`

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ height: h }}>
      <path d={area} fill="rgba(153,15,61,.10)" />
      <path d={path} fill="none" stroke="#990f3d" strokeWidth="1.3" />
      {history.map((p, i) => (
        <circle key={i} cx={x(i)} cy={y(p.count)} r="1.6" fill="#990f3d" />
      ))}
      <title>
        {history.map((p) => `${p.period}: ${p.count}`).join('\n')}
      </title>
    </svg>
  )
}
