import type { Granularity, TrendPoint } from '../lib/api'

type Props = {
  points: TrendPoint[] | undefined
  granularity: Granularity
  isLoading: boolean
}

function pickWindows(points: TrendPoint[], granularity: Granularity) {
  // Returns 4 windows: last 1, last 3, last 6, full — labelled by granularity unit.
  const total = points.reduce((a, p) => a + p.count, 0)
  const tail = (n: number) => points.slice(-n).reduce((a, p) => a + p.count, 0)
  const unit = granularity === 'year' ? 'год' : 'мес'
  return [
    { head: `1 ${unit}`, value: tail(1) },
    { head: `3 ${unit}.`, value: tail(3) },
    { head: `6 ${unit}.`, value: tail(6) },
    { head: 'Всего', value: total },
  ]
}

export default function MetricsRow({ points, granularity, isLoading }: Props) {
  const cells = points ? pickWindows(points, granularity) : null

  return (
    <section className="grid grid-cols-4 border-y border-border py-4 mb-6">
      {(cells ?? Array.from({ length: 4 }, () => ({ head: '—', value: 0 }))).map((c, i) => (
        <div
          key={i}
          className={'px-6 ' + (i === 0 ? 'pl-0 border-l-0' : 'border-l border-dotted border-border')}
        >
          <h4 className="text-[10px] uppercase tracking-[0.1em] text-text-muted font-bold mb-2">
            {c.head}
          </h4>
          <div className="font-serif text-[30px] font-semibold tracking-[-0.02em] leading-none tnum">
            {isLoading || !points ? '—' : c.value.toLocaleString('ru-RU')}
          </div>
          <div className="text-[12px] font-semibold mt-1.5 font-mono text-text-muted">
            публикаций
          </div>
        </div>
      ))}
    </section>
  )
}
