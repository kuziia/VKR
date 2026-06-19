import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

type Props = {
  from: string
  to: string
  country: string  // 'all' or ISO-2 lower
}

export default function CoverageBlock({ from, to, country }: Props) {
  const enabled = country !== 'all'  // OpenAIRE comparison only for country
  const cov = useQuery({
    queryKey: ['coverage', from, to, country],
    queryFn: () =>
      api.coverage({
        from,
        to,
        country: country === 'all' ? null : country,
      }),
    enabled,
    staleTime: 10 * 60_000,
  })

  if (!enabled) {
    return (
      <div className="text-[12px] italic font-serif text-text-muted">
        Сравнение с OpenAIRE доступно при выборе страны (RU / US / DE / CN).
      </div>
    )
  }

  if (cov.isLoading) {
    return (
      <div className="text-[12px] italic font-serif text-text-muted">
        считаем покрытие…
      </div>
    )
  }
  if (cov.isError || !cov.data) {
    return (
      <div className="text-[12px] italic font-serif text-loss">
        Не удалось получить данные OpenAIRE.
      </div>
    )
  }

  const oa = cov.data.openalex_count
  const air = cov.data.openaire_count
  const delta = oa > 0 ? ((air - oa) / oa) * 100 : 0
  const deltaColor =
    delta > 5 ? 'text-profit' : delta < -5 ? 'text-loss' : 'text-text-muted'
  const deltaSign = delta > 0 ? '+' : ''

  return (
    <div className="border border-border bg-surface p-4">
      <div className="text-[10px] uppercase tracking-[0.08em] font-bold text-accent mb-2">
        Покрытие · OpenAlex vs OpenAIRE
      </div>
      <div className="grid grid-cols-3 gap-4">
        <Stat label="OpenAlex" value={oa.toLocaleString('ru-RU')} />
        <Stat label="OpenAIRE" value={air.toLocaleString('ru-RU')} />
        <Stat
          label="Δ"
          value={
            oa === 0
              ? '—'
              : `${deltaSign}${delta.toFixed(0)}%`
          }
          colorClass={deltaColor}
        />
      </div>
      <div className="mt-3 text-[11px] italic font-serif text-text-muted leading-snug">
        Период {from} → {to}, страна <span className="font-mono not-italic uppercase">{country}</span>.
        OpenAIRE обычно даёт больше для России — за счёт репозиториев,
        не подключённых к Crossref. Полностью разные пайплайны индексации,
        прямого пересечения по DOI у нас нет.
        {air > oa &&
          ' Дополнительно нашли в OpenAIRE ≈ ' +
            (air - oa).toLocaleString('ru-RU') +
            ' работ.'}
      </div>
    </div>
  )
}

function Stat({
  label,
  value,
  colorClass = 'text-text',
}: {
  label: string
  value: string
  colorClass?: string
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.08em] text-text-muted font-bold mb-0.5">
        {label}
      </div>
      <div className={`font-mono font-semibold text-[20px] tnum ${colorClass}`}>
        {value}
      </div>
    </div>
  )
}
