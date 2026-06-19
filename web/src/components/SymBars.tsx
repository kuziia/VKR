import type { ByFieldItem } from '../lib/api'

type Props = {
  items: ByFieldItem[] | undefined
  isLoading: boolean
}

export default function SymBars({ items, isLoading }: Props) {
  if (isLoading || !items) {
    return (
      <div className="space-y-2.5">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="dotted-rule pb-2.5 grid grid-cols-[140px_1fr_60px] gap-3 items-center">
            <span className="font-serif text-[15px] text-text-dim italic">загрузка…</span>
            <div className="h-[3px] bg-surface-3" />
            <span className="font-mono text-text-dim text-right">—</span>
          </div>
        ))}
      </div>
    )
  }
  if (items.length === 0) {
    return <div className="text-text-dim italic font-serif">Нет данных</div>
  }
  const max = Math.max(1, ...items.map((i) => i.count))
  return (
    <div className="space-y-2.5">
      {items.map((it) => (
        <div
          key={it.id}
          className="dotted-rule pb-2.5 grid grid-cols-[140px_1fr_60px] gap-3 items-center"
        >
          <span className="font-serif text-[14px] font-semibold leading-tight">
            {it.display_name}
          </span>
          <div className="h-[3px] bg-surface-3">
            <div
              className="h-full bg-accent"
              style={{ width: `${(it.count / max) * 100}%` }}
            />
          </div>
          <span className="font-mono font-semibold text-[13px] text-right">
            {it.count.toLocaleString('ru-RU')}
          </span>
        </div>
      ))}
    </div>
  )
}
