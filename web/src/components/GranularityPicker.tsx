import type { Granularity } from '../lib/api'

const OPTIONS: { code: Granularity; label: string }[] = [
  { code: 'month', label: 'M' },
  { code: 'quarter', label: 'Q' },
  { code: 'year', label: 'Y' },
]

type Props = {
  value: Granularity
  onChange: (next: Granularity) => void
  disabled?: boolean
}

export default function GranularityPicker({ value, onChange, disabled }: Props) {
  return (
    <div className={'flex gap-1 ' + (disabled ? 'opacity-40 pointer-events-none' : '')}>
      {OPTIONS.map((o) => (
        <button
          key={o.code}
          type="button"
          onClick={() => onChange(o.code)}
          className={
            'px-2 py-1 text-[11px] font-mono font-semibold border tracking-wide ' +
            (o.code === value
              ? 'bg-accent text-white border-accent'
              : 'bg-surface text-text-muted border-border hover:text-text')
          }
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}
