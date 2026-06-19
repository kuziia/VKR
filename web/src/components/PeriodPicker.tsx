import type { PresetKey } from '../lib/period'

const presets: PresetKey[] = ['1M', '6M', '1Y', '5Y', '10Y']

type Props = {
  value: PresetKey
  onChange: (next: PresetKey) => void
}

export default function PeriodPicker({ value, onChange }: Props) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {presets.map((p) => (
        <button
          key={p}
          type="button"
          onClick={() => onChange(p)}
          className={
            'px-2.5 py-1 text-[12px] font-mono font-semibold border tracking-wide ' +
            (p === value
              ? 'bg-accent text-white border-accent'
              : 'bg-surface text-text-muted border-border hover:text-text')
          }
        >
          {p}
        </button>
      ))}
    </div>
  )
}
