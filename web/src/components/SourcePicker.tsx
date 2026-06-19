const SOURCES = [
  { code: 'openalex', label: 'OPENALEX' },
  { code: 'openaire', label: 'OPENAIRE' },
] as const

export type Source = 'openalex' | 'openaire'

type Props = {
  value: Source
  onChange: (next: Source) => void
}

export default function SourcePicker({ value, onChange }: Props) {
  return (
    <div className="flex gap-1.5">
      {SOURCES.map((s) => (
        <button
          key={s.code}
          type="button"
          onClick={() => onChange(s.code as Source)}
          className={
            'px-2.5 py-1 text-[12px] font-mono font-semibold border tracking-wide ' +
            (s.code === value
              ? 'bg-accent text-white border-accent'
              : 'bg-surface text-text-muted border-border hover:text-text')
          }
        >
          {s.label}
        </button>
      ))}
    </div>
  )
}
