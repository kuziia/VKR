const COUNTRIES = [
  { code: 'all', label: 'ALL' },
  { code: 'ru', label: 'RU' },
  { code: 'us', label: 'US' },
  { code: 'cn', label: 'CN' },
  { code: 'de', label: 'DE' },
]

type Props = {
  value: string  // 'all' or ISO-2 lower-case
  onChange: (next: string) => void
}

export default function CountryPicker({ value, onChange }: Props) {
  return (
    <div className="flex gap-1.5 flex-wrap">
      {COUNTRIES.map((c) => (
        <button
          key={c.code}
          type="button"
          onClick={() => onChange(c.code)}
          className={
            'px-2.5 py-1 text-[12px] font-mono font-semibold border tracking-wide ' +
            (c.code === value
              ? 'bg-accent text-white border-accent'
              : 'bg-surface text-text-muted border-border hover:text-text')
          }
        >
          {c.label}
        </button>
      ))}
    </div>
  )
}
