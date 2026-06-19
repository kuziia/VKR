const LANGS = [
  { code: 'ru', label: 'RU' },
  { code: 'en', label: 'EN' },
  { code: 'all', label: 'ALL' },
]

type Props = {
  value: string
  onChange: (lang: string) => void
}

export default function LangPicker({ value, onChange }: Props) {
  return (
    <div className="flex gap-1.5">
      {LANGS.map((l) => (
        <button
          key={l.code}
          type="button"
          onClick={() => onChange(l.code)}
          className={
            'px-2.5 py-1 text-[12px] font-mono font-semibold border tracking-wide ' +
            (l.code === value
              ? 'bg-accent text-white border-accent'
              : 'bg-surface text-text-muted border-border hover:text-text')
          }
        >
          {l.label}
        </button>
      ))}
    </div>
  )
}
