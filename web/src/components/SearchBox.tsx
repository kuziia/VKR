import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

type Props = {
  initial?: string
  compact?: boolean
}

export default function SearchBox({ initial = '', compact = false }: Props) {
  const [q, setQ] = useState(initial)
  const navigate = useNavigate()

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = q.trim()
    if (!trimmed) return
    navigate(`/search?q=${encodeURIComponent(trimmed)}`)
  }

  return (
    <form
      onSubmit={onSubmit}
      className={
        'flex items-stretch border border-border bg-surface ' +
        (compact ? 'h-8' : 'h-9')
      }
    >
      <input
        type="search"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Поиск по корпусу OpenAlex…"
        className={
          'flex-1 px-2.5 bg-transparent outline-none font-serif text-[13px] ' +
          'placeholder:italic placeholder:text-text-dim'
        }
      />
      <button
        type="submit"
        className="px-3 text-[10px] font-mono font-semibold uppercase tracking-wide bg-accent text-white hover:bg-text"
      >
        Найти
      </button>
    </form>
  )
}
