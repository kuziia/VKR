import SearchBox from './SearchBox'

export default function Masthead() {
  const today = new Date().toLocaleDateString('ru-RU', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  })
  return (
    <header className="double-rule pb-3.5 mb-6 flex items-end justify-between gap-6">
      <div className="flex items-end gap-4">
        <div className="font-serif text-[34px] font-bold tracking-[-0.02em] leading-none">
          NAUKA-MONITOR
        </div>
        <div className="text-[11px] italic font-serif text-text-muted pb-1">
          Мониторинг научных публикаций · {today}
        </div>
      </div>
      <div className="flex items-center gap-4">
        <div className="w-[280px]">
          <SearchBox compact />
        </div>
        <div className="flex flex-col items-end text-xs text-text-muted leading-tight">
          <span>OpenAlex · OpenAIRE</span>
          <span className="font-mono font-semibold text-text">live · кэш 24ч</span>
        </div>
      </div>
    </header>
  )
}
