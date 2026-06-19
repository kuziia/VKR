import { useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  api,
  type BertrendResponse,
  type ByFieldResponse,
  type TopCitedResponse,
  type TrendsResponse,
} from '../lib/api'

type Props = {
  trends: TrendsResponse | undefined
  top: TopCitedResponse | undefined
  byField: ByFieldResponse | undefined
  bertrend?: BertrendResponse | undefined
}

export default function InterpretBlock({ trends, top, byField, bertrend }: Props) {
  const [text, setText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [streaming, setStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const canRun = !!trends && !streaming

  const onClick = () => {
    if (streaming || !trends) return
    setText('')
    setError(null)
    setStreaming(true)
    const ctrl = new AbortController()
    abortRef.current = ctrl
    api.interpretTrends(
      {
        trends,
        top,
        by_field: byField,
        bertrend: bertrend?.available ? bertrend : undefined,
      },
      {
        onChunk: (c) => setText((prev) => prev + c),
        onError: (msg) => setError(msg),
        onDone: () => {
          setStreaming(false)
          abortRef.current = null
        },
        signal: ctrl.signal,
      },
    )
  }

  const onCancel = () => {
    abortRef.current?.abort()
    abortRef.current = null
    setStreaming(false)
  }

  return (
    <section className="mb-10">
      <div className="flex items-baseline justify-between pb-2 mb-3 border-b-2 border-text">
        <h2 className="font-serif text-[22px] font-semibold tracking-[-0.015em]">
          Интерпретация ИИ-агента
        </h2>
        <div className="text-[11px] italic text-text-muted font-serif">
          {streaming ? 'генерация…' : text ? 'готово' : '—'}
        </div>
      </div>

      <div className="bg-surface border-l-4 border-accent pl-5 pr-5 py-4 mb-3">
        {!text && !streaming && !error && (
          <div className="font-serif italic text-text-muted text-[14px]">
            Нажмите «Получить комментарий» — отправим текущий контекст
            (динамика, топ статей, распределение по полям) в Claude и
            получим аналитический комментарий.
          </div>
        )}
        {error && (
          <div className="font-serif italic text-loss text-[14px]">
            {error}
          </div>
        )}
        {(text || streaming) && (
          <div className="prose prose-sm max-w-none font-serif text-[15px] leading-relaxed [&>ul]:list-disc [&>ul]:pl-6 [&>ol]:list-decimal [&>ol]:pl-6 [&_strong]:font-semibold [&>h1]:font-serif [&>h2]:font-serif [&>h3]:font-serif [&>h2]:mt-4 [&>h3]:mt-3 [&>p]:mb-3 [&>ul]:mb-3">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
            {streaming && <span className="inline-block w-2 h-4 bg-accent align-middle animate-pulse" />}
          </div>
        )}
      </div>

      <div className="flex items-center gap-3">
        {!streaming ? (
          <button
            type="button"
            onClick={onClick}
            disabled={!canRun}
            className="px-3 py-1.5 text-[11px] font-mono font-semibold border border-accent bg-accent text-white hover:bg-text hover:border-text uppercase tracking-wide disabled:opacity-50"
          >
            {text ? 'обновить комментарий' : 'получить комментарий'}
          </button>
        ) : (
          <button
            type="button"
            onClick={onCancel}
            className="px-3 py-1.5 text-[11px] font-mono font-semibold border border-text bg-surface text-text hover:bg-text hover:text-white uppercase tracking-wide"
          >
            отменить
          </button>
        )}
        <span className="text-[10px] italic font-serif text-text-dim">
          сгенерировано LLM, не редакторский комментарий
        </span>
      </div>
    </section>
  )
}
