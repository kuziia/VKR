import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import Masthead from '../components/Masthead'
import NavStrip from '../components/NavStrip'
import CitationGraphView from '../components/CitationGraph'

export default function GraphPage() {
  const { id = '' } = useParams<{ id: string }>()
  const [depth, setDepth] = useState(1)
  const [fanout, setFanout] = useState(8)

  const article = useQuery({
    queryKey: ['article', id],
    queryFn: () => api.article(id),
    enabled: !!id,
    staleTime: 5 * 60_000,
  })
  const graph = useQuery({
    queryKey: ['graph', id, depth, fanout],
    queryFn: () => api.citationGraph(id, { depth, fanout }),
    enabled: !!id,
    staleTime: 10 * 60_000,
  })

  return (
    <div className="max-w-[1340px] mx-auto px-9 pt-6 pb-16">
      <Masthead />
      <NavStrip />

      <div className="text-[11px] italic font-serif text-text-muted mb-4">
        <Link to="/" className="hover:text-accent">← Главная</Link>
        {' · '}
        <Link to={`/article/${id}`} className="hover:text-accent">
          ← статья
        </Link>
        {' · '}
        <span className="font-mono not-italic">{id}</span>
      </div>

      <div className="text-[10px] tracking-[0.14em] uppercase text-accent font-bold mb-2.5">
        Граф цитирования · OpenAlex
      </div>
      <h1 className="font-serif text-[36px] font-semibold tracking-[-0.018em] leading-[1.1] mb-3 max-w-[900px]">
        {article.data?.title ?? '—'}
      </h1>

      <div className="flex items-center gap-6 mb-5 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-[0.08em] font-bold text-text-muted">
            Глубина
          </span>
          {[1, 2, 3].map((d) => (
            <button
              key={d}
              type="button"
              onClick={() => setDepth(d)}
              className={
                'px-2.5 py-1 text-[12px] font-mono font-semibold border tracking-wide ' +
                (d === depth
                  ? 'bg-accent text-white border-accent'
                  : 'bg-surface text-text-muted border-border hover:text-text')
              }
            >
              {d}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-[0.08em] font-bold text-text-muted">
            Ветвление
          </span>
          {[5, 8, 12].map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setFanout(f)}
              className={
                'px-2.5 py-1 text-[12px] font-mono font-semibold border tracking-wide ' +
                (f === fanout
                  ? 'bg-accent text-white border-accent'
                  : 'bg-surface text-text-muted border-border hover:text-text')
              }
            >
              {f}
            </button>
          ))}
        </div>
        <div className="text-[11px] italic font-serif text-text-muted">
          {graph.isLoading || graph.isFetching
            ? 'строим граф…'
            : graph.data
              ? `узлов: ${graph.data.nodes.length}, рёбер: ${graph.data.edges.length}`
              : '—'}
        </div>
      </div>

      {graph.isLoading && !graph.data && (
        <div className="border border-dashed border-border h-[560px] grid place-items-center text-text-dim italic font-serif">
          OpenAlex может отвечать долго: depth=2/3 — десятки запросов.
        </div>
      )}
      {graph.isError && (
        <div className="border border-loss bg-surface px-4 py-3 font-serif text-loss">
          Не удалось построить граф: {(graph.error as Error).message}
        </div>
      )}
      {graph.data && <CitationGraphView graph={graph.data} />}

      <div className="mt-6 text-[12px] font-serif italic text-text-muted leading-snug max-w-[840px]">
        Стрелки серого цвета — <strong className="not-italic font-semibold">refs</strong>{' '}
        (статья ссылается на источник). Стрелки бордового —{' '}
        <strong className="not-italic font-semibold">cites</strong> (источник цитирует
        статью). На больших глубинах граф сильно разрастается; число
        отображаемых узлов ограничено 80.
      </div>
    </div>
  )
}
