import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type Simulation,
  type SimulationNodeDatum,
  type SimulationLinkDatum,
} from 'd3-force'
import type { CitationGraph, GraphNode } from '../lib/api'

type Props = {
  graph: CitationGraph
  width?: number
  height?: number
}

type SimNode = SimulationNodeDatum & {
  id: string
  data: GraphNode
  r: number
}
type SimLink = SimulationLinkDatum<SimNode> & {
  kind: 'refs' | 'cites'
}

const PALETTE: Record<number, string> = {
  0: '#990f3d', // root — accent
  1: '#5e564a', // 1 hop
  2: '#a89c82',
  3: '#c8c0a8',
}

function radiusFor(cited: number): number {
  // log-scale, root needs to be big-ish too
  return 6 + Math.min(28, Math.sqrt(Math.max(1, cited)) * 0.5)
}

export default function CitationGraphView({ graph, width = 920, height = 560 }: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null)
  const navigate = useNavigate()
  const [hover, setHover] = useState<SimNode | null>(null)
  const [, setTick] = useState(0)
  const simRef = useRef<Simulation<SimNode, SimLink> | null>(null)

  const { nodes, links } = useMemo(() => {
    const ns: SimNode[] = graph.nodes.map((n) => ({
      id: n.id,
      data: n,
      r: radiusFor(n.cited_by_count),
    }))
    const ls: SimLink[] = graph.edges.map((e) => ({
      source: e.source,
      target: e.target,
      kind: e.kind,
    }))
    return { nodes: ns, links: ls }
  }, [graph])

  useEffect(() => {
    const sim = forceSimulation<SimNode>(nodes)
      .force(
        'link',
        forceLink<SimNode, SimLink>(links)
          .id((d) => d.id)
          .distance(85)
          .strength(0.8),
      )
      .force('charge', forceManyBody().strength(-220))
      .force('center', forceCenter(width / 2, height / 2))
      .force('collide', forceCollide<SimNode>().radius((d) => d.r + 4))
      .alphaDecay(0.04)
      .on('tick', () => setTick((t) => t + 1))

    simRef.current = sim
    return () => {
      sim.stop()
    }
  }, [nodes, links, width, height])

  const onNodeClick = (n: SimNode) => {
    if (n.id === graph.root_id) return
    navigate(`/article/${n.id}`)
  }

  return (
    <div className="border border-border bg-surface p-3">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        style={{ height }}
      >
        <defs>
          <marker
            id="arrow-refs"
            viewBox="0 -5 10 10"
            refX="14"
            refY="0"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path d="M0,-5L10,0L0,5" fill="#a89c82" />
          </marker>
          <marker
            id="arrow-cites"
            viewBox="0 -5 10 10"
            refX="14"
            refY="0"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path d="M0,-5L10,0L0,5" fill="#990f3d" opacity="0.55" />
          </marker>
        </defs>

        {/* edges */}
        {links.map((l, i) => {
          const s = l.source as SimNode
          const t = l.target as SimNode
          if (typeof s === 'string' || typeof t === 'string') return null
          return (
            <line
              key={i}
              x1={s.x ?? 0}
              y1={s.y ?? 0}
              x2={t.x ?? 0}
              y2={t.y ?? 0}
              stroke={l.kind === 'refs' ? '#a89c82' : '#990f3d'}
              strokeOpacity={l.kind === 'refs' ? 0.6 : 0.45}
              strokeWidth={1.1}
              markerEnd={`url(#arrow-${l.kind})`}
            />
          )
        })}

        {/* nodes */}
        {nodes.map((n) => {
          const isRoot = n.id === graph.root_id
          const fill = PALETTE[n.data.depth] ?? '#a89c82'
          return (
            <g
              key={n.id}
              transform={`translate(${n.x ?? 0},${n.y ?? 0})`}
              onClick={() => onNodeClick(n)}
              onMouseEnter={() => setHover(n)}
              onMouseLeave={() => setHover(null)}
              className={isRoot ? 'cursor-default' : 'cursor-pointer'}
            >
              <circle
                r={n.r}
                fill={fill}
                stroke={isRoot ? '#990f3d' : '#1a1a1a'}
                strokeWidth={isRoot ? 3 : 0.8}
                fillOpacity={isRoot ? 0.92 : 0.75}
              />
              {(isRoot || n.r >= 14) && (
                <text
                  y={n.r + 12}
                  textAnchor="middle"
                  fontFamily="JetBrains Mono, monospace"
                  fontSize={isRoot ? 11 : 10}
                  fill="#1a1a1a"
                  className="pointer-events-none"
                >
                  {n.data.year ?? '—'}
                </text>
              )}
            </g>
          )
        })}

        {/* hover tooltip — last so it's on top */}
        {hover && (
          <g
            transform={`translate(${(hover.x ?? 0) + 16},${(hover.y ?? 0) - 28})`}
            className="pointer-events-none"
          >
            <rect
              x={0}
              y={0}
              width={Math.min(420, hover.data.title.length * 6 + 60)}
              height={42}
              fill="#fff9f0"
              stroke="#1a1a1a"
              strokeWidth={1}
            />
            <text
              x={8}
              y={16}
              fontFamily="Source Serif 4, Georgia, serif"
              fontSize={12}
              fill="#1a1a1a"
            >
              {hover.data.title.length > 70
                ? hover.data.title.slice(0, 67) + '…'
                : hover.data.title}
            </text>
            <text
              x={8}
              y={32}
              fontFamily="JetBrains Mono, monospace"
              fontSize={10}
              fill="#5e564a"
            >
              {hover.data.year ?? '—'} · cit={hover.data.cited_by_count.toLocaleString('ru-RU')} · depth={hover.data.depth}
            </text>
          </g>
        )}
      </svg>

      <div className="flex items-center gap-5 px-2 pt-3 text-[11px] font-mono text-text-muted">
        <Legend label="root" color={PALETTE[0]} />
        <Legend label="depth 1" color={PALETTE[1]} />
        <Legend label="depth 2" color={PALETTE[2]} />
        <Legend label="depth 3" color={PALETTE[3]} />
        <span className="ml-auto italic font-serif text-[10px]">
          размер ∝ √цитирований; кликни узел — откроется статья
        </span>
      </div>
    </div>
  )
}

function Legend({ label, color }: { label: string; color: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="inline-block w-3 h-3 rounded-full"
        style={{ background: color }}
      />
      {label}
    </span>
  )
}
