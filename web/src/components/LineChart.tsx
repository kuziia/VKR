import { useMemo } from 'react'
import type { TrendPoint } from '../lib/api'

type Props = {
  points: TrendPoint[]
  height?: number
}

export default function LineChart({ points, height = 300 }: Props) {
  const view = useMemo(() => {
    if (!points.length) return null
    const w = 800
    const h = height
    const padL = 44
    const padR = 16
    const padT = 12
    const padB = 28
    const innerW = w - padL - padR
    const innerH = h - padT - padB
    const max = Math.max(1, ...points.map((p) => p.count))
    const stepX = innerW / Math.max(1, points.length - 1)

    const x = (i: number) => padL + i * stepX
    const y = (v: number) => padT + innerH - (v / max) * innerH

    const xs = points.map((_, i) => x(i))
    const ys = points.map((p) => y(p.count))

    const linePath = points
      .map((_, i) => `${i === 0 ? 'M' : 'L'} ${xs[i].toFixed(1)} ${ys[i].toFixed(1)}`)
      .join(' ')

    const areaPath =
      `M ${xs[0]} ${y(0)} ` +
      points.map((_, i) => `L ${xs[i].toFixed(1)} ${ys[i].toFixed(1)}`).join(' ') +
      ` L ${xs[xs.length - 1]} ${y(0)} Z`

    const yTicks = [0, max / 2, max].map((v) => ({
      v: Math.round(v),
      y: y(v),
    }))

    const labelEvery = Math.max(1, Math.ceil(points.length / 8))
    const xLabels = points
      .map((p, i) => ({ period: p.period, i, x: xs[i] }))
      .filter((p) => p.i % labelEvery === 0 || p.i === points.length - 1)

    return {
      w, h, padL, padR, padT, padB, innerW, innerH,
      linePath, areaPath, yTicks, xLabels, xs, ys, max,
    }
  }, [points, height])

  if (!view) {
    return (
      <div
        className="grid place-items-center text-text-dim italic font-serif text-sm border border-dashed border-border"
        style={{ height }}
      >
        Нет данных
      </div>
    )
  }

  const { w, h, padL, padR, padT, innerH, linePath, areaPath, yTicks, xLabels, xs, ys } = view

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="xMidYMid meet"
      className="w-full"
      style={{ height }}
    >
      {yTicks.map((t, i) => (
        <g key={i}>
          <line
            x1={padL}
            x2={w - padR}
            y1={t.y}
            y2={t.y}
            stroke="rgba(216,205,182,.7)"
            strokeDasharray="2,3"
          />
          <text
            x={padL - 6}
            y={t.y + 3}
            textAnchor="end"
            fontFamily="JetBrains Mono, monospace"
            fontSize="10"
            fill="#8a8170"
          >
            {t.v.toLocaleString('ru-RU')}
          </text>
        </g>
      ))}
      <path d={areaPath} fill="rgba(153,15,61,.08)" />
      <path d={linePath} stroke="#990f3d" strokeWidth="1.5" fill="none" />
      {xs.map((cx, i) => (
        <circle key={i} cx={cx} cy={ys[i]} r={2} fill="#990f3d" />
      ))}
      {xLabels.map((p) => (
        <text
          key={p.i}
          x={p.x}
          y={padT + innerH + 18}
          textAnchor="middle"
          fontFamily="JetBrains Mono, monospace"
          fontSize="10"
          fill="#8a8170"
        >
          {p.period}
        </text>
      ))}
    </svg>
  )
}
