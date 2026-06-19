import type { Granularity } from './api'

export type PresetKey = '1M' | '6M' | '1Y' | '5Y' | '10Y'

export type Period = {
  preset: PresetKey
  from: string // YYYY or YYYY-MM
  to: string
  granularity: Granularity
}

function pad(n: number): string {
  return String(n).padStart(2, '0')
}

function ymd(d: Date): string {
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}`
}

function ymdYear(d: Date): string {
  return String(d.getUTCFullYear())
}

export function periodFor(preset: PresetKey, today: Date = new Date()): Period {
  // OpenAlex updates trail real-time by ~weeks. We end at the previous
  // complete month for stable counts, except for 10Y where year granularity
  // smooths it out.
  const t = new Date(Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), 1))
  const endMonth = new Date(t)
  endMonth.setUTCMonth(endMonth.getUTCMonth() - 1)

  switch (preset) {
    case '1M': {
      // last 1 month, monthly granularity
      const from = new Date(endMonth)
      return { preset, from: ymd(from), to: ymd(endMonth), granularity: 'month' }
    }
    case '6M': {
      const from = new Date(endMonth)
      from.setUTCMonth(from.getUTCMonth() - 5)
      return { preset, from: ymd(from), to: ymd(endMonth), granularity: 'month' }
    }
    case '1Y': {
      const from = new Date(endMonth)
      from.setUTCMonth(from.getUTCMonth() - 11)
      return { preset, from: ymd(from), to: ymd(endMonth), granularity: 'month' }
    }
    case '5Y': {
      const fromYear = today.getUTCFullYear() - 4
      return { preset, from: String(fromYear), to: ymdYear(today), granularity: 'year' }
    }
    case '10Y': {
      const fromYear = today.getUTCFullYear() - 9
      return { preset, from: String(fromYear), to: ymdYear(today), granularity: 'year' }
    }
  }
}
