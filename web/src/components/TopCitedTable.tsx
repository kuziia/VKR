import { Link } from 'react-router-dom'
import type { TopCitedItem } from '../lib/api'

type Props = {
  items: TopCitedItem[] | undefined
  isLoading: boolean
  isError: boolean
}

export default function TopCitedTable({ items, isLoading, isError }: Props) {
  return (
    <table className="w-full text-[13px]">
      <thead>
        <tr className="border-b border-text">
          <th className="text-left py-2 px-2.5 text-[9px] uppercase tracking-[0.1em] text-text-muted font-bold">
            Title / Authors
          </th>
          <th className="text-right py-2 px-2.5 text-[9px] uppercase tracking-[0.1em] text-text-muted font-bold">
            Year
          </th>
          <th className="text-right py-2 px-2.5 text-[9px] uppercase tracking-[0.1em] text-text-muted font-bold">
            Cit.
          </th>
          <th className="text-center py-2 px-2.5 text-[9px] uppercase tracking-[0.1em] text-text-muted font-bold">
            OA
          </th>
        </tr>
      </thead>
      <tbody>
        {isLoading &&
          Array.from({ length: 5 }).map((_, i) => (
            <tr key={i} className="dotted-rule">
              <td className="py-2.5 px-2.5 text-text-dim italic font-serif">загрузка…</td>
              <td className="py-2.5 px-2.5 text-right font-mono text-text-dim">—</td>
              <td className="py-2.5 px-2.5 text-right font-mono text-text-dim">—</td>
              <td className="py-2.5 px-2.5 text-center text-text-dim">—</td>
            </tr>
          ))}
        {isError && (
          <tr>
            <td colSpan={4} className="py-3 px-2.5 text-loss italic font-serif">
              Ошибка загрузки
            </td>
          </tr>
        )}
        {!isLoading && !isError && (!items || items.length === 0) && (
          <tr>
            <td colSpan={4} className="py-3 px-2.5 text-text-dim italic font-serif">
              За выбранный период публикаций не найдено
            </td>
          </tr>
        )}
        {items?.map((it) => {
          const isLinkable = !!it.openalex_id && !it.openalex_id.startsWith('oaire:')
          return (
          <tr key={it.openalex_id} className="dotted-rule hover:bg-surface-2">
            <td className="py-2.5 px-2.5 align-top">
              {isLinkable ? (
                <Link
                  to={`/article/${it.openalex_id}`}
                  className="font-serif font-semibold leading-snug hover:text-accent block"
                >
                  {it.title}
                </Link>
              ) : (
                <span className="font-serif font-semibold leading-snug block">
                  {it.title}
                </span>
              )}
              {it.authors.length > 0 && (
                <div className="text-[11px] italic text-text-muted mt-0.5">
                  {it.authors.slice(0, 4).join(', ')}
                  {it.authors.length > 4 && ` +${it.authors.length - 4}`}
                </div>
              )}
              {it.primary_topic.display_name && (
                <div className="text-[10px] uppercase tracking-[0.06em] text-text-dim mt-1">
                  {it.primary_topic.field.display_name} · {it.primary_topic.display_name}
                </div>
              )}
            </td>
            <td className="py-2.5 px-2.5 text-right font-mono align-top">
              {it.publication_year ?? '—'}
            </td>
            <td className="py-2.5 px-2.5 text-right font-mono font-semibold align-top">
              {it.cited_by_count.toLocaleString('ru-RU')}
            </td>
            <td className="py-2.5 px-2.5 text-center align-top">
              {it.open_access.is_oa ? (
                <span className="inline-block px-1.5 py-px text-[9px] font-bold uppercase tracking-[0.06em] text-profit border border-profit">
                  {it.open_access.oa_status ?? 'oa'}
                </span>
              ) : (
                <span className="text-text-dim">—</span>
              )}
            </td>
          </tr>
          )
        })}
      </tbody>
    </table>
  )
}
