import { Link, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import Masthead from '../components/Masthead'
import NavStrip from '../components/NavStrip'

export default function AuthorPage() {
  const { id = '' } = useParams<{ id: string }>()

  const profile = useQuery({
    queryKey: ['author', id],
    queryFn: () => api.author(id),
    enabled: !!id,
    staleTime: 5 * 60_000,
  })
  const works = useQuery({
    queryKey: ['author-works', id],
    queryFn: () => api.authorWorks(id, { limit: 25 }),
    enabled: !!id,
    staleTime: 5 * 60_000,
  })

  return (
    <div className="max-w-[1340px] mx-auto px-9 pt-6 pb-16">
      <Masthead />
      <NavStrip />

      <div className="text-[11px] italic font-serif text-text-muted mb-4">
        <Link to="/" className="hover:text-accent">← Главная</Link>
        {' · '}
        <span className="font-mono not-italic">{id}</span>
      </div>

      {profile.isLoading && (
        <div className="py-12 text-text-dim italic font-serif text-center">
          загрузка профиля…
        </div>
      )}
      {profile.isError && (
        <div className="py-8 text-loss italic font-serif">
          Не удалось загрузить автора: {(profile.error as Error).message}
        </div>
      )}

      {profile.data && (
        <section className="grid grid-cols-[2fr_1fr] gap-10 pb-8 border-b border-border mb-8">
          <div>
            <div className="text-[10px] tracking-[0.14em] uppercase text-accent font-bold mb-2.5">
              Профиль автора · OpenAlex
            </div>
            <h1 className="font-serif text-[44px] font-semibold tracking-[-0.018em] leading-[1.05] mb-3">
              {profile.data.display_name}
            </h1>
            {profile.data.alternatives.length > 0 && (
              <div className="font-serif italic text-[13px] text-text-muted mb-3">
                также:{' '}
                {profile.data.alternatives.slice(0, 4).join(' · ')}
              </div>
            )}
            {profile.data.last_known_institutions.length > 0 && (
              <div className="text-[13px] font-serif text-text-muted mb-3">
                {profile.data.last_known_institutions
                  .slice(0, 3)
                  .map((i) =>
                    i.country_code
                      ? `${i.display_name} (${i.country_code.toUpperCase()})`
                      : i.display_name,
                  )
                  .join(' · ')}
              </div>
            )}
            {profile.data.orcid && (
              <a
                href={`https://orcid.org/${profile.data.orcid}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[12px] font-mono text-accent hover:underline"
              >
                ORCID {profile.data.orcid}
              </a>
            )}
          </div>

          <aside className="border-l border-border pl-8 grid grid-cols-2 gap-x-4 gap-y-3 self-start">
            <Stat label="Публикаций" value={profile.data.works_count.toLocaleString('ru-RU')} />
            <Stat label="Цитирований" value={profile.data.cited_by_count.toLocaleString('ru-RU')} />
            <Stat label="h-index" value={profile.data.h_index ?? '—'} />
            <Stat label="i10" value={profile.data.i10_index ?? '—'} />
            {profile.data.mean_citedness !== null && (
              <Stat
                label="2yr mean cit."
                value={profile.data.mean_citedness.toFixed(2)}
              />
            )}
            {profile.data.topics.length > 0 && (
              <div className="col-span-2 mt-2">
                <div className="text-[10px] uppercase tracking-[0.08em] font-bold text-text-muted mb-1.5">
                  Темы
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {profile.data.topics.slice(0, 5).map((t) => (
                    <span
                      key={t.id ?? t.display_name}
                      className="px-2 py-0.5 text-[11px] font-mono border border-border bg-surface text-text-muted"
                    >
                      {t.display_name}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </aside>
        </section>
      )}

      <section>
        <div className="flex items-baseline justify-between pb-2.5 mb-4 border-b-2 border-text">
          <h2 className="font-serif text-[24px] font-semibold tracking-[-0.015em]">
            Самые цитируемые работы
          </h2>
          <div className="text-[11px] italic text-text-muted font-serif">
            top-{works.data?.items?.length ?? 0}
          </div>
        </div>

        {works.isLoading && (
          <div className="py-8 text-text-dim italic font-serif text-center">
            загрузка работ…
          </div>
        )}
        {works.isError && (
          <div className="py-4 text-loss italic font-serif">
            Не удалось загрузить работы автора.
          </div>
        )}
        {works.data && works.data.items.length === 0 && (
          <div className="py-4 text-text-dim italic font-serif">
            У этого автора нет работ в OpenAlex.
          </div>
        )}
        {works.data && works.data.items.length > 0 && (
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
              {works.data.items.map((it) => (
                <tr key={it.openalex_id} className="dotted-rule hover:bg-surface-2">
                  <td className="py-2.5 px-2.5 align-top">
                    <Link
                      to={`/article/${it.openalex_id}`}
                      className="font-serif font-semibold leading-snug hover:text-accent block"
                    >
                      {it.title}
                    </Link>
                    {it.authors.length > 0 && (
                      <div className="text-[11px] italic text-text-muted mt-0.5">
                        {it.authors.slice(0, 5).join(', ')}
                        {it.authors.length > 5 && ` +${it.authors.length - 5}`}
                      </div>
                    )}
                    {it.primary_topic.field.display_name && (
                      <div className="text-[10px] uppercase tracking-[0.06em] text-text-dim mt-1">
                        {it.primary_topic.field.display_name}
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
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}

function Stat({
  label,
  value,
}: {
  label: string
  value: string | number
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.08em] text-text-muted font-bold mb-0.5">
        {label}
      </div>
      <div className="font-mono font-semibold text-base tnum text-text">
        {value}
      </div>
    </div>
  )
}
