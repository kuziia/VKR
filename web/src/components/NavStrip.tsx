import { Link, useLocation } from 'react-router-dom'

const items = [
  { label: 'Dashboard', to: '/' },
  { label: 'Search', to: '/search' },
]

export default function NavStrip() {
  const loc = useLocation()
  return (
    <nav className="flex border-b border-border mb-8">
      {items.map((it, i) => {
        const active =
          (it.to === '/' && loc.pathname === '/') ||
          (it.to !== '/' && loc.pathname.startsWith(it.to))
        const cls =
          'py-2.5 px-4 text-xs font-semibold uppercase tracking-[0.06em] ' +
          (i === 0 ? 'pl-0 ' : '') +
          'border-r border-border ' +
          (active
            ? 'text-accent border-b-2 border-b-accent -mb-px'
            : 'text-text-muted hover:text-text')
        return (
          <Link key={it.label} to={it.to} className={cls}>
            {it.label}
          </Link>
        )
      })}
    </nav>
  )
}
