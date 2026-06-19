import { Link, useLocation } from 'react-router-dom'
import Masthead from '../components/Masthead'
import NavStrip from '../components/NavStrip'

export default function NotFound() {
  const loc = useLocation()
  return (
    <div className="max-w-[1340px] mx-auto px-9 pt-6 pb-16">
      <Masthead />
      <NavStrip />

      <section className="py-16 border-t-2 border-text">
        <div className="text-[10px] tracking-[0.14em] uppercase text-accent font-bold mb-2.5">
          ошибка маршрута
        </div>
        <h1 className="font-serif text-[64px] font-semibold tracking-[-0.02em] leading-none mb-5">
          404 · Не найдено
        </h1>
        <p className="font-serif italic text-[18px] text-text-muted leading-snug mb-6 max-w-[600px]">
          Страница{' '}
          <span className="font-mono not-italic text-text">{loc.pathname}</span>{' '}
          не существует или была переименована. Доступные разделы — в навигации
          выше.
        </p>
        <Link
          to="/"
          className="inline-block px-3 py-1.5 text-[11px] font-mono font-semibold border border-accent bg-accent text-white hover:bg-text hover:border-text uppercase tracking-wide"
        >
          ← На главную
        </Link>
      </section>
    </div>
  )
}
