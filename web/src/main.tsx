import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import './index.css'
import Dashboard from './pages/Dashboard'
import ArticlePage from './pages/Article'
import SearchPage from './pages/Search'
import AuthorPage from './pages/Author'
import GraphPage from './pages/Graph'
import NotFound from './pages/NotFound'

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 60_000, refetchOnWindowFocus: false } },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/article/:id" element={<ArticlePage />} />
          <Route path="/article/:id/graph" element={<GraphPage />} />
          <Route path="/author/:id" element={<AuthorPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
)
