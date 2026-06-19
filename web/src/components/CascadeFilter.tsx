import { useMemo } from 'react'
import type { Domain, Field, Subfield, Taxonomy, Topic, Level } from '../lib/api'

export type CascadeValue = {
  level: Level
  id: string | null
}

type Props = {
  taxonomy: Taxonomy | undefined
  isLoading: boolean
  isError: boolean
  domain: string | null
  field: string | null
  subfield: string | null
  topic: string | null
  onChange: (next: { domain: string | null; field: string | null; subfield: string | null; topic: string | null }) => void
  layout?: 'vertical' | 'horizontal'
}

function findDomain(t: Taxonomy, id: string | null): Domain | undefined {
  return id ? t.domains.find((d) => d.id === id) : undefined
}
function findField(d: Domain | undefined, id: string | null): Field | undefined {
  return d && id ? d.fields.find((f) => f.id === id) : undefined
}
function findSubfield(f: Field | undefined, id: string | null): Subfield | undefined {
  return f && id ? f.subfields.find((s) => s.id === id) : undefined
}

export function selectionToValue(sel: {
  domain: string | null
  field: string | null
  subfield: string | null
  topic: string | null
}): CascadeValue {
  if (sel.topic) return { level: 'topic', id: sel.topic.replace(/^T/, '') }
  if (sel.subfield) return { level: 'subfield', id: sel.subfield }
  if (sel.field) return { level: 'field', id: sel.field }
  if (sel.domain) return { level: 'domain', id: sel.domain }
  return { level: 'all', id: null }
}

export default function CascadeFilter({
  taxonomy,
  isLoading,
  isError,
  domain,
  field,
  subfield,
  topic,
  onChange,
  layout = 'vertical',
}: Props) {
  const dom = useMemo(() => (taxonomy ? findDomain(taxonomy, domain) : undefined), [taxonomy, domain])
  const fld = useMemo(() => findField(dom, field), [dom, field])
  const sub = useMemo(() => findSubfield(fld, subfield), [fld, subfield])

  if (layout === 'horizontal') {
    return (
      <div className="grid grid-cols-4 gap-1.5">
        <CompactSelect
          value={domain ?? ''}
          disabled={isLoading || isError}
          onChange={(v) =>
            onChange({ domain: v || null, field: null, subfield: null, topic: null })
          }
          placeholder="domain — все —"
        >
          {taxonomy?.domains.map((d) => (
            <option key={d.id} value={d.id}>
              {d.display_name}
            </option>
          ))}
        </CompactSelect>
        <CompactSelect
          value={field ?? ''}
          disabled={!dom}
          onChange={(v) => onChange({ domain, field: v || null, subfield: null, topic: null })}
          placeholder="field — все —"
        >
          {dom?.fields.map((f) => (
            <option key={f.id} value={f.id}>
              {f.display_name}
            </option>
          ))}
        </CompactSelect>
        <CompactSelect
          value={subfield ?? ''}
          disabled={!fld}
          onChange={(v) => onChange({ domain, field, subfield: v || null, topic: null })}
          placeholder="subfield — все —"
        >
          {fld?.subfields.map((s) => (
            <option key={s.id} value={s.id}>
              {s.display_name}
            </option>
          ))}
        </CompactSelect>
        <CompactSelect
          value={topic ?? ''}
          disabled={!sub}
          onChange={(v) => onChange({ domain, field, subfield, topic: v || null })}
          placeholder="topic — все —"
        >
          {sub?.topics.map((t: Topic) => (
            <option key={t.id} value={t.id}>
              {t.display_name}
            </option>
          ))}
        </CompactSelect>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <Row label="Domain">
        <Select
          value={domain ?? ''}
          disabled={isLoading || isError}
          onChange={(v) =>
            onChange({ domain: v || null, field: null, subfield: null, topic: null })
          }
        >
          <option value="">— все —</option>
          {taxonomy?.domains.map((d) => (
            <option key={d.id} value={d.id}>
              {d.display_name}
            </option>
          ))}
        </Select>
      </Row>
      <Row label="Field">
        <Select
          value={field ?? ''}
          disabled={!dom}
          onChange={(v) => onChange({ domain, field: v || null, subfield: null, topic: null })}
        >
          <option value="">— все —</option>
          {dom?.fields.map((f) => (
            <option key={f.id} value={f.id}>
              {f.display_name}
            </option>
          ))}
        </Select>
      </Row>
      <Row label="Subfield">
        <Select
          value={subfield ?? ''}
          disabled={!fld}
          onChange={(v) => onChange({ domain, field, subfield: v || null, topic: null })}
        >
          <option value="">— все —</option>
          {fld?.subfields.map((s) => (
            <option key={s.id} value={s.id}>
              {s.display_name}
            </option>
          ))}
        </Select>
      </Row>
      <Row label="Topic">
        <Select
          value={topic ?? ''}
          disabled={!sub}
          onChange={(v) => onChange({ domain, field, subfield, topic: v || null })}
        >
          <option value="">— все —</option>
          {sub?.topics.map((t: Topic) => (
            <option key={t.id} value={t.id}>
              {t.display_name}
            </option>
          ))}
        </Select>
      </Row>
    </div>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[80px_1fr] items-center gap-2">
      <div className="text-[11px] uppercase text-text-muted tracking-[0.06em] font-semibold">
        {label}
      </div>
      {children}
    </div>
  )
}

function Select({
  value,
  disabled,
  onChange,
  children,
}: {
  value: string
  disabled?: boolean
  onChange: (v: string) => void
  children: React.ReactNode
}) {
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className="w-full bg-surface border border-border px-2 py-1 text-[13px] font-mono disabled:text-text-dim disabled:bg-surface-2"
    >
      {children}
    </select>
  )
}

function CompactSelect({
  value,
  disabled,
  onChange,
  placeholder,
  children,
}: {
  value: string
  disabled?: boolean
  onChange: (v: string) => void
  placeholder: string
  children: React.ReactNode
}) {
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className="flt w-full bg-surface border border-border px-1.5 py-1 text-[11px] font-mono disabled:text-text-dim disabled:bg-surface-2 truncate"
    >
      <option value="">{placeholder}</option>
      {children}
    </select>
  )
}
