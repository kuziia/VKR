import type { Granularity, Source, Taxonomy } from '../lib/api'
import type { PresetKey } from '../lib/period'
import CascadeFilter from './CascadeFilter'
import CountryPicker from './CountryPicker'
import GranularityPicker from './GranularityPicker'
import LangPicker from './LangPicker'
import PeriodPicker from './PeriodPicker'
import SourcePicker from './SourcePicker'

type Props = {
  source: Source
  onSource: (s: Source) => void

  taxonomy: Taxonomy | undefined
  taxLoading: boolean
  taxError: boolean
  domain: string | null
  field: string | null
  subfield: string | null
  topic: string | null
  onCascade: (next: { domain: string | null; field: string | null; subfield: string | null; topic: string | null }) => void

  preset: PresetKey
  onPreset: (p: PresetKey) => void
  granularity: Granularity
  onGranularity: (g: Granularity) => void
  granularityLocked?: boolean
  country: string
  onCountry: (c: string) => void
  lang: string
  onLang: (l: string) => void
}

export default function FilterBar(props: Props) {
  const taxonomyDisabled = props.source === 'openaire'
  const topicsCount = props.taxonomy
    ? props.taxonomy.domains.reduce(
        (a, d) =>
          a +
          d.fields.reduce(
            (b, f) => b + f.subfields.reduce((c, s) => c + s.topics.length, 0),
            0,
          ),
        0,
      )
    : 0

  const langDisabled = props.source === 'openaire'

  return (
    <section className="grid grid-cols-[1fr_1.8fr_0.9fr_0.6fr_1.1fr_1.1fr] gap-5 pb-5 mb-6 border-b border-border">
      <div>
        <h3 className="font-serif text-[11px] font-bold uppercase tracking-[0.08em] mb-2 text-accent">
          Источник
        </h3>
        <SourcePicker value={props.source} onChange={props.onSource} />
      </div>

      <div>
        <h3 className="font-serif text-[11px] font-bold uppercase tracking-[0.08em] mb-2 text-accent">
          Узел таксономии
          {props.taxonomy && (
            <span className="ml-2 font-sans font-normal italic text-text-muted normal-case tracking-normal text-[10px]">
              {props.taxonomy.domains.length} доменов · {topicsCount} топиков
            </span>
          )}
          {taxonomyDisabled && (
            <span className="ml-2 font-sans font-normal italic text-text-dim normal-case tracking-normal text-[10px]">
              недоступно для OpenAIRE
            </span>
          )}
        </h3>
        <div className={taxonomyDisabled ? 'opacity-40 pointer-events-none' : ''}>
          <CascadeFilter
            layout="horizontal"
            taxonomy={props.taxonomy}
            isLoading={props.taxLoading}
            isError={props.taxError}
            domain={props.domain}
            field={props.field}
            subfield={props.subfield}
            topic={props.topic}
            onChange={props.onCascade}
          />
        </div>
      </div>

      <div>
        <h3 className="font-serif text-[11px] font-bold uppercase tracking-[0.08em] mb-2 text-accent">
          Период
        </h3>
        <PeriodPicker value={props.preset} onChange={props.onPreset} />
      </div>

      <div>
        <h3 className="font-serif text-[11px] font-bold uppercase tracking-[0.08em] mb-2 text-accent">
          Гран.
          {props.granularityLocked && (
            <span className="ml-1.5 font-sans font-normal italic text-text-dim normal-case tracking-normal text-[10px]">
              (Y)
            </span>
          )}
        </h3>
        <GranularityPicker
          value={props.granularity}
          onChange={props.onGranularity}
          disabled={props.granularityLocked}
        />
      </div>

      <div>
        <h3 className="font-serif text-[11px] font-bold uppercase tracking-[0.08em] mb-2 text-accent">
          Страна авторов
        </h3>
        <CountryPicker value={props.country} onChange={props.onCountry} />
      </div>

      <div>
        <h3 className="font-serif text-[11px] font-bold uppercase tracking-[0.08em] mb-2 text-accent">
          Язык статьи
          {langDisabled && (
            <span className="ml-1.5 font-sans font-normal italic text-text-dim normal-case tracking-normal text-[10px]">
              (n/a)
            </span>
          )}
        </h3>
        <div className={langDisabled ? 'opacity-40 pointer-events-none' : ''}>
          <LangPicker value={props.lang} onChange={props.onLang} />
        </div>
      </div>
    </section>
  )
}
