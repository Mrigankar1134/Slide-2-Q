import React, { useMemo, useState } from 'react'

export default function Questions({ slides, aiMeta }) {
  const [query, setQuery] = useState('')

  const filteredSlides = useMemo(() => {
    if (!query) return slides
    const q = query.toLowerCase()
    return slides.map(s => ({
      ...s,
      questions: (s.questions || []).filter(qo => qo.question.toLowerCase().includes(q))
    }))
  }, [slides, query])

  return (
    <section className="mt-2">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold tracking-tight">Generated Questions</h2>
          <span className="text-[10px] uppercase bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded-full">AI refined{aiMeta?.ai_model ? ` · ${aiMeta.ai_model}` : ''}</span>
        </div>
        <div className="relative w-full sm:w-72">
          <input
            type="text"
            placeholder="Search questions..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full border rounded-md pl-9 pr-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <svg className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" strokeWidth="1.5" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35m0 0A7.5 7.5 0 104.5 4.5a7.5 7.5 0 0012.15 12.15z" />
          </svg>
        </div>
      </div>

      <div className="space-y-6">
        {filteredSlides.map((slide) => (
          <div key={slide.slide_index} className="bg-white/80 border rounded-xl p-6 shadow-sm">
            <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
              <div>
                <div className="text-xs uppercase tracking-wide text-gray-500">Slide {slide.slide_index}</div>
                <div className="text-sm text-gray-800 mt-1 whitespace-pre-wrap">{slide.text}</div>
              </div>
              <div className="text-left md:text-right">
                <div className="text-xs text-gray-500">Keywords</div>
                <div className="flex flex-wrap gap-1 mt-1 md:justify-end">
                  {slide.keywords?.map((k) => (
                    <span key={k} className="text-[10px] bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full">{k}</span>
                  ))}
                </div>
                {slide.topics?.length > 0 && (
                  <div className="mt-2">
                    <div className="text-xs text-gray-500">Topics</div>
                    <div className="flex flex-wrap gap-1 mt-1 md:justify-end">
                      {slide.topics.map((t) => (
                        <span key={t} className="text-[10px] bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-full">{t}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="mt-4 divide-y">
              {(slide.questions || []).map((q, idx) => {
                const type = (q.type || '').toLowerCase()
                const pillClass = type.includes('fact') ? 'bg-sky-50 text-sky-700' :
                                  type.includes('process') ? 'bg-indigo-50 text-indigo-700' :
                                  type.includes('topic') ? 'bg-amber-50 text-amber-700' :
                                  type.includes('challenge') || type.includes('bias') ? 'bg-rose-50 text-rose-700' :
                                  type.includes('compare') ? 'bg-violet-50 text-violet-700' :
                                  type.includes('application') ? 'bg-teal-50 text-teal-700' :
                                  type.includes('impact') ? 'bg-fuchsia-50 text-fuchsia-700' :
                                  'bg-gray-100 text-gray-700'
                return (
                  <div key={idx} className="py-2 flex items-start gap-3">
                    <span className={`text-[10px] uppercase tracking-wide mt-1 min-w-[84px] text-center px-2 py-0.5 rounded-full ${pillClass}`}>
                      {q.type}
                    </span>
                    <span className="text-sm leading-relaxed">{q.question}</span>
                  </div>
                )
              })}
              {(!slide.questions || slide.questions.length === 0) && (
                <div className="text-sm text-gray-500 py-2">No questions generated for this slide.</div>
              )}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
