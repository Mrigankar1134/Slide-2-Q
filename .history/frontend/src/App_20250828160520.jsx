import React, { useState } from 'react'
import Upload from './components/Upload'
import Questions from './components/Questions'
import Loader from './components/Loader'

export default function App() {
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [slides, setSlides] = useState([])
  const [aiMeta, setAiMeta] = useState({ ai_used: false, ai_model: null })

  return (
    <div className="min-h-screen app-gradient text-gray-900">
      <header className="sticky top-0 z-40 border-b bg-white/70 backdrop-blur supports-[backdrop-filter]:bg-white/60">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-md bg-gradient-to-tr from-blue-600 to-indigo-500" />
            <h1 className="text-lg sm:text-xl font-semibold tracking-tight">PPTX Question Generator</h1>
          </div>
          <div className="flex items-center gap-3">
            <a
              href="https://github.com"
              target="_blank"
              rel="noreferrer"
              className="text-sm text-gray-600 hover:text-gray-900 hover:underline"
            >
              Docs
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-10">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1">
            <Upload setLoading={setLoading} setSlides={setSlides} setProgress={setProgress} setAiMeta={setAiMeta} />
          </div>
          <div className="lg:col-span-2">
            {loading && (
              <Loader progress={progress < 100 ? progress : null} message={progress < 100 ? 'Uploading...' : 'Generating questions...'} />
            )}
            {!loading && slides.length > 0 && <Questions slides={slides} aiMeta={aiMeta} />}
            {!loading && slides.length === 0 && (
              <div className="bg-white/70 border rounded-xl p-10 text-center text-gray-500">
                Upload a .pptx to generate insightful questions.
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="text-center text-xs text-gray-500 py-8">
        Built with FastAPI & React · © {new Date().getFullYear()}
      </footer>
    </div>
  )
}
