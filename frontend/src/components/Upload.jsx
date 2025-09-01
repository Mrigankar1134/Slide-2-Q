import React, { useRef, useState } from 'react'
import { uploadPptx } from '../api'

export default function Upload({ setLoading, setSlides, setProgress, setAiMeta }) {
  const inputRef = useRef(null)
  const [fileName, setFileName] = useState('')
  const [error, setError] = useState('')

  const onFileChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.pptx')) {
      setError('Please upload a .pptx file')
      return
    }
    setError('')
    setFileName(file.name)
  }

  const onSubmit = async (e) => {
    e.preventDefault()
    const file = inputRef.current?.files?.[0]
    if (!file) {
      setError('Select a .pptx first')
      return
    }
    setLoading(true)
    setProgress(0)
    setSlides([])
    try {
      const data = await uploadPptx(file, (evt) => {
        if (!evt.total) return
        const pct = Math.round((evt.loaded / evt.total) * 100)
        setProgress(pct)
      }, true)
      // Ensure full progress while backend processes after upload complete
      setProgress(100)
      setSlides(data.slides || [])
      setAiMeta({ ai_used: !!data.ai_used, ai_model: data.ai_model || null })
    } catch (err) {
      console.error(err)
      const msg = err?.response?.data?.error || err?.message || 'Upload failed.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="relative overflow-hidden rounded-2xl border bg-white/80 shadow-sm">
      <div className="absolute -top-24 -left-24 h-48 w-48 rounded-full bg-blue-100 blur-3xl opacity-60" />
      <div className="absolute -bottom-24 -right-24 h-48 w-48 rounded-full bg-indigo-100 blur-3xl opacity-60" />
      <div className="relative p-6">
        <h2 className="text-lg font-semibold mb-2 tracking-tight">Upload your presentation</h2>
        <p className="text-sm text-gray-600 mb-4">We support .pptx files up to 25MB.</p>
        <form onSubmit={onSubmit} className="flex flex-col gap-4">
          <label className="flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-gray-300 bg-white hover:bg-gray-50 transition-colors p-6 cursor-pointer">
            <input
              ref={inputRef}
              onChange={onFileChange}
              type="file"
              accept=".pptx"
              className="hidden"
            />
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" strokeWidth="1.5" stroke="currentColor" className="h-8 w-8 text-blue-600">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5V18a2.25 2.25 0 002.25 2.25h13.5A2.25 2.25 0 0021 18v-1.5m-9 0V6.75m0 8.25l-3-3m3 3l3-3M3.375 7.5h17.25" />
            </svg>
            <div className="text-sm"><span className="font-medium text-gray-900">Click to upload</span> or drag and drop</div>
            <div className="text-xs text-gray-500">.pptx only</div>
          </label>

          {fileName && (
            <div className="text-xs text-gray-600 bg-gray-50 border rounded-md px-3 py-2">Selected: {fileName}</div>
          )}
          {error && <div className="text-sm text-red-600">{error}</div>}

          <div className="flex items-center justify-end gap-3">
            <button
              type="submit"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow hover:from-blue-700 hover:to-indigo-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" strokeWidth="1.5" stroke="currentColor" className="h-4 w-4">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m0 0l6-6m-6 6l-6-6" />
              </svg>
              Generate Questions
            </button>
          </div>
        </form>
      </div>
    </section>
  )
}
