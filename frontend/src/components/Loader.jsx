import React from 'react'

export default function Loader({ progress = null, message = 'Processing...' }) {
  // If progress is null, show an infinite circular spinner; otherwise, show a determinate ring.
  const isIndeterminate = progress == null
  const pct = isIndeterminate ? 0 : Math.max(0, Math.min(100, Math.round(progress)))
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
      <div className="w-full max-w-xs rounded-2xl bg-white shadow-lg p-6 flex flex-col items-center">
        <div className="mb-4 text-sm font-medium text-gray-700 text-center">{message}</div>
        {isIndeterminate ? (
          <svg className="animate-spin h-10 w-10 text-blue-600" viewBox="0 0 24 24">
            <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-90" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
          </svg>
        ) : (
          <div className="relative h-16 w-16">
            <svg className="h-16 w-16 -rotate-90" viewBox="0 0 100 100">
              <circle cx="50" cy="50" r="45" stroke="#e5e7eb" strokeWidth="10" fill="none" />
              <circle
                cx="50"
                cy="50"
                r="45"
                stroke="#2563eb"
                strokeWidth="10"
                strokeLinecap="round"
                fill="none"
                strokeDasharray={Math.PI * 2 * 45}
                strokeDashoffset={(1 - pct / 100) * Math.PI * 2 * 45}
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center text-sm font-semibold text-gray-700">{pct}%</div>
          </div>
        )}
      </div>
    </div>
  )
}
