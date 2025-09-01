import axios from 'axios'

// In production (Netlify), default to same-origin '/api' and let Netlify proxy
// to the Render backend via _redirects. Override with VITE_API_BASE_URL when needed.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

const api = axios.create({
  baseURL: API_BASE_URL,
})

export async function uploadPptx(file, onUploadProgress, useAI = false) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post(`/generate-questions?use_ai=${useAI ? 'true' : 'false'}`, form, {
    // Do NOT set Content-Type manually; let the browser add the boundary
    onUploadProgress,
  })
  return data
}

export async function health() {
  const { data } = await api.get('/health')
  return data
}
