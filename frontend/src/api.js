import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
})

export async function uploadPptx(file, onUploadProgress, useAI = false) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post(`/generate-questions?use_ai=${useAI ? 'true' : 'false'}`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress,
  })
  return data
}

export async function health() {
  const { data } = await api.get('/health')
  return data
}
