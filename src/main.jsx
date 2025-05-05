import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import GradCAMVisualizer from './GradCAMVisualizer.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <GradCAMVisualizer />
  </StrictMode>,
)
