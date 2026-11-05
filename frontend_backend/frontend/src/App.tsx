import { lazy, Suspense } from 'react'
import { MotionConfig } from 'motion/react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { AppShell } from './layout/AppShell'

const OperationsPage = lazy(() => import('./screens/operations/OperationsPage').then((module) => ({ default: module.OperationsPage })))
const TrainingPage = lazy(() => import('./screens/training/TrainingPage').then((module) => ({ default: module.TrainingPage })))
const PrepareRunPage = lazy(() => import('./screens/prepare/PrepareRunPage').then((module) => ({ default: module.PrepareRunPage })))
const ModelsPage = lazy(() => import('./screens/models/ModelsPage').then((module) => ({ default: module.ModelsPage })))
const RiskSettingsPage = lazy(() => import('./screens/risk/RiskSettingsPage').then((module) => ({ default: module.RiskSettingsPage })))

function RoutePending() {
  return <div className="route-pending" role="status">Loading workspace view…</div>
}

export default function App() {
  return (
    <MotionConfig reducedMotion="user">
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Suspense fallback={<RoutePending />}><OperationsPage /></Suspense>} />
          <Route path="training" element={<Suspense fallback={<RoutePending />}><TrainingPage /></Suspense>} />
          <Route path="prepare" element={<Suspense fallback={<RoutePending />}><PrepareRunPage /></Suspense>} />
          <Route path="models" element={<Suspense fallback={<RoutePending />}><ModelsPage /></Suspense>} />
          <Route path="risk" element={<Suspense fallback={<RoutePending />}><RiskSettingsPage /></Suspense>} />
          <Route path="*" element={<Suspense fallback={<RoutePending />}><OperationsPage /></Suspense>} />
        </Route>
      </Routes>
    </BrowserRouter>
    </MotionConfig>
  )
}
