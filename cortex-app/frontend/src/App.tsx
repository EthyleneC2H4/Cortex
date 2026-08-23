import { Navigate, Route, Routes } from "react-router-dom";

import { AppLayout } from "@/components/AppLayout";
import { ApiKeyProvider } from "@/context/ApiKeyContext";
import { ThemeProvider } from "@/context/ThemeContext";
import { ToastProvider } from "@/context/ToastContext";
import { ChatPage } from "@/pages/ChatPage";
import { KnowledgePage } from "@/pages/KnowledgePage";
import { StatusPage } from "@/pages/StatusPage";
import { WorkbenchPage } from "@/pages/WorkbenchPage";

export default function App() {
  return (
    <ThemeProvider>
      <ToastProvider>
        <ApiKeyProvider>
          <Routes>
            <Route element={<AppLayout />}>
              <Route index element={<Navigate to="/research" replace />} />
              <Route path="/research" element={<WorkbenchPage />} />
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/knowledge" element={<KnowledgePage />} />
              <Route path="/status" element={<StatusPage />} />
              <Route path="*" element={<Navigate to="/research" replace />} />
            </Route>
          </Routes>
        </ApiKeyProvider>
      </ToastProvider>
    </ThemeProvider>
  );
}
