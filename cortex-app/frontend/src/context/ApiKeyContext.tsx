import { createContext, useContext, useState } from "react";
import type { ReactNode } from "react";

import { loadApiKey } from "@/api/key";
import { ApiKeyModal } from "@/components/ApiKeyModal";

interface ApiKeyCtx {
  apiKey: string;
  /** 保存新 key（localStorage + 内存同步）。 */
  saveKey: (value: string) => void;
  /** 打开录入弹窗（401 时由请求方触发）。 */
  openModal: () => void;
}

const Ctx = createContext<ApiKeyCtx>({
  apiKey: "",
  saveKey: () => {},
  openModal: () => {},
});

export function ApiKeyProvider({ children }: { children: ReactNode }) {
  const [apiKey, setApiKey] = useState(loadApiKey());
  const [open, setOpen] = useState(false);

  const value: ApiKeyCtx = {
    apiKey,
    saveKey: (v) => setApiKey(v),
    openModal: () => setOpen(true),
  };

  return (
    <Ctx.Provider value={value}>
      {children}
      <ApiKeyModal
        open={open}
        onClose={() => setOpen(false)}
        onSave={value.saveKey}
        currentKey={apiKey}
      />
    </Ctx.Provider>
  );
}

export function useApiKey() {
  return useContext(Ctx);
}
