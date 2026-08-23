/** API Key 存取：localStorage 持久化，client 与 ApiKeyProvider 共用。 */

const STORAGE_KEY = "cortex-api-key";

export function loadApiKey(): string {
  try {
    return localStorage.getItem(STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

export function saveApiKey(value: string): void {
  try {
    if (value) localStorage.setItem(STORAGE_KEY, value);
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* 隐私模式等场景下静默失败 */
  }
}
