// The Hugging Face token gated downloads carry (SAM 3: Meta gates the weights
// behind its licence). The manager keeps it; the Studio only ever learns
// whether one is in reach and where it came from - never the token itself.

/** where the manager found the token */
export type HfTokenSource = 'saved' | 'environment' | 'login'

export interface HfTokenStatus {
  configured: boolean
  source: HfTokenSource | null
  /** the account Hugging Face says the token belongs to (on save only) */
  user?: string | null
}

async function readStatus(r: Response): Promise<HfTokenStatus> {
  if (!r.ok) {
    let msg = `the manager answered ${r.status}`
    try {
      const body = (await r.json()) as { error?: { message?: string } }
      if (body.error?.message) msg = body.error.message
    } catch {
      // not JSON - keep the status line
    }
    throw new Error(msg)
  }
  return (await r.json()) as HfTokenStatus
}

export async function getHfToken(): Promise<HfTokenStatus> {
  return readStatus(await fetch('/api/huggingface/token'))
}

export async function saveHfToken(token: string): Promise<HfTokenStatus> {
  return readStatus(
    await fetch('/api/huggingface/token', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    }),
  )
}

export async function removeHfToken(): Promise<HfTokenStatus> {
  return readStatus(await fetch('/api/huggingface/token', { method: 'DELETE' }))
}

/** One line for where a token came from. */
export function hfTokenSourceLabel(s: HfTokenStatus): string {
  if (!s.configured) return 'No token'
  if (s.source === 'saved') return 'Token saved'
  if (s.source === 'environment') return 'Using HF_TOKEN from the environment'
  return 'Using your Hugging Face login (hf auth login)'
}
