<script setup lang="ts">
// The network key gate: a browser on another machine hits
// a keyed manager, every /api call 401s, and without this the whole Studio
// just rendered empty. The key is on the manager's console banner (or a
// stored Studio API key); a successful login turns it into an HttpOnly
// session cookie and the page reloads into a working Studio.
import { ref } from 'vue'
import Icon from '@/components/Icon.vue'

const key = ref('')
const err = ref('')
const busy = ref(false)

async function unlock(): Promise<void> {
  if (!key.value.trim() || busy.value) return
  busy.value = true
  err.value = ''
  try {
    const res = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: key.value.trim() }),
    })
    if (res.ok) {
      location.reload()
      return
    }
    err.value =
      res.status === 401
        ? 'That key does not open this paddock.'
        : res.status === 403
          ? 'That key is for the API only. The Studio needs an admin key.'
          : `Login failed (${res.status}).`
  } catch {
    err.value = 'Could not reach the manager.'
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div class="keygate">
    <form class="keygate__card" @submit.prevent="unlock">
      <Icon name="shield" :size="22" class="keygate__icon" />
      <h1 class="keygate__title">This Paddock is locked</h1>
      <p class="keygate__sub">
        You are connecting over the network. Enter the API key from the manager's console.
      </p>
      <input
        v-model="key"
        type="password"
        class="keygate__input"
        placeholder="pk-..."
        autocomplete="off"
        autofocus
      />
      <button class="keygate__btn" type="submit" :disabled="!key.trim() || busy">
        {{ busy ? 'Checking...' : 'Unlock' }}
      </button>
      <p v-if="err" class="keygate__err">{{ err }}</p>
    </form>
  </div>
</template>

<style scoped>
.keygate {
  position: fixed;
  inset: 0;
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--pk-bg-base);
}
.keygate__card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  width: min(360px, calc(100vw - 48px));
  padding: 28px 26px;
  background: var(--pk-bg-surface);
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
}
.keygate__icon {
  color: var(--pk-text-muted);
}
.keygate__title {
  margin: 0;
  font-size: var(--pk-font-size-lg);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.keygate__sub {
  margin: 0 0 6px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  text-align: center;
}
.keygate__input {
  width: 100%;
  padding: 8px 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-sm);
  background: var(--pk-bg-base);
  color: var(--pk-text-primary);
  font-size: var(--pk-font-size-sm);
  font-family: var(--pk-font-mono, monospace);
}
.keygate__input:focus {
  outline: none;
  border-color: var(--pk-accent);
}
.keygate__btn {
  width: 100%;
  padding: 8px 10px;
  border: 0;
  border-radius: var(--pk-radius-sm);
  background: var(--pk-accent);
  color: var(--pk-text-on-accent, #fff);
  font-size: var(--pk-font-size-sm);
  font-weight: 600;
  cursor: pointer;
}
.keygate__btn:disabled {
  opacity: 0.55;
  cursor: default;
}
.keygate__err {
  margin: 2px 0 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-status-error);
}
</style>
