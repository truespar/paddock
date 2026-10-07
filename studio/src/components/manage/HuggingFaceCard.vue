<script setup lang="ts">
// The Hugging Face token gated downloads carry. Some models (SAM 3) are
// gated by their makers: the user accepts the licence on Hugging Face with
// their own account and downloads with their own token, which the manager
// keeps and the Studio never reads back.
import { onMounted, ref } from 'vue'
import Icon from '@/components/Icon.vue'
import TextInput from '@/components/ui/TextInput.vue'
import { getHfToken, hfTokenSourceLabel, removeHfToken, saveHfToken, type HfTokenStatus } from '@/lib/huggingface'

const status = ref<HfTokenStatus | null>(null)
const draft = ref('')
const busy = ref(false)
const error = ref('')
const user = ref<string | null>(null)

onMounted(async () => {
  try {
    status.value = await getHfToken()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
})

async function save(): Promise<void> {
  busy.value = true
  error.value = ''
  try {
    const s = await saveHfToken(draft.value.trim())
    status.value = s
    user.value = s.user ?? null
    draft.value = ''
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

async function remove(): Promise<void> {
  busy.value = true
  error.value = ''
  try {
    status.value = await removeHfToken()
    user.value = null
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <section class="hf">
    <div class="hf__head">
      <h2>Hugging Face</h2>
      <span v-if="status" class="hf__state" :class="{ 'hf__state--on': status.configured }">
        <Icon :name="status.configured ? 'check-circle' : 'lock'" :size="14" />
        {{ hfTokenSourceLabel(status) }}<template v-if="user"> - signed in as {{ user }}</template>
      </span>
    </div>
    <p class="hf__sub">
      Gated models download with your own token, after you accept their licence on Hugging Face.
    </p>
    <div class="hf__row">
      <TextInput
        v-model="draft"
        type="password"
        reveal
        block
        placeholder="hf_..."
        :disabled="busy"
        @keydown.enter="draft.trim() && save()"
      />
      <button class="pk-btn" :disabled="busy || !draft.trim()" @click="save">
        <Icon name="save" :size="15" /> Save
      </button>
      <button
        v-if="status?.source === 'saved'"
        class="pk-btn pk-btn--danger"
        :disabled="busy"
        @click="remove"
      >
        <Icon name="trash" :size="15" /> Remove
      </button>
    </div>
    <p v-if="error" class="hf__err" role="alert">{{ error }}</p>
    <a
      class="hf__link"
      href="https://huggingface.co/settings/tokens"
      target="_blank"
      rel="noopener noreferrer"
    >
      <Icon name="external-link" :size="13" /> Create a read token
    </a>
  </section>
</template>

<style scoped>
.hf {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 16px 20px 20px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.hf__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.hf__head h2 {
  font-size: var(--pk-font-size-base);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.hf__state {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
}
.hf__state--on {
  color: var(--pk-text-secondary);
}
.hf__sub {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
  line-height: 1.5;
}
.hf__row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.hf__err {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-danger);
}
.hf__link {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  align-self: flex-start;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-accent);
}
</style>
