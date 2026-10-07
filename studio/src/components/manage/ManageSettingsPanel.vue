<script setup lang="ts">
// Manager settings: admin of the BOX, not of chats. Starts
// small deliberately - the export lives here because it downloads the
// MANAGER's database; future manager-level admin (bind address, auth,
// activity retention, data paths) lands here as it arrives.
import { onMounted, ref } from 'vue'
import Icon from '@/components/Icon.vue'
import UpdateCard from './UpdateCard.vue'
import HuggingFaceCard from './HuggingFaceCard.vue'

// Whether this box has a certificate, so the card can say what it is rather
// than offering a page that would only explain why there is nothing to do.
const tlsOn = ref<boolean | null>(null)
onMounted(async () => {
  try {
    const r = await fetch('/tls/info')
    if (r.ok) tlsOn.value = ((await r.json()) as { enabled: boolean }).enabled
  } catch {
    tlsOn.value = null
  }
})
</script>

<template>
  <div class="ms">
    <h1 class="ms__title">Settings</h1>

    <UpdateCard />

    <HuggingFaceCard />

    <section class="ms__card">
      <div class="ms__head"><h2>Data</h2></div>
      <p class="ms__sub">
        The manager's database as a SQLite file - conversations, prompts, settings, and
        per-turn run metrics. API keys are stripped.
      </p>
      <a class="pk-btn" href="/api/export" download="paddock-export.db">
        <Icon name="arrow-down" :size="15" /> Export database
      </a>
    </section>

    <section class="ms__card">
      <div class="ms__head"><h2>Browsing from another device</h2></div>
      <p class="ms__sub">
        <template v-if="tlsOn === false">
          This computer could not set up a certificate, so browsers elsewhere on the network get
          no microphone and no clipboard.
        </template>
        <template v-else>
          Install this computer's certificate on your phone or laptop and the Studio stops
          warning you there.
        </template>
      </p>
      <RouterLink class="pk-btn" :to="{ name: 'trust' }">
        <Icon name="shield" :size="15" /> Trust this computer
      </RouterLink>
    </section>
  </div>
</template>

<style scoped>
.ms {
  max-width: var(--pk-panel-width);
  width: 100%;
  margin: 0 auto;
  /* The CONTAINER owns the rhythm, not each card. Cards used to carry their
     own margin-bottom, which works right up until somebody adds one that does
     not know the convention - which is exactly how the update card landed
     flush against its neighbour. A gap cannot be forgotten. */
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.ms__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
}
.ms__card {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 16px 20px 20px;
}
.ms__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}
.ms__head h2 {
  font-size: var(--pk-font-size-base);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.ms__sub {
  margin: 0 0 12px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-muted);
  line-height: 1.5;
}
</style>
