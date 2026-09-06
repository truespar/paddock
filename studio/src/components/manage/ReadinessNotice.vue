<script setup lang="ts">
// The verdict about this computer, in the Manager's own card idiom - the same
// construction as the Manager/Studio blocks it sits beside on first run
// (bordered surface card, uppercase head with an icon, plain sentence), with
// the border carrying the state exactly as `srv__area--here` already uses it
// to mark "you are here". Nothing invented: an unfamiliar shape next to two
// familiar ones is what made the earlier version read as somebody else's
// design.
//
// There was a fourth state here, 'needs-setup', with a progress bar for a
// one-time download of NVIDIA's maths libraries. Paddock ships and fetches
// none, so the only thing a supported card can still be waiting
// on is its own driver - which `driver-too-old` already says, with a link.
import { computed } from 'vue'
import { useReadinessStore } from '@/stores/readiness'
import Icon from '@/components/Icon.vue'

const rd = useReadinessStore()

const card = computed(() => rd.info?.card || 'this graphics card')
const windows = computed(() => rd.info?.os === 'windows')

const HEAD: Record<string, { icon: string; label: string }> = {
  'no-card': { icon: 'x-circle', label: 'Not supported' },
  untested: { icon: 'alert-triangle', label: 'Not tested yet' },
  'driver-too-old': { icon: 'alert-triangle', label: 'Needs attention' },
}
const head = computed(() => HEAD[rd.notice?.state ?? ''] ?? HEAD['no-card'])
</script>

<template>
  <section v-if="rd.notice" class="rn" :class="`rn--${rd.notice.state}`">
    <p class="rn__hd"><Icon :name="head.icon" :size="14" /> {{ head.label }}</p>

    <template v-if="rd.notice.state === 'driver-too-old'">
      <p class="rn__txt">
        <strong>Update your graphics driver.</strong>
        {{ card }} can run models - its driver is behind what Paddock needs. Updating takes a
        couple of minutes.
      </p>
      <div class="rn__acts">
        <a
          class="pk-btn pk-btn--primary"
          href="https://www.nvidia.com/en-us/drivers/"
          target="_blank"
          rel="noreferrer"
        >
          <Icon name="external-link" :size="14" /> Get the latest driver
        </a>
        <span class="rn__hint">
          {{ windows ? 'The NVIDIA app can do it for you.' : "Or your distribution's driver package." }}
        </span>
      </div>
    </template>

    <template v-else-if="rd.notice.state === 'untested'">
      <p class="rn__txt">
        <strong>{{ card }} hasn't been tested yet.</strong>
        Models will run on it, but Paddock hasn't measured this card, so speed and stability are
        unverified - every server started here says so in its log.
      </p>
    </template>

    <template v-else>
      <p class="rn__txt">
        <strong>Models can't run on this computer.</strong>
        Paddock runs models on an NVIDIA graphics card and this machine doesn't have one it can
        use. Everything else works: connect a cloud model and chat, compare and use tools as
        normal.
      </p>
      <div class="rn__acts">
        <RouterLink class="pk-btn pk-btn--primary" :to="{ name: 'cloud' }">
          <Icon name="cloud" :size="14" /> Add a cloud model
        </RouterLink>
      </div>
      <p class="rn__alt">
        To use this machine's own hardware,
        <a href="https://ollama.com/download" target="_blank" rel="noreferrer">Ollama</a> and
        <a href="https://github.com/ggml-org/llama.cpp" target="_blank" rel="noreferrer">llama.cpp</a>
        support a wider range of computers than Paddock does.
      </p>
    </template>

    <RouterLink class="rn__more" :to="{ name: 'gpus' }">
      Which graphics cards run models <Icon name="chevron-right" :size="13" />
    </RouterLink>
  </section>
</template>

<style scoped>
/* the srv__area card, with the border carrying the state */
.rn {
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 14px 16px;
  width: 100%;
  text-align: left;
}
.rn--no-card {
  border-color: var(--pk-status-error);
}
.rn--untested,
.rn--driver-too-old {
  border-color: var(--pk-status-warning);
}

/* srv__area-hd, tinted by state */
.rn__hd {
  display: flex;
  align-items: center;
  gap: 7px;
  margin: 0 0 6px;
  font-size: var(--pk-font-size-xs);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--pk-text-primary);
}
.rn--no-card .rn__hd {
  color: var(--pk-status-error);
}
.rn--untested .rn__hd,
.rn--driver-too-old .rn__hd {
  color: var(--pk-status-warning);
}

/* srv__area-txt */
.rn__txt {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.5;
}
.rn__txt strong {
  color: var(--pk-text-primary);
}
.rn__acts {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 12px;
}
.rn__hint {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.rn__alt {
  margin: 12px 0 0;
  font-size: var(--pk-font-size-xs);
  line-height: 1.5;
  color: var(--pk-text-muted);
}
.rn__alt a {
  color: var(--pk-text-secondary);
  text-decoration: underline;
}
.rn__more {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-top: 12px;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-accent);
}
.rn__more:hover {
  text-decoration: underline;
}
</style>
