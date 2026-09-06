import { createRouter, createWebHistory } from 'vue-router'
import { useReadinessStore } from '@/stores/readiness'
import AppShell from '@/components/layout/AppShell.vue'
import ChatView from '@/components/chat/ChatView.vue'
import ServersPanel from '@/components/manage/ServersPanel.vue'
import ServerNewPick from '@/components/manage/ServerNewPick.vue'
import ManageSettingsPanel from '@/components/manage/ManageSettingsPanel.vue'
import ServerForm from '@/components/manage/ServerForm.vue'
import ServerDetail from '@/components/manage/ServerDetail.vue'
import ApiReference from '@/components/manage/ApiReference.vue'
import InstrumentPanel from '@/components/manage/InstrumentPanel.vue'
import GpuSupportPanel from '@/components/manage/GpuSupportPanel.vue'
import TrustPanel from '@/components/manage/TrustPanel.vue'
import PromptsPanel from '@/components/prompts/PromptsPanel.vue'
import EmbeddingsPanel from '@/components/embeddings/EmbeddingsPanel.vue'
import CloudPanel from '@/components/cloud/CloudPanel.vue'
import ConnectorsPanel from '@/components/connectors/ConnectorsPanel.vue'
import SettingsPanel from '@/components/settings/SettingsPanel.vue'

// Two areas, never mixed:
//
//   /manage - the MANAGER: models (the fleet - start/stop/pin, with the
//             catalog inside the start flow) and instrument (activity / GPU /
//             logs). The landing area - you start models here first.
//   /studio - the STUDIO: chats, tools, prompts, settings - a CLIENT of the
//             servers the manager runs.
//
// The unit of thought differs per area (endpoint vs conversation), and every
// feature lives with the lifetime it belongs to. Route NAMES are stable across
// this restructure so `router.push({ name })` call sites did not move.
const router = createRouter({
  history: createWebHistory(),
  routes: [
    // The manager is the front door: you can't chat with a server that isn't
    // running, so the fleet comes first.
    { path: '/', redirect: { name: 'servers' } },
    {
      path: '/manage',
      component: AppShell,
      children: [
        // Everything is a proper route - no modals for real workflows, and
        // the vocabulary is MODELS you START, never "servers" you "deploy"
        // /manage/models is the fleet (the Manager's
        // home), models/start picks one (vendor blocks + comparison - there
        // is no separate catalog page; starting downloads what's missing),
        // models/start/:model is the workload form, models/:port the detail.
        // Route NAMES kept from the servers era so push({ name }) sites and
        // area checks did not move.
        { path: '', redirect: '/manage/models' },
        { path: 'models', name: 'servers', component: ServersPanel },
        { path: 'models/start', name: 'server-new', component: ServerNewPick },
        { path: 'models/start/:model', name: 'server-new-config', component: ServerForm },
        { path: 'models/:port(\\d+)', name: 'server-detail', component: ServerDetail },
        { path: 'models/:port(\\d+)/edit', name: 'server-edit', component: ServerForm },
        // The endpoint's API reference: the runner's /openapi.json rendered
        // Per-port because the contract is a per-endpoint fact.
        { path: 'models/:port(\\d+)/api', name: 'server-api', component: ApiReference },
        // pre-rename URLs
        { path: 'servers', redirect: '/manage/models' },
        { path: 'servers/new', redirect: '/manage/models/start' },
        { path: 'servers/new/:model', redirect: (to) => `/manage/models/start/${String(to.params.model)}` },
        { path: 'servers/:port(\\d+)', redirect: (to) => `/manage/models/${String(to.params.port)}` },
        { path: 'servers/:port(\\d+)/edit', redirect: (to) => `/manage/models/${String(to.params.port)}/edit` },
        // Tabs are routes too; ?port= narrows to one server (the server
        // rows deep-link here).
        { path: 'instrument', redirect: '/manage/instrument/usage' },
        {
          path: 'instrument/:tab(usage|activity|gpu|cache|logs)',
          name: 'instrument',
          component: InstrumentPanel,
        },
        // Which cards run models. Its own route because it is a page people
        // link to and search, not a fold-out inside a notice - it wants a
        // separate url with a real, clear table.
        { path: 'gpus', name: 'gpus', component: GpuSupportPanel },
        // Installing this box's certificate. Its own URL because you open it
        // on the device that needs it - the phone, the other laptop - so it
        // has to be something you can type or send, not a panel nested inside
        // settings you reached from here.
        { path: 'trust', name: 'trust', component: TrustPanel },
        // Manager-level admin (box concerns: export today; bind/auth/
        // retention as they arrive). Chat preferences stay in /studio.
        { path: 'settings', name: 'manage-settings', component: ManageSettingsPanel },
      ],
    },
    {
      path: '/studio',
      component: AppShell,
      children: [
        // `/studio` is the start page: a centred composer on a fresh draft.
        // Sending lands you in chat/:id. Same component - a chat with no
        // messages is a state of the chat surface, not a separate screen.
        { path: '', name: 'home', component: ChatView },
        // "New chat" keeps the history sidebar in place (static segment
        // outranks `chat/:id?`).
        { path: 'chat/new', name: 'chat-new', component: ChatView },
        { path: 'chat/:id?', name: 'chat', component: ChatView },
        // Connectors: the personal MCP library - servers the
        // user tries per chat, riding inside the request. A model's own tools
        // (the endpoint contract every API client sees) stay on its
        // Start/Edit page in the Manager - two tiers, deliberately.
        { path: 'connectors', name: 'connectors', component: ConnectorsPanel },
        { path: 'mcp', redirect: '/studio/connectors' },
        // Try a running encoder hands-on - the Studio's "you can try
        // everything you start" promise for non-chat models. Named for what
        // it does; the Studio itself is the playground.
        { path: 'embeddings', name: 'embeddings', component: EmbeddingsPanel },
        { path: 'playground', redirect: '/studio/embeddings' },
        // Transcription used to be a page here. It is a CONVERSATION now
        // a speech model holds a lane in the chat like any other
        // model, the composer switches to audio, and the turn persists with
        // its clip, timings and tokens. Old links land in the chat, which is
        // where transcribing happens.
        { path: 'transcribe', redirect: '/studio' },
        // Cloud models: BYO-key provider endpoints whose models join the
        // pickers - Studio-side because they exist to be CHATTED with (the
        // Manager area stays models you start on this box).
        { path: 'cloud', name: 'cloud', component: CloudPanel },
        { path: 'prompts', name: 'prompts', component: PromptsPanel },
        { path: 'settings', name: 'settings', component: SettingsPanel },
      ],
    },
    // Legacy chat-first paths (bookmarks, older sessions) land in their new area.
    { path: '/models', redirect: '/manage/models' },
    { path: '/mcp', redirect: '/studio/mcp' },
    { path: '/prompts', redirect: '/studio/prompts' },
    { path: '/settings', redirect: '/studio/settings' },
    {
      path: '/chat/:id?',
      redirect: (to) => `/studio/chat/${String(to.params.id ?? '')}`,
    },
    // Unknown paths fall back to the front door.
    { path: '/:pathMatch(.*)*', redirect: '/' },
  ],
})

// Pages that only exist because this box can run models. ServersPanel already
// hides the "Start your first model" button when the GPU probe says no, but
// the URLs behind it stayed open - you could type /manage/models/start, pick a
// model, configure it, and only find out at the end. Same
// for the embeddings playground: an encoder runs on the local GPU or not at
// all. A control that cannot work must not be offered, and that has to include
// its address.
//
// /manage/models itself stays reachable deliberately - it is where the readiness
// notice lives, so "why can't I start anything" has somewhere to be answered.
const NEEDS_A_GPU = new Set(['server-new', 'server-new-config', 'embeddings'])

router.beforeEach(async (to) => {
  if (!NEEDS_A_GPU.has(String(to.name))) return true
  // Called here and not at module scope: the store needs an active pinia, and
  // this module is evaluated while main.ts is still assembling the app. By the
  // time a guard runs, pinia is installed.
  const readiness = useReadinessStore()
  await readiness.ensureLoaded()
  if (!readiness.blocked) return true
  return to.name === 'embeddings' ? { name: 'home' } : { name: 'servers' }
})

export default router
