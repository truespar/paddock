import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
vi.mock('@/lib/chat-title', () => ({ titleGenerator: { cancel: vi.fn() } }))
import { useModelsStore, type ModelInfo } from '@/stores/models'
import { useChatStore } from '@/stores/chat'
import { useRegistryStore } from '@/stores/registry'
import { studioModelHeader } from './studio-model-header'
import { messageStamp, speculationBadge } from './model-name'
import fixture from './studio-model-header.fixture.json'
import webHeader from '../components/layout/AppHeader.vue?raw'
import webComposer from '../components/chat/Composer.vue?raw'
import webInstances from '../components/manage/ServersPanel.vue?raw'
import controller from '../../native-workspace/controller.ts?raw'
import toolbar from '../../../apps/macos/Sources/PaddockUI/WorkspaceToolbar.swift?raw'
import composer from '../../../apps/macos/Sources/PaddockUI/StudioComposerView.swift?raw'
import workspace from '../../../apps/macos/Sources/PaddockUI/WorkspaceView.swift?raw'
import sidebar from '../../../apps/macos/Sources/PaddockUI/StudioHistoryView.swift?raw'
import panels from '../../../apps/macos/Sources/PaddockUI/WorkspacePanels.swift?raw'
import endpoints from '../../../apps/macos/Sources/PaddockUI/EndpointsView.swift?raw'
import conversation from '../../../apps/macos/Sources/PaddockUI/StudioConversationView.swift?raw'
import rendererCommands from '../../../apps/macos/Sources/PaddockUI/StudioRendererCommands.swift?raw'
import modelPicker from '../../../apps/macos/Sources/PaddockUI/StudioComposerModelPicker.swift?raw'
import selection from '../../../apps/macos/Sources/PaddockUI/StudioOptionsView.swift?raw'
import footer from '../../../apps/macos/Sources/PaddockUI/StudioSidebarFooter.swift?raw'

const local: ModelInfo = { id: 'local-qwen', display: 'Qwen 3.8 27B', ownedBy: 'paddock', vendor: 'Alibaba', port: 12481, vision: true, spec: 'MTP', kind: 'chat', status: 'ok' }
const cloud: ModelInfo = { id: 'cloud-qwen', display: 'Cloud Qwen', ownedBy: 'cloud', vendor: 'Alibaba', kind: 'chat', status: 'ok', cloud: { endpoint: 'example', endpointName: 'OpenRouter' } }

describe('shared web/native model header', () => {
  beforeEach(() => {
    vi.stubGlobal('localStorage', { getItem: () => null, setItem: vi.fn(), removeItem: vi.fn() })
    vi.stubGlobal('window', { matchMedia: () => ({ matches: false }) })
    vi.stubGlobal('fetch', vi.fn(async () => Response.json({})))
    setActivePinia(createPinia())
    const models = useModelsStore()
    models.models = [structuredClone(local), structuredClone(cloud)]
    models.currentId = local.id
  })
  it('shares the wire fixture, names, provider hints and badges without making requests', () => {
    expect(studioModelHeader()).toEqual(fixture)
    expect(fetch).not.toHaveBeenCalled()
  })
  it('omits disabled speculation in single and Compare headers, preserving active labels', () => {
    for (const spec of [undefined, null, '', 'off', 'OFF', ' off ', 'false', 'no', 'none', '0', 'disabled', ' DISABLED ']) expect(speculationBadge(spec)).toBe('')
    const models = useModelsStore(), chat = useChatStore()
    models.models[0]!.spec = 'off'
    expect(studioModelHeader().specLabel).toBe('')
    chat.startDraft(local.id)
    chat.active!.compareModels = [local.id, cloud.id]
    expect(studioModelHeader().compareLanes.map(l => l.spec)).toEqual(['', ''])
    for (const active of ['MTP', 'DFlash2', 'MTP+DFlash2']) {
      models.models[0]!.spec = active
      expect(studioModelHeader().specLabel).toBe(active)
      expect(studioModelHeader().compareLanes[0]!.spec).toBe(active)
    }
    expect(fetch).not.toHaveBeenCalled()
  })
  it('filters both live and stopped instance badges and the Compare picker, not only the header', () => {
    for (const value of ['en.r.spec', 'en.c.spec']) {
      expect(webInstances).toContain(`v-if="speculationBadge(${value})"`)
      expect(webInstances).toContain(`{{ speculationBadge(${value}) }}`)
      expect(webInstances).not.toContain(`{{ ${value} }}`)
    }
    expect(webComposer).toContain('v-if="speculationBadge(models.specFor(m.id))"')
    expect(webComposer).not.toContain('{{ models.specFor(m.id) }}')
  })
  it('keeps recorded off out of message badges without substituting the current active mechanism', () => {
    const run = { model: local.id, spec: 'off' } as NonNullable<Parameters<typeof messageStamp>[0]['run']>
    expect(messageStamp({ role: 'assistant', model: local.id, run }).spec).toBe('')
    expect(run.spec).toBe('off') // preserve provenance and explicit settings
    run.spec = ' MTP+DFlash2 '
    expect(messageStamp({ role: 'assistant', model: local.id, run }).spec).toBe('MTP+DFlash2')
  })
  it('does not offer locked cloud credentials and restores the same provider pin after unlock', async () => {
    let ready = false
    vi.stubGlobal('fetch', vi.fn(async (url: string) => Response.json(url === '/api/runners' ? [] : [{
      id: 'fixture-account', name: 'OpenRouter', kind: 'openai-compat',
      baseUrl: 'https://openrouter.ai/api/v1', hasKey: true, credentialReady: ready,
      models: [{ id: 'meta/muse-spark-1.3', provider: 'meta' }],
    }])))
    const models = useModelsStore()
    models.currentId = ''
    await models.refresh()
    expect(models.models[0]).toMatchObject({ id: 'cloud:fixture-account:meta/muse-spark-1.3@meta', status: 'credential-unavailable' })
    expect(studioModelHeader().pickerOptions.filter(o => o.available)).toEqual([])
    ready = true
    await models.refresh()
    expect(models.models[0]!.status).toBe('ok')
    expect(studioModelHeader().pickerOptions.filter(o => o.available).map(o => o.value)).toEqual(['cloud:fixture-account:meta/muse-spark-1.3@meta'])
  })
  it('keeps model startup in Manager rather than the Studio landing page', () => {
    expect(conversation).not.toContain('Start a model')
    expect(conversation).not.toContain('onStart')
    expect(composer).toContain('.padding(.bottom, StudioComposerLayout.bottomInset)')
    expect(composer).toContain('StudioComposerLayout.horizontalControlInset - StudioComposerLayout.contentInset')
    expect(composer).not.toContain('.padding(.bottom, StudioComposerLayout.horizontalControlInset)')
  })
  it('shows the conversation target even when the fleet seat has moved', () => {
    useChatStore().startDraft(cloud.id)
    useModelsStore().currentId = local.id
    const header = studioModelHeader()
    expect(header.currentModel).toBe(cloud.id)
    expect(header.isVision).toBe(false)
    expect(header.specLabel).toBe('')
  })
  it('retains stopped targets with catalog names and marks instead of substituting a runner', () => {
    useChatStore().startDraft('retired-model')
    useRegistryStore().models = [{ id: 'retired-model', display: 'Retired model', vendor: 'IBM', artifacts: [] }] as unknown as ReturnType<typeof useRegistryStore>['models']
    const header = studioModelHeader()
    expect(header.pickerOptions[0]).toMatchObject({ value: 'retired-model', label: 'Retired model', hint: 'not running', vendor: 'IBM', available: false })
    expect(header.pickerOptions.slice(1).map(o => o.value)).toEqual([local.id, cloud.id])
  })
  it('keeps ordered compare names and per-lane speculation, including stopped lanes', () => {
    const chat = useChatStore(); chat.startDraft(local.id)
    chat.active!.compareModels = [cloud.id, local.id, 'stopped']
    const header = studioModelHeader()
    expect(header.comparing).toBe(true)
    expect(header.compareLanes.map(l => [l.id, l.spec])).toEqual([[cloud.id, ''], [local.id, 'MTP'], ['stopped', '']])
  })
  it('offers transcription but not encoders, aligners or unreachable unselected runners', () => {
    const models = useModelsStore()
    models.models = [
      { ...local, id: 'asr', kind: 'transcriber' }, { ...local, id: 'encoder', kind: 'encoder' },
      { ...local, id: 'aligner', kind: 'aligner' }, { ...local, id: 'down', status: 'unreachable' },
    ]
    models.currentId = 'asr'
    expect(studioModelHeader().pickerOptions.map(o => o.value)).toEqual(['asr'])
    models.currentId = ''; models.models = [models.models[1]!]
    expect(studioModelHeader()).toMatchObject({ currentModel: '', pickerOptions: [], soleEncoder: local.display })
  })
  it('keeps shared model data but puts native selection in the composer, per the Bionic reference', () => {
    expect(webHeader).toContain('computed(studioModelHeader)')
    expect(controller).toContain('computed(studioModelHeader)')
    expect(toolbar).not.toContain('StudioHeaderModelView(chat: model.chat)')
    expect(toolbar).not.toContain('WorkspaceModeMenu(')
    // Settings opens from the sidebar footer only (asserted below), not the toolbar.
    expect(toolbar).not.toContain('navigation.showSettings()')
    expect(toolbar).not.toContain('areaPicker')
    expect(composer).toContain('StudioComposerModelPicker(chat: chat')
    expect(modelPicker).toContain('accessibilityIdentifier("composer-model")')
    // The picker opens one searchable selection view; Single and Compare are its two tabs.
    expect(modelPicker).toContain('StudioModelSelectionView(chat: chat')
    expect(selection).toContain('chat.perform("models"')
    expect(selection).toContain('.disabled(model.status != "ok")')
    expect(selection).toContain('Text("Compare").tag(true)')
    expect(composer).not.toContain('StudioCompareView(')
    expect(workspace).not.toContain('StudioActivityRail(')
    expect(workspace).toContain('StudioSidebarFooter(navigation: $model.navigation)')
    expect(footer).toContain('destination(.settings, title: "Settings")')
    expect(workspace).not.toContain('HSplitView {')
    expect(workspace).not.toContain('.workspacePanel(')
    // The native split shell owns the top inset; the SwiftUI layer runs under the titlebar.
    expect(workspace).toContain('.ignoresSafeArea(.container, edges: .top)')
    expect(workspace).toContain('.background(PaddockStyle.sidebar)')
    expect(sidebar).not.toContain('.background(PaddockStyle.sidebar)')
  })
  it('uses one thin full-height sidebar edge and rounded composer controls', () => {
    expect(panels).not.toMatch(/strokeBorder|Divider\(|WorkspaceRule\(/)
    expect(panels).toContain('Rectangle().fill(PaddockStyle.border).frame(width: 1)')
    expect(composer).toContain('in: Capsule()')
    expect(composer).toContain('.contentShape(Circle())')
    expect(workspace.split('private var managerSidebar')[1]!.split('private func navigationRow')[0]).not.toContain('WorkspaceRule()')
    expect(endpoints).not.toContain('WorkspaceRule()')
  })
  it('groups composer actions like Bionic and preserves the web settings sequence', () => {
    const tools = composer.split('private func composerTools')[1]!.split('private func composerActions')[0]!
    const actions = composer.split('private func composerActions')[1]!.split('private func control(')[0]!
    const inOrder = (source: string, labels: string[]) => {
      const positions = labels.map(label => source.indexOf(label))
      expect(positions.every(p => p >= 0)).toBe(true)
      expect(positions).toEqual([...positions].sort((a, b) => a - b))
    }
    // Bionic's left edge begins with attachment, rather than reasoning.
    inOrder(tools, ['"Attach a picture to edit" : "Attach files"', '"Thinking", icon:', 'Button("Web search"',
      '"Tools and connectors", icon:', '"Instructions", icon:', '"Sampling", icon:',
      'accessibilityIdentifier("composer-context")'])
    // Extra Studio settings retain their web ordering, while the microphone
    // moves to the model/send group instead of splitting attachment/settings.
    inOrder(webComposer, ['aria-label="Thinking"', 'aria-label="Web search"',
      'aria-label="Tools for this chat"', 'aria-label="System prompt"', 'aria-label="Sampling"',
      '<ContextMeter'])
    inOrder(actions, ['StudioComposerModelPicker(chat: chat', 'StudioMicrophoneButton(chat: chat',
      'accessibilityIdentifier(responding ? "studio-stop" : "studio-send")'])
    expect(tools).not.toContain('StudioMicrophoneButton(')
    expect(actions).not.toContain('composer-context')
    expect(actions).not.toContain('"Sampling"')
  })
  it('keeps compact settings reachable without duplicating Compare in two menus', () => {
    const tools = composer.split('private func composerTools')[1]!.split('private func composerActions')[0]!
    const overflow = composer.split('private var overflowPresented')[1]!.split('@ViewBuilder')[0]!
    expect(tools).toContain('Button("Thinking…") { panel = .reasoning }')
    expect(tools).toContain('accessibilityIdentifier("composer-more")')
    expect(overflow).toContain('get: { panel?.usesOverflow == true }')
    expect(composer).toContain('case .image: false')
    expect(composer).not.toContain('.compare')
    expect(tools).not.toContain('Button("Compare models…")')
    expect(modelPicker).not.toContain('Compare models')
  })
  it('keeps samples diagnostic-only and offers no web renderer switch', () => {
    expect(conversation).not.toMatch(/Transcript renderer|NativeMarkdownSamples|pickerStyle\(.segmented\)/)
    expect(conversation).toContain('NativeStudioTranscript(')
    expect(conversation).toContain('WorkspaceContentView(session: chat, role: .document)')
    expect(conversation).toContain('WorkspaceContentView(session: chat, role: .graph)')
    expect(rendererCommands).toContain('#if DEBUG')
    expect(rendererCommands).toContain('CommandMenu("Renderer")')
    expect(rendererCommands).not.toContain('model.chat.perform("renderer"')
    expect(rendererCommands).not.toContain('Shared Web')
  })
})
