<script setup lang="ts">
// The questions of a camera read, one line each: a type, the question, and -
// for a choice or a score - its answers as one comma-separated field. The
// full row (ids, what yes and no mean, conditions, option descriptions) stays
// in the Text mode's editor; this list edits the same questions, so a set
// built in either mode works in both, and nothing typed here is lost there.
import { computed, ref } from 'vue'
import Icon from '@/components/Icon.vue'
import Select, { type SelectOption } from '@/components/ui/Select.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import { READ_TYPES, type ReadOption, type ReadQuestion, type ReadType } from '@/lib/reads'

const props = defineProps<{
  questions: ReadQuestion[]
  /** a row's problem, by question key - shown under that row */
  errors: Record<string, string | undefined>
  /** the types the running model reads; empty = all */
  types: string[]
}>()
const emit = defineEmits<{
  (e: 'patch', key: string, patch: Partial<ReadQuestion>): void
  (e: 'add', type: ReadType): void
  (e: 'remove', index: number): void
}>()

const typeOptions = computed<SelectOption[]>(() =>
  READ_TYPES.filter((t) => !props.types.length || props.types.includes(t.value)).map((t) => ({
    value: t.value,
    label: t.label,
  })),
)

/** The answers of a choice or a score as the one field shows them. */
function answersText(q: ReadQuestion): string {
  const names = q.type === 'choice' ? q.options.map((o) => o.name) : q.levels
  return names.filter((n) => n.trim()).join(', ')
}
/** The field back into options or levels. A choice keeps each surviving
 *  option's description (written in the Text mode), matched by name. */
function setAnswers(q: ReadQuestion, text: string): void {
  const names = text.split(',').map((n) => n.trim())
  // a trailing comma is someone about to type the next answer
  const kept = names.filter((n, i) => n || i === names.length - 1)
  if (q.type === 'choice') {
    const was = new Map(q.options.map((o) => [o.name.trim(), o.description]))
    const options: ReadOption[] = kept.map((name) => ({ name, description: was.get(name) ?? '' }))
    emit('patch', q.key, { options })
  } else {
    emit('patch', q.key, { levels: kept })
  }
}
function answersHint(q: ReadQuestion): string {
  return q.type === 'choice' ? 'The answers, separated by commas: peace, thumbs up, none' : 'The levels, lowest first: low, medium, high'
}

const inputs = ref<HTMLInputElement[]>([])
/** The cursor into the first question with no text yet, else the first. */
function focusFirstEmpty(): void {
  const i = props.questions.findIndex((q) => !q.instructions.trim())
  const el = inputs.value[i >= 0 ? i : 0]
  el?.focus()
  el?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
}
defineExpose({ focusFirstEmpty })
</script>

<template>
  <div class="lq">
    <div v-for="(q, i) in questions" :key="q.key" class="lq__row">
      <div class="lq__line">
        <Select
          class="lq__type"
          :model-value="q.type"
          :options="typeOptions"
          @update:model-value="emit('patch', q.key, { type: String($event) as ReadType })"
        />
        <input
          :ref="(el) => { if (el) inputs[i] = el as HTMLInputElement }"
          class="pk-input lq__q"
          :value="q.instructions"
          :placeholder="i === 0 ? 'Ask something about what the camera sees' : 'Another question'"
          :aria-label="`Question ${i + 1}`"
          @input="emit('patch', q.key, { instructions: ($event.target as HTMLInputElement).value })"
        />
        <Tooltip label="Remove the question">
          <button
            class="pk-icon-btn lq__x"
            type="button"
            :aria-label="`Remove question ${i + 1}`"
            :disabled="questions.length === 1"
            @click="emit('remove', i)"
          >
            <Icon name="x" :size="14" />
          </button>
        </Tooltip>
      </div>
      <input
        v-if="q.type !== 'noul'"
        class="pk-input lq__answers"
        :value="answersText(q)"
        :placeholder="answersHint(q)"
        :aria-label="`Answers of question ${i + 1}`"
        @input="setAnswers(q, ($event.target as HTMLInputElement).value)"
      />
      <p v-if="errors[q.key]" class="lq__err" role="alert">{{ errors[q.key] }}</p>
    </div>
    <div class="lq__add">
      <button class="pk-btn pk-btn--sm pk-btn--ghost" type="button" @click="emit('add', 'noul')">
        <Icon name="plus" :size="13" /> Yes / no
      </button>
      <button class="pk-btn pk-btn--sm pk-btn--ghost" type="button" @click="emit('add', 'choice')">
        <Icon name="plus" :size="13" /> Choice
      </button>
    </div>
  </div>
</template>

<style scoped>
.lq {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.lq__row {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-base);
}
.lq__line {
  display: flex;
  align-items: center;
  gap: 8px;
}
.lq__type {
  flex: none;
}
.lq__q {
  flex: 1;
  min-width: 0;
}
.lq__answers {
  margin-left: 0;
}
.lq__x {
  flex: none;
  width: 30px;
  height: 30px;
}
.lq__err {
  margin: 0;
  color: var(--pk-status-warning);
  font-size: var(--pk-font-size-sm);
}
.lq__add {
  display: flex;
  gap: 6px;
}
</style>
