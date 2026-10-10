//! Render LightOnOCR-3's real chat template (the checkpoint's
//! `chat_template.jinja`, which our GGUF conversion carries byte-identical in
//! `tokenizer.chat_template`) through the `ocr` resolution and our minijinja
//! pipeline, and compare byte-for-byte against transformers 5.13's
//! `apply_chat_template` on the same checkpoint for the two prompts the model
//! was trained on.
//!
//! The expected strings are that reference output, verbatim. Thinking is off
//! by the template's own default (`enable_thinking` undefined renders the
//! empty think block), which is also how LightOn trained and evaluated. Our
//! renderer DEFINES `enable_thinking` true for every model (llama.cpp's
//! default), which flips this template into thinking - so the serving path
//! adds `enable_thinking: false` for this checkpoint
//! (`ServingModel::template_defaults`), and the render below passes the same.
//!
//! Skips cleanly when the checkpoint is absent (`LIGHTONOCR3_DIR`, default
//! E:/paddock/models/LightOnOCR-3-4B).
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

use paddock_runner::{chat_template, lighton_ocr};
use serde_json::{Value, json};

const PLAIN: &str = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n\
                     <|im_start|>assistant\n<think>\n\n</think>\n\n";
const GROUNDING: &str = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>grounding\
                         <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

fn template() -> Option<String> {
    let dir = std::env::var("LIGHTONOCR3_DIR")
        .unwrap_or_else(|_| "E:/paddock/models/LightOnOCR-3-4B".to_owned());
    std::fs::read_to_string(std::path::Path::new(&dir).join("chat_template.jinja")).ok()
}

/// One user turn the way an OpenAI client sends a page: an `image_url`
/// data-URI part, then any text.
fn page(text: Option<&str>) -> Vec<Value> {
    let mut parts = vec![json!({"type": "image_url",
                                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}})];
    if let Some(t) = text {
        parts.push(json!({"type": "text", "text": t}));
    }
    chat_template::normalize_messages(&[json!({"role": "user", "content": parts})])
}

fn render(template: &str, mut msgs: Vec<Value>, mode: Option<lighton_ocr::LoMode>) -> String {
    lighton_ocr::resolve(&mut msgs, mode, 1).unwrap();
    // what template_defaults adds for this checkpoint when the caller sent none
    let serving = json!({"enable_thinking": false});
    chat_template::render(template, &msgs, None, Some(&serving)).expect("render")
}

/// The trap the serving default exists for: rendered with the renderer's own
/// default, this template opens a think block the model never trained on.
#[test]
fn the_renderer_default_alone_would_open_thinking() {
    let Some(t) = template() else {
        eprintln!("LightOnOCR-3 checkpoint missing - skipping");
        return;
    };
    let raw = chat_template::render(&t, &page(None), None, None).expect("render");
    assert!(raw.ends_with("<|im_start|>assistant\n<think>\n"), "{raw:?}");
}

#[test]
fn the_two_trained_prompts_render_byte_identical_to_the_processor() {
    let Some(t) = template() else {
        eprintln!("LightOnOCR-3 checkpoint missing - skipping");
        return;
    };
    use lighton_ocr::LoMode::{Grounding, Plain};
    // what the official client sends, with no `ocr` object
    assert_eq!(
        render(&t, page(None), None),
        PLAIN,
        "plain: the image alone"
    );
    assert_eq!(render(&t, page(Some("grounding")), None), GROUNDING);
    // the same two through the request object, whatever text came with it
    assert_eq!(
        render(&t, page(Some("transcribe this")), Some(Plain)),
        PLAIN
    );
    assert_eq!(
        render(&t, page(Some("transcribe this")), Some(Grounding)),
        GROUNDING
    );
    assert_eq!(render(&t, page(None), Some(Grounding)), GROUNDING);
}
