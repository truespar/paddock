# Provider artwork

Reused from Paddock web Studio on 2026-09-13. These are provider identities,
not Paddock artwork or an endorsement. Marks retain their respective owners.
No new runtime icon package or network dependency is introduced.

`studio/src/components/manage/VendorLogo.vue` supplies Qwen (Alibaba), OpenAI,
Poolside, IBM, NB AI-Lab and CoRal Project (Alexandra Institute). Vue bindings
become static SVG attributes; Poolside IDs are file-local. Path geometry and
aspect ratios are unchanged. See that component for original asset attribution.
Google and KBLab are exact copies of `studio/src/assets/google-g.svg` and
`studio/src/assets/kblab.svg`. Native views supply monochrome ink; KBLab retains
its original luminance, since an alpha template would erase its internal art.

Prism ML (2026-09-21) uses the emblem from the official prismml.com header
wordmark, shared with `VendorLogo.vue`. The first four paths are unchanged;
the viewBox is cropped to the emblem and native views supply monochrome ink.
Source: https://cdn.prod.website-files.com/697a3312d33c2cc715ec3899/69961847735f3ec32aafc18e_prism-logo.svg
The mark remains Prism ML's property.

The following SVGs are copied from Studio's installed simple-icons 16.28.0.
The package's full license and disclaimer are bundled as
`SimpleIcons-LICENSE.md` and `SimpleIcons-DISCLAIMER.md`. Package CC0 is not a
blanket license for the brands; source and per-icon metadata follow.

- Anthropic: https://www.anthropic.com
- DeepSeek: https://www.deepseek.com
- Meta: https://www.meta.com — guidelines: https://www.facebook.com/brand/resources/meta/company-brand
- Mistral AI: https://chat.mistral.ai
- NVIDIA: https://www.nvidia.com/en-us
- Xiaomi: https://www.mi.com/global
- Moonshot AI: https://www.moonshot.cn
- Perplexity: https://www.perplexity.ai
- Baidu: https://www.baidu.com
- ByteDance: https://www.bytedance.com
- PaddlePaddle: https://www.paddlepaddle.org.cn/en
- MiniMax: https://github.com/MiniMax-AI/MiniMax-01/blob/57cf223b177e99636c7711a0f179e9fdc9c38e8a/figures/minimax.svg — license metadata: {"type":"custom","url":"https://github.com/simple-icons/simple-icons/pull/13982#issuecomment-3531627803"}
- Hugging Face: https://huggingface.co/brand — guidelines: https://huggingface.co/brand
- OpenRouter: https://openrouter.ai
- Cloudflare (2026-10-03): https://www.cloudflare.com/logo/ — guidelines: https://www.cloudflare.com/trademark/

Search-provider controls (2026-09-16) reuse Exa and Firecrawl path geometry from
`studio/src/components/manage/SearchLogo.vue`, the Tavily SVG from
`studio/src/assets/tavily.svg`, and Brave from the same installed simple-icons
package (https://brave.com). Perplexity reuses the existing mark above. Native
search controls and result cards use SearchLogo.vue's brand colours (including
its light/dark Exa blue) and Tavily's original badge. Model-maker avatars remain
neutral. No new icon dependency or remote fetch.
