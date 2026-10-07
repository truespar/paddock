//! Downloads behind a licence gate at their origin - Hugging Face's "accept
//! the terms with your account" step (Meta's SAM 3 today). We do not mirror
//! such weights; the user downloads them with their own token, from the
//! model's own repository, and the catalog pins the revision and every hash
//! exactly as it does for our mirror.
//!
//! The token comes from, in order: the one saved in Manager > Settings, the
//! `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` environment variables, and the file
//! `hf auth login` writes (`$HF_HOME/token`, else `~/.cache/huggingface/token`).
//! Those are the places every Hugging Face tool already looks, so someone who
//! has signed in once is not asked again.
//!
//! It rides only requests to huggingface.co itself. The resolve URL redirects
//! to a signed CDN address, and reqwest drops `Authorization` on any redirect
//! that changes host, scheme or port (`redirect::remove_sensitive_headers`),
//! so the token never reaches the CDN.
use super::*;

/// The only gate host there is today.
const HF_HOST: &str = "huggingface.co";

/// Where a Hugging Face token was found, for the Settings card to say.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TokenSource {
    /// saved in Manager > Settings
    Saved,
    /// `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`
    Environment,
    /// the file `hf auth login` writes
    Login,
}

/// Whether `url` is served by Hugging Face itself (not a mirror of it).
pub(crate) fn is_huggingface(url: &str) -> bool {
    reqwest::Url::parse(url).is_ok_and(|u| u.scheme() == "https" && u.host_str() == Some(HF_HOST))
}

/// The model page a gated file belongs to: `https://huggingface.co/<org>/<repo>`
/// from its resolve URL - where the user accepts the terms.
pub(crate) fn repo_page(url: &str) -> Option<String> {
    let u = reqwest::Url::parse(url).ok()?;
    if u.host_str() != Some(HF_HOST) {
        return None;
    }
    let mut seg = u.path_segments()?;
    let (org, repo) = (seg.next()?, seg.next()?);
    (!org.is_empty() && !repo.is_empty()).then(|| format!("https://{HF_HOST}/{org}/{repo}"))
}

/// The login file `hf auth login` writes.
fn login_file() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(home).join("token"));
    }
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .filter(|v| !v.is_empty())?;
    Some(
        PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("token"),
    )
}

/// A plausible token: Hugging Face's are `hf_` and printable ASCII, and a
/// header value cannot carry whitespace or control bytes anyway.
pub(crate) fn valid_token(t: &str) -> bool {
    (8..=256).contains(&t.len()) && t.bytes().all(|b| b.is_ascii_graphic())
}

/// The user's Hugging Face token and where it came from, if there is one.
pub fn huggingface_token(store: Option<&crate::store::Store>) -> Option<(String, TokenSource)> {
    if let Some(t) = store
        .and_then(|s| s.huggingface_token().ok().flatten())
        .filter(|t| valid_token(t))
    {
        return Some((t, TokenSource::Saved));
    }
    for var in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Some(t) = std::env::var(var)
            .ok()
            .map(|t| t.trim().to_owned())
            .filter(|t| valid_token(t))
        {
            return Some((t, TokenSource::Environment));
        }
    }
    let t = std::fs::read_to_string(login_file()?).ok()?;
    let t = t.trim();
    valid_token(t).then(|| (t.to_owned(), TokenSource::Login))
}

impl Registry {
    /// The client a file's download goes through: for a Hugging Face file
    /// one that carries the user's token on its requests to huggingface.co
    /// (see the module note), for every other origin the shared client. No
    /// token, no header - a public repository still downloads, and a gated
    /// one fails with [`DlError::Gated`] saying how to get access.
    pub(crate) fn client_for(&self, url: &str) -> reqwest::Client {
        if !is_huggingface(url) {
            return self.client.clone();
        }
        let Some((token, _)) = huggingface_token(self.store.as_deref()) else {
            return self.client.clone();
        };
        let Ok(mut value) = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
        else {
            return self.client.clone();
        };
        value.set_sensitive(true);
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::AUTHORIZATION, value);
        reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .unwrap_or_else(|_| self.client.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_huggingface_itself_counts() {
        assert!(is_huggingface(
            "https://huggingface.co/facebook/sam3/resolve/abc/model.safetensors"
        ));
        assert!(!is_huggingface(
            "https://models.truespar.io/models/x/model.safetensors"
        ));
        assert!(!is_huggingface(
            "https://huggingface.co.evil.example/facebook/sam3"
        ));
        assert!(!is_huggingface(
            "http://huggingface.co/facebook/sam3/resolve/abc/x"
        ));
        assert_eq!(
            repo_page("https://huggingface.co/facebook/sam3/resolve/abc/model.safetensors")
                .as_deref(),
            Some("https://huggingface.co/facebook/sam3")
        );
        assert_eq!(repo_page("https://models.truespar.io/models/x"), None);
    }

    #[test]
    fn tokens_are_plain_printable_ascii() {
        assert!(valid_token("hf_abcdefghijklmnopqrstuvwxyz0123456789"));
        assert!(!valid_token("hf_abc def"));
        assert!(!valid_token("hf_abc\ndef0123"));
        assert!(!valid_token("short"));
    }
}
