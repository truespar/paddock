//! A byte-budgeted store of encoded pictures, shared by every vision family.
//!
//! One owner per encoded picture - a prefill borrows it (`Arc`) instead of
//! copying it, an entry in use is never evicted - and the bound is BYTES, so
//! the plan can reserve exactly what the store may hold. The shape follows
//! vLLM's encoder cache. Each family keys it with `picture_key` and stores its
//! own payload (the embedding rows its splice reads).

use std::sync::Arc;

/// Content identity: blake3 over the dimensions and the raw bytes. 256 bits
/// makes a collision a non-event, so the raw bytes need not stay resident for
/// an exact compare.
pub(crate) type PictureKey = [u8; 32];

pub(crate) fn picture_key(rgb: &[u8], w: usize, h: usize) -> PictureKey {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(w as u64).to_le_bytes());
    hasher.update(&(h as u64).to_le_bytes());
    hasher.update(rgb);
    *hasher.finalize().as_bytes()
}

/// A byte-budgeted LRU over shared payloads. An entry some prefill still
/// holds (`Arc` count above one) is never evicted, so the budget can only be
/// overrun by pictures in use - which a pass bounds.
pub(crate) struct PictureStore<T> {
    entries: Vec<Slot<T>>,
    budget: u64,
    clock: u64,
    /// pictures served without the tower (test/telemetry hook)
    pub(crate) reused: u64,
}

struct Slot<T> {
    key: PictureKey,
    value: Arc<T>,
    bytes: u64,
    last_used: u64,
}

impl<T> PictureStore<T> {
    pub(crate) fn new(budget: u64) -> Self {
        Self {
            entries: Vec::new(),
            budget,
            clock: 0,
            reused: 0,
        }
    }

    pub(crate) fn set_budget(&mut self, budget: u64) {
        self.budget = budget;
        self.evict(0);
    }

    pub(crate) fn budget(&self) -> u64 {
        self.budget
    }

    /// Bytes the store holds now, pictures in use included.
    pub(crate) fn bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bytes).sum()
    }

    pub(crate) fn get(&mut self, key: &PictureKey) -> Option<Arc<T>> {
        self.clock += 1;
        let e = self.entries.iter_mut().find(|e| &e.key == key)?;
        e.last_used = self.clock;
        self.reused += 1;
        Some(e.value.clone())
    }

    /// Take ownership of a freshly encoded picture and hand back its borrow,
    /// making room first by evicting the least recently used idle entries.
    pub(crate) fn insert(&mut self, key: PictureKey, value: T, bytes: u64) -> Arc<T> {
        self.evict(bytes);
        self.clock += 1;
        let value = Arc::new(value);
        self.entries.push(Slot {
            key,
            value: value.clone(),
            bytes,
            last_used: self.clock,
        });
        value
    }

    fn evict(&mut self, incoming: u64) {
        let mut held = self.bytes();
        while held + incoming > self.budget {
            let Some(i) = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| Arc::strong_count(&e.value) == 1)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
            else {
                break; // everything left is in use
            };
            held -= self.entries.swap_remove(i).bytes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PictureStore, picture_key};

    fn key(n: u8) -> [u8; 32] {
        picture_key(&[n; 12], 2, 2)
    }

    /// Bytes, not entries: a budget of three pictures holds three, and the
    /// fourth evicts the least recently USED one, not the oldest inserted.
    #[test]
    fn the_budget_is_bytes_and_eviction_is_lru() {
        let mut s = PictureStore::<u32>::new(300);
        drop(s.insert(key(1), 1, 100));
        drop(s.insert(key(2), 2, 100));
        drop(s.insert(key(3), 3, 100));
        assert!(s.get(&key(1)).is_some(), "a hit refreshes 1");
        drop(s.insert(key(4), 4, 100));
        assert_eq!(s.bytes(), 300);
        assert!(s.get(&key(2)).is_none(), "2 was least recently used");
        assert!(s.get(&key(1)).is_some() && s.get(&key(3)).is_some());
        assert_eq!(
            s.reused, 3,
            "hits only - the evicted picture's lookup missed"
        );
    }

    /// A picture a prefill still holds is never evicted - the store runs over
    /// its budget rather than free memory a pass is reading - and it becomes
    /// evictable the moment the borrow ends.
    #[test]
    fn a_picture_in_use_is_never_evicted() {
        let mut s = PictureStore::<u32>::new(200);
        let held = s.insert(key(1), 1, 150);
        let also = s.insert(key(2), 2, 150);
        assert_eq!(s.bytes(), 300, "both in use: over budget, nothing freed");
        drop(held);
        drop(s.insert(key(3), 3, 10));
        assert!(s.get(&key(1)).is_none(), "released, so evicted");
        assert!(s.get(&key(2)).is_some(), "still borrowed, so kept");
        drop(also);
    }

    /// Shrinking the budget (a smaller elected pass) evicts at once.
    #[test]
    fn a_smaller_budget_evicts_idle_pictures() {
        let mut s = PictureStore::<u32>::new(1000);
        for n in 0..5 {
            drop(s.insert(key(n), n as u32, 100));
        }
        s.set_budget(250);
        assert!(s.bytes() <= 250, "{}", s.bytes());
    }

    /// The key covers the dimensions as well as the bytes: the same bytes read
    /// as another shape are another picture.
    #[test]
    fn the_key_covers_dimensions() {
        let bytes = [7u8; 24];
        assert_ne!(picture_key(&bytes, 4, 2), picture_key(&bytes, 2, 4));
        assert_eq!(picture_key(&bytes, 4, 2), picture_key(&bytes, 4, 2));
    }
}
