//! What the scheduler does on an idle wake (every 250 ms with nothing in
//! flight).

use crate::generator::Generator;
use crate::pacing::Pacers;
use crate::spec_policy::SpecController;
use crate::tick_calibration::Saver;

/// Save what the last busy stretch learned, and pump the KV tier: idle is
/// when the durability plan writes the sessions that just went quiet to
/// disk - pumped only on busy passes, a server that went idle kept them
/// RAM-only until the next request or shutdown.
pub(super) fn wake(
    calib: &mut Saver,
    pacers: &Pacers,
    spec: &SpecController,
    generator: &mut dyn Generator,
) {
    calib.tick(true, pacers, spec);
    generator.tier_pump();
}

/// Leaving the scheduler on a shutdown request: the KV tier gets its last
/// chance to keep what only the GPU holds (`Generator::tier_flush_at_exit`).
pub(super) fn leave(generator: &mut dyn Generator, live: usize) {
    tracing::info!("engine: shutdown requested - leaving the scheduler ({live} live)");
    generator.tier_flush_at_exit();
}
