import PaddockStudio
import SwiftUI

/// Stable identity across streamed deltas. Geometry lives outside observation;
/// only layout changes notify the small scrolling modifier, never the transcript.
@Observable @MainActor final class TranscriptScrollIntent {
  var pinned = true {
    didSet {
      if pinned, !oldValue {
        reading = nil
        memory?.reading(.init(following: true))
      }
    }
  }
  var readingDisclosure = false
  var awayFromBottom = false
  var layoutRevision = 0
  let initiallyFollowing: Bool
  @ObservationIgnored var reading: StudioConversationViewState.Reading?
  @ObservationIgnored let memory: StudioConversationPosition?
  @ObservationIgnored private var measuredRows: [String: CGSize] = [:]
  @ObservationIgnored private var frames: [String: CGRect] = [:]
  @ObservationIgnored private var order: [String] = []

  init(memory: StudioConversationPosition? = nil) {
    self.memory = memory
    reading = memory?.value.reading.flatMap { $0.following ? nil : $0 }
    initiallyFollowing = memory?.value.reading?.following ?? true
    pinned = initiallyFollowing
  }
  func didLayout(_ id: String, size: CGSize) {
    guard size.width.isFinite, size.height.isFinite, size.width > 0, size.height >= 0,
      measuredRows[id] != size
    else { return }
    measuredRows[id] = size
    if pinned || reading != nil { layoutRevision &+= 1 }
  }
  func didLayout(_ id: String, frame: CGRect) {
    guard frame.minY.isFinite, frame.width.isFinite, frame.height.isFinite,
      frame.width > 0, frame.height >= 0, frames[id] != frame
    else { return }
    frames[id] = frame
    if pinned || reading != nil { layoutRevision &+= 1 }
  }
  func reveal() {
    readingDisclosure = true
    pinned = false
    reading = nil
  }
  func retainRows(_ ids: [String]) {
    order = ids
    let keep = Set(ids + ["native-bottom"])
    measuredRows = measuredRows.filter { keep.contains($0.key) }
    frames = frames.filter { keep.contains($0.key) }
  }
  func capture(y: CGFloat) {
    guard y.isFinite, !order.isEmpty else { return }
    if pinned {
      memory?.reading(.init(following: true))
      return
    }
    guard
      let index = order.firstIndex(where: { (frames[$0]?.maxY ?? -.infinity) > y + 1 })
        ?? order.indices.last,
      let frame = frames[order[index]], frame.height > 0
    else { return }
    let value = StudioConversationViewState.Reading(
      following: false, blockID: order[index], index: index,
      offset: (y - frame.minY).rounded(), height: frame.height, width: frame.width)
    reading = value
    memory?.reading(value)
  }
  func restorationY() -> CGFloat? {
    guard !pinned, let reading, !order.isEmpty else { return nil }
    let id =
      order.contains(reading.blockID)
      ? reading.blockID : order[min(max(0, reading.index), order.count - 1)]
    guard let frame = frames[id], frame.height > 0 else { return nil }
    // Pixels are exact at the same width. A changed line wrap uses the saved
    // position within that message, not a fraction of the whole conversation.
    let offset =
      abs(frame.width - reading.width) >= 1 && reading.height > 0
      ? reading.offset / reading.height * frame.height : reading.offset
    return max(0, frame.minY + min(max(-4096, offset), frame.height - 1))
  }
}
