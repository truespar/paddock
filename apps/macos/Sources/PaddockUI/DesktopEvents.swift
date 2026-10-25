import Foundation
import PaddockClient
import PaddockNativeMarkdown
import PaddockStudio

struct DesktopEvent: Equatable, Sendable {
  enum Kind: String, Sendable { case modelReady, modelIssue, replyReady, replyIssue, approval }
  let id: String
  let kind: Kind
  let route: DesktopAction
  var preview: String? = nil
  var title: String {
    switch kind {
    case .modelReady: "Your model is ready"
    case .modelIssue: "A model needs attention"
    case .replyReady: "Your reply is ready"
    case .replyIssue: "A reply needs attention"
    case .approval: "A tool needs your approval"
    }
  }
  var body: String {
    switch kind {
    case .modelReady: "Open Studio to start a conversation."
    case .modelIssue: "Open Settings > Instances to check the endpoint."
    case .replyReady: "Open Studio to read the response."
    case .replyIssue: "Open Studio to review the response status."
    case .approval: "Review the requested action in Studio."
    }
  }
}

/// Edge detection over the authoritative projections. Bounded state, no timer,
/// raw response bodies, model names, tool arguments or notification-side engine.
struct DesktopEventTracker {
  private var previous: ManagerSnapshot?
  private var unhealthy = Set<String>()
  private var missed = Set<String>()
  private var replySessions: [String: Set<String>] = [:]
  private var approvalSessions: [String: Set<String>] = [:]
  private var sessionOrder: [String] = []
  private var completedTurns = Set<String>()
  private var completionOrder: [String] = []
  private var studioInitialized = false

  mutating func management(_ value: ManagerSnapshot) -> [DesktopEvent] {
    defer { previous = value }
    guard let previous else { return [] }  // Restoring inventory is not an event.
    var events: [DesktopEvent] = []
    let jobs = value.jobs ?? []
    let oldJobs = Dictionary(uniqueKeysWithValues: (previous.jobs ?? []).map { ($0.id, $0.state) })
    for job in jobs where !job.isActive && oldJobs[job.id] != job.state {
      if job.state == "failed" {
        events.append(.init(id: "job-\(job.id)", kind: .modelIssue, route: .manager))
      }
    }
    let stopping = Set(
      jobs.filter {
        $0.action == "stop" && ($0.isActive || oldJobs[$0.id] == "running" || oldJobs[$0.id] == nil)
      }.map(\.port))
    for runner in value.runners where runner.status == "ok" {
      if !previous.runners.contains(where: { $0.id == runner.id && $0.status == "ok" }),
        !stopping.contains(runner.port)
      {
        events.append(
          .init(id: "ready-\(runner.id)", kind: .modelReady, route: .chat(port: runner.port)))
      }
      unhealthy.remove(runner.id)
      missed.remove(runner.id)
    }
    // Two consecutive successful inventory samples are required for an outage.
    // A failed snapshot doesn't count as a crashed runner. Keep disappeared
    // identities for the second sample; don't mistake planned stops for crashes.
    let candidates = Set(previous.runners.filter { $0.status == "ok" }.map(\.id)).union(missed)
    for id in candidates {
      let port = UInt16(id.split(separator: ":").first ?? "")
      if port.map(stopping.contains) == true {
        missed.remove(id)
        unhealthy.remove(id)
        continue
      }
      if value.runners.contains(where: { $0.port == port && $0.status == "ok" }) {
        // Healthy same-port takeover is not an outage of the old process.
        missed.remove(id)
        unhealthy.remove(id)
        continue
      }
      if missed.contains(id), !unhealthy.contains(id) {
        events.append(.init(id: "unhealthy-\(id)", kind: .modelIssue, route: .manager))
        unhealthy.insert(id)
        missed.remove(id)
      } else if !unhealthy.contains(id) {
        missed.insert(id)
      }
    }
    let live = Set(value.runners.map(\.id)).union(missed)
    unhealthy.formIntersection(live)
    return events
  }

  mutating func studio(_ value: StudioState) -> [DesktopEvent] {
    guard let activity = value.activity else { return [] }
    let events = studio(
      conversationID: value.conversation?.id, busy: value.busy, activity: activity)
    return events.map { event in
      guard event.kind == .replyReady,
        case .conversation(let id) = event.route, id == value.conversation?.id,
        value.nativeTranscript?.conversationId == id,
        let messages = value.nativeTranscript?.messages
      else { return event }
      let start = messages.lastIndex { $0.role == "user" }.map { $0 + 1 } ?? messages.endIndex
      let replies = messages[start...].filter { message in
        message.role == "assistant" && !message.streaming && !message.stopped
          && message.error.isEmpty
          && activity.replies.contains(where: { $0.id == message.id && $0.state == "completed" })
      }
      let reply: StudioState.NativeTranscript.Message?
      if event.id.hasPrefix("reply-") {
        reply = replies.first { event.id == "reply-\($0.id)" }
      } else if let group = replies.last?.group {
        reply = replies.first { $0.group == group && !$0.text.isEmpty }
      } else {
        reply = replies.last
      }
      guard let reply, let excerpt = NotificationExcerpt.text(reply.text) else { return event }
      var event = event
      // Compare sends one banner. Name the lane supplying the excerpt rather
      // than suggesting that both models gave this answer.
      let name = reply.chrome?.modelName ?? reply.model
      event.preview =
        reply.group == nil || name.isEmpty
        ? excerpt
        : NotificationExcerpt.text("\(name): \(excerpt)")
      return event
    }
  }

  mutating func studio(conversationID: String?, busy: Bool, activity: StudioState.Activity)
    -> [DesktopEvent]
  {
    let completion = activity.completedTurn
    let freshCompletion =
      studioInitialized && completion.map { !completedTurns.contains($0.id) } == true
    if let completion, completedTurns.insert(completion.id).inserted {
      completionOrder.append(completion.id)
      if completionOrder.count > 256 { completedTurns.remove(completionOrder.removeFirst()) }
    }
    studioInitialized = true
    guard let conversationID else { return [] }
    var observedReplies = replySessions[conversationID] ?? []
    var approvals = approvalSessions[conversationID] ?? []
    defer {
      replySessions[conversationID] = observedReplies
      approvalSessions[conversationID] = approvals
      sessionOrder.removeAll { $0 == conversationID }
      sessionOrder.append(conversationID)
      if sessionOrder.count > 128 {
        let oldest = sessionOrder.removeFirst()
        replySessions[oldest] = nil
        approvalSessions[oldest] = nil
      }
    }
    var events: [DesktopEvent] = []
    let currentApprovals = Set(activity.approvals)
    for id in currentApprovals.subtracting(approvals) {
      events.append(
        .init(id: "approval-\(id)", kind: .approval, route: .conversation(conversationID)))
    }
    approvals = currentApprovals
    for reply in activity.replies {
      if reply.state == "streaming" {
        observedReplies.insert(reply.id)
      } else if !busy, observedReplies.remove(reply.id) != nil, reply.state != "stopped",
        !freshCompletion
      {
        events.append(
          .init(
            id: "reply-\(reply.id)", kind: reply.state == "completed" ? .replyReady : .replyIssue,
            route: .conversation(conversationID)))
      }
    }
    observedReplies.formIntersection(Set(activity.replies.map(\.id)))
    if freshCompletion, let completion, completion.state != "stopped" {
      events.append(
        .init(
          id: "turn-\(completion.id)",
          kind: completion.state == "completed" ? .replyReady : .replyIssue,
          route: .conversation(completion.conversationId)))
    }
    // One notification per terminal compare turn, not one per lane.
    if let issue = events.first(where: { $0.kind == .replyIssue }) {
      events.removeAll { $0.kind == .replyReady || $0.kind == .replyIssue }
      events.append(issue)
    } else if let reply = events.first(where: { $0.kind == .replyReady }) {
      events.removeAll { $0.kind == .replyReady }
      events.append(reply)
    }
    return events
  }
}
