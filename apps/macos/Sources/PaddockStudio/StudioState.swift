import Foundation

/// A small allowlisted presentation projection, not a second conversation
/// document. Raw Responses objects and credentials stay out. The isolated
/// native renderer trial opts into a bounded projection of active-path text.
public struct StudioState: Decodable, Sendable {
  public struct Audio: Decodable, Sendable {
    public struct Menu: Decodable, Sendable {
      public let offered: Bool
      public let menu: Bool
      public let needsSetup: Bool
      public let jobChoice: Bool
      public let deviceChoice: Bool
      public let earChoice: Bool
      public let setupMessage: String
      public let setupAction: String
    }
    public struct SpeechModel: Decodable, Sendable, Identifiable {
      public let port: UInt16
      public let model: String?
      public let title: String
      public let vendor: String
      public let running: Bool
      public let busy: Bool
      public let status: String
      public let canStart: Bool
      public let canStop: Bool
      public var id: UInt16 { port }
    }
    public let menu: Menu?
    public let speechModels: [SpeechModel]?
    public let speechError: String?
    public struct Named: Decodable, Sendable, Identifiable {
      public let id: String
      public let label: String
      public let available: Bool?
    }
    public struct Item: Decodable, Sendable, Equatable {
      public let index: Int
      public let text: String
    }
    public struct Attachment: Decodable, Sendable {
      public let attachmentId: String
      public let name: String
      public let mime: String
      public let size: Int?
    }
    public let mode: String
    public let phase: String
    public let retryAvailable: Bool
    public let jobs: [String]
    public let audioMode: Bool
    public let audioOk: Bool
    public let liveBlocked: Bool
    public let liveReason: String
    public let transcribers: [Named]
    public let transcriber: String
    public let devices: [Named]
    public let devicesNamed: Bool?
    public let device: String
    public let language: String
    public let languages: [Composer.Choice]
    public let session: String
    public let dictation: [Item]
    public let provisional: String
    public let levels: [Double]
    public let elapsed: Double
    public let remaining: Double
    public let limit: Double
    public let arming: Bool
    public let idle: Bool
    public let shouldStop: Bool
    public let error: String
    public let deviceNote: String
    public let attachment: Attachment?
    public var busy: Bool { phase != "idle" }
    /// Only an active dictation session decorates the text composer. A stale
    /// snapshot after cancel/failure must never leave words looking committed.
    public var composerProvisional: String {
      mode == "dictate" && busy && error.isEmpty ? provisional : ""
    }
  }
  public let audio: Audio?
  public struct Activity: Decodable, Sendable {
    public struct CompletedTurn: Decodable, Sendable {
      public let id: String
      public let conversationId: String
      public let state: String
    }
    public let completedTurn: CompletedTurn?
    public struct Reply: Decodable, Sendable {
      public let id: String
      public let state: String
    }
    public let replies: [Reply]
    public let approvals: [String]
  }
  public let activity: Activity?
  public struct Document: Decodable, Sendable, Identifiable, Equatable {
    public let id: String
    public let name: String
    public let kind: String
    public let pages: Int?
    public let pageRange: String?
    public let textOnly: Bool
  }
  public let nativeDocument: Document?
  public struct Graph: Decodable, Sendable {
    public let available: Bool
    public let visible: Bool
  }
  public let nativeGraph: Graph?
  public struct Artifact: Decodable, Sendable, Identifiable {
    public let id: String
    public let kind: String
    public let title: String
    public let language: String
    public let model: String
    public let versions: Int
    public let updatedAt: Double
  }
  public let nativeArtifacts: [Artifact]?
  public let nativeArtifactsPaneOpen: Bool?
  public struct AudioClip: Decodable, Sendable, Equatable, Identifiable {
    public let id: String
    public let name: String
    public let mime: String
    public let size: Int?
    public let duration: Double?
  }
  public let nativeAudioPreview: AudioClip?
  public let markUnsure: Bool?
  public struct Speech: Decodable, Sendable, Equatable {
    public struct Diarizer: Decodable, Sendable, Equatable {
      public let id: String
      public let title: String
      public let port: Int
    }
    public struct Diarization: Decodable, Sendable, Equatable {
      public struct Interval: Decodable, Sendable, Equatable {
        public let speaker: Int
        public let start: Double
        public let end: Double
      }
      public let model: String
      public let duration: Double
      public let segments: [Interval]
    }
    public struct Segment: Decodable, Sendable, Equatable {
      public let start: Double
    }
    public struct Word: Decodable, Sendable, Equatable {
      public let word: String
      public let start: Double?
      public let end: Double?
      public let confidence: Double?
      public let alt: String?
      public let margin: Double?
      public let segment: Int
      public let speaker: Int?
      public let speakers: [Int]?
    }
    public struct Fact: Decodable, Sendable, Equatable {
      public let label: String
      public let value: String
    }
    public struct Guard: Decodable, Sendable, Equatable {
      public let start: Double
      public let end: Double
      public let note: String
      public let dropped: Bool?
      public let reason: String?
    }
    public let clip: AudioClip?
    public let words: [Word]
    public let segments: [Segment]?
    public let differs: [Int]
    public let facts: [Fact]
    public let guards: [Guard]
    public let subtitleExport: Bool
    public let messageId: String?
    public let conversationId: String?
    public let diarization: Diarization?
    public let diarizers: [Diarizer]?
  }
  public struct NativeTranscript: Decodable, Sendable {
    public struct Message: Decodable, Sendable, Identifiable, Equatable {
      public struct Actions: Decodable, Sendable, Equatable {
        public struct Branch: Decodable, Sendable, Equatable {
          public let index: Int
          public let count: Int
          public let previous: String?
          public let next: String?
        }
        public let edit: Bool
        public let retry: Bool
        public let continueReply: Bool
        public let branch: Branch?
      }
      public let actions: Actions?
      public struct Chrome: Decodable, Sendable, Equatable {
        public struct Section: Decodable, Sendable, Identifiable, Equatable {
          public struct Row: Decodable, Sendable, Identifiable, Equatable {
            public let label: String
            public let value: String
            public var id: String { label }
          }
          public let id: String
          public let title: String
          public let rows: [Row]
        }
        public let modelName: String
        public let vendor: String
        public let spec: String
        public let footer: String
        public let footerHint: String
        public let thinkingLabel: String
        public let thinkingMeta: String
        public let cutNote: String
        public let sections: [Section]
        public let promptText: String
        /// Actual MCP configuration and completed speech comparison result.
        public let tools: [String]?
        public let fastest: Bool?
      }
      public let id: String
      public let role: String
      public let text: String
      public let reasoning: String
      public let model: String
      public let streaming: Bool
      public let stopped: Bool
      public let error: String
      public let incomplete: Bool
      public let chrome: Chrome?
      public let attachments: [Document]?
      public let audioClips: [AudioClip]?
      public let audioPending: Bool?
      public let speech: Speech?
      public let automatic: Bool?
      public struct Picture: Decodable, Sendable, Identifiable, Equatable {
        public let id: String
        public let name: String
        public let preview: Bool
        public let dataURL: String?
      }
      public let pictures: [Picture]?
      public let imageGeneration: Bool?
      public struct ToolCall: Decodable, Sendable, Identifiable, Equatable {
        public let id: String
        public let name: String
        public let server: String
        public let arguments: String
        public let output: String
        public let status: String
        public let error: String
        public let approvalId: String?
        public let artifactId: String?
      }
      public struct Search: Decodable, Sendable, Identifiable, Equatable {
        public struct Source: Decodable, Sendable, Equatable {
          public let title: String
          public let url: String
        }
        public let id: String
        public let query: String
        public let status: String
        public let provider: String
        public let error: String
        public let sources: [Source]
      }
      public struct File: Decodable, Sendable, Identifiable, Equatable {
        public let id: String
        public let name: String
        public let kind: String
        public let mime: String
        public let stored: Bool
      }
      public struct DocumentResult: Decodable, Sendable, Equatable {
        public struct Page: Decodable, Sendable, Identifiable, Equatable {
          public struct Region: Decodable, Sendable, Equatable {
            public let label: String
            public let text: String
            public let boxes: [[Double]]
            public let quads: [[Double]]
          }
          public let id: Int
          public let state: String
          public let text: String
          public let note: String
          public let regions: [Region]
          public let unsure: [Speech.Fact]
          public let number: Int?
          public let pdfPage: Int?
          public let sourceID: String?
          public let attachmentID: String?
          public let name: String?
        }
        public let facts: [Speech.Fact]
        public let pages: [Page]
      }
      public let toolCalls: [ToolCall]?
      public let searches: [Search]?
      public let files: [File]?
      public let documentResult: DocumentResult?
      /// Shared tree-step anchor, not the currently selected composer models.
      public let group: String?
      public let contended: Bool?
    }
    public let available: Bool
    public let notice: String
    public let conversationId: String?
    public let leafId: String?
    public let messages: [Message]
    public struct Context: Decodable, Sendable {
      public let before: String?
      public let title: String
      public let summary: String
      public let working: Bool
      public let error: String
    }
    public var context: Context? = nil
    public struct Block: Identifiable, Sendable {
      public let id: String
      public let comparison: Bool
      public var messages: [Message]
    }
    public var blocks: [Block] {
      var result: [Block] = []
      for message in messages {
        if let group = message.group, let last = result.last, last.comparison, last.id == group {
          result[result.count - 1].messages.append(message)
        } else {
          result.append(
            Block(
              id: message.group ?? message.id, comparison: message.group != nil, messages: [message]
            ))
        }
      }
      return result
    }
    public var hasComparisons: Bool { messages.contains { $0.group != nil } }
  }
  public let nativeTranscript: NativeTranscript?
  public struct Conversation: Decodable, Sendable, Identifiable {
    public let id: String
    public let title: String
    public let model: String
    public let messageCount: Int
  }
  public struct History: Decodable, Sendable, Identifiable {
    public let id: String
    public let title: String
    public let model: String
    public let updatedAt: Double
    public let pinned: Bool?
    public let kind: String?
    public let busy: Bool?
    public let titleState: String?
  }
  public let historyTotal: Int?
  public let autoTitle: Bool?
  public struct Library: Decodable, Sendable {
    public let rows: [History]
    public let page: Int
    public let pageSize: Int
    public let total: Int
    public let matched: Int
    public let search: String
    public let sort: String
  }
  public let library: Library?
  public struct Model: Decodable, Sendable, Identifiable {
    public let id: String
    public let title: String
    public let provider: String
    public let vendor: String
    public let status: String
    public let port: UInt16?
    public let vision: Bool
    public let audio: Bool
    public let chat: Bool
    public let image: Bool?
  }
  public struct Viewport: Decodable, Sendable {
    public let left: Double
    public let width: Double
  }
  public struct Capabilities: Decodable, Sendable {
    public let reasoning: String
    public let levels: [String]
    public let reasoningDefault: String
    public let reasoningOff: Bool
    public let preserveThinking: Bool
    public let thinkingBudget: Bool
    public let webSearch: Bool
    public let vision: Bool
    public let context: Int
    public let ocrModes: [String]
    public let ocrGrounding: Bool?
    public let docParser: Bool
    public let hasDocument: Bool?
    public let pdfRaster: Bool
    public let imageLanes: [ImageLane]?
  }
  public struct ToolGroup: Decodable, Sendable, Identifiable {
    public struct Tool: Decodable, Sendable, Identifiable {
      public let name: String
      public let description: String?
      public let selected: Bool?
      public var id: String { name }
    }
    public let id: String
    public let label: String
    public let connectorId: String
    public let tools: [Tool]
    public let status: String
    public let checked: String?
    public let total: Int?
    public let selectedCount: Int?
  }
  public struct Composer: Decodable, Sendable {
    public let imageMode: Bool?
    public let imageCaps: [String: StudioValue]?
    public let imageEditing: Bool?
    public struct Choice: Decodable, Sendable, Identifiable {
      public let value: String
      public let label: String
      public var id: String { value }
    }
    public struct Dial: Decodable, Sendable, Identifiable {
      public let key: String
      public let label: String
      public let min: Double
      public let max: Double
      public let step: Double
      public let value: Double
      public let display: String
      public let set: Bool
      public var id: String { key }
    }
    public let reasoning: [Choice]
    public let reasoningChoice: String
    public let preserveThinking: Bool
    public let thinkingBudget: Bool
    public let webSearch: Bool
    public let audioMode: Bool
    public let docParser: Bool
    public let inputIssue: String
    public let warnings: [String]
    public let reasoningNotice: String?
    public let toolCount: Int
    public let samplerSet: Bool
    public let samplingSource: String
    public let samplingWarnings: [String]
    public let sampling: [Dial]
    public let cost: Double
    public let contextUsed: Int
  }
  public let version: Int
  public let revision: UInt64
  public let conversation: Conversation?
  public let history: [History]
  public let models: [Model]
  public let selectedModels: [String]
  public let modelHeader: StudioModelHeader?
  public let busy: Bool
  public let loading: Bool
  public let error: String
  public let viewport: Viewport
  public let previewing: Bool
  public let unsavedEdits: Bool
  public let settings: [String: StudioValue]
  public let capabilities: Capabilities
  public let tools: [ToolGroup]
  public let composer: Composer?
}

/// Codable data only. The command vocabulary is fixed on both sides; this
/// never becomes an evaluateJavaScript source string or a filesystem path.
public enum StudioValue: Codable, Sendable, Equatable {
  case string(String)
  case number(Double)
  case bool(Bool)
  case array([StudioValue])
  case object([String: StudioValue])
  case null
  public init(from decoder: any Decoder) throws {
    let c = try decoder.singleValueContainer()
    if c.decodeNil() {
      self = .null
    } else if let v = try? c.decode(Bool.self) {
      self = .bool(v)
    } else if let v = try? c.decode(Double.self) {
      self = .number(v)
    } else if let v = try? c.decode(String.self) {
      self = .string(v)
    } else if let v = try? c.decode([StudioValue].self) {
      self = .array(v)
    } else {
      self = .object(try c.decode([String: StudioValue].self))
    }
  }
  public func encode(to encoder: any Encoder) throws {
    var c = encoder.singleValueContainer()
    switch self {
    case .string(let v): try c.encode(v)
    case .number(let v): try c.encode(v)
    case .bool(let v): try c.encode(v)
    case .array(let v): try c.encode(v)
    case .object(let v): try c.encode(v)
    case .null: try c.encodeNil()
    }
  }
  public var text: String? { if case .string(let v) = self { v } else { nil } }
  public var boolean: Bool? { if case .bool(let v) = self { v } else { nil } }
  public var number: Double? { if case .number(let v) = self { v } else { nil } }
  public var object: [String: StudioValue]? { if case .object(let v) = self { v } else { nil } }
  public var array: [StudioValue]? { if case .array(let v) = self { v } else { nil } }
}

public struct StudioAttachment: Identifiable, Sendable {
  public let id: String
  public let name: String
  public let mime: String
  public let size: Int
  public var phase: String
  public var error: String?
  public var pages: Int?
  public var thumbnail: Data?
  public var width: Int?
  public var height: Int?
  public var detail = "auto"
  // Keep the API's interoperable auto/low/high values in saved conversations.
  // Originals stay in the attachment store; only model input is resized.
  public static let imageDetailOptions: [(value: String, title: String, help: String)] = [
    ("auto", "Auto-resize", "Fit large images for analysis. Your original is kept."),
    (
      "high", "Original",
      "Use full-detail input. Model limits still apply; processing can take longer."
    ),
    ("low", "Smaller", "Use the model's smallest image size for a quick overview."),
  ]
  public var textOnly = false
  public var isAudio: Bool {
    mime.hasPrefix("audio/")
      || ["wav", "mp3", "m4a", "aac", "flac", "ogg", "oga", "opus", "webm", "mp4", "mpga", "mpeg"]
        .contains((name as NSString).pathExtension.lowercased())
  }
  // Keep invalid intermediate edits too: a formatted numeric TextField can
  // display rejected text while Send silently uses the previous valid number.
  public var firstPage = ""
  public var lastPage = ""
  public var from: Int? {
    get { Int(firstPage.trimmingCharacters(in: .whitespacesAndNewlines)) }
    set { firstPage = newValue.map(String.init) ?? "" }
  }
  public var to: Int? {
    get { Int(lastPage.trimmingCharacters(in: .whitespacesAndNewlines)) }
    set { lastPage = newValue.map(String.init) ?? "" }
  }
  public var isPDF: Bool { mime == "application/pdf" || name.lowercased().hasSuffix(".pdf") }
  public var supportsPageSelection: Bool {
    isPDF || mime == "image/tiff"
      || ["tif", "tiff"].contains((name as NSString).pathExtension.lowercased())
  }
  public var allPages: Bool {
    firstPage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      && lastPage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
  }
  public var pageSummary: String {
    if selectionError != nil { return "Check page range" }
    if from == nil && to == nil {
      return pages.map { "All \($0) \($0 == 1 ? "page" : "pages")" } ?? "All pages"
    }
    let start = from ?? 1
    if let end = to ?? pages {
      return start == end ? "Page \(start)" : "Pages \(start)-\(end)"
    }
    return "Pages \(start)-end"
  }
  public var ready: Bool { phase == "Ready" && error == nil }
  public var selectionError: String? {
    for raw in [firstPage, lastPage] {
      let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
      if !value.isEmpty,
        !value.utf8.allSatisfy({ (48...57).contains($0) })
          || Int(value).map({ $0 > 9_007_199_254_740_991 }) != false
      {
        return "Enter whole page numbers, or leave a bound blank."
      }
    }
    if let from, from < 1 { return "The first page must be at least 1." }
    if let to, to < 1 { return "The last page must be at least 1." }
    if let from, let to, from > to { return "The page range ends before it starts." }
    if let pages, (from ?? 1) > pages || (to ?? 1) > pages {
      return "This file has only \(pages) pages."
    }
    return nil
  }
  public var choices: StudioValue {
    var p: [String: StudioValue] = [
      "id": .string(id), "detail": .string(detail), "text": .bool(textOnly),
    ]
    if let from { p["from"] = .number(Double(from)) }
    if let to { p["to"] = .number(Double(to)) }
    return .object(p)
  }
}
