import Foundation

/// Upstream Chatterbox text normalization, applied immediately before tokenization.
///
/// Ports `punc_norm` from `chatterbox/mtl_tts.py`. Every upstream variant - English,
/// multilingual and Turbo - calls it unconditionally on the way into the tokenizer, so
/// it describes the text distribution the model was trained on rather than an optional
/// cleanup pass.
///
/// The terminal-punctuation rule is the load-bearing one: with no sentence-ending
/// character the model has no cue to emit EOS and trails off instead. The whitespace
/// collapse matters here too, because ``MTLTokenizer`` maps *every* space to a
/// `[SPACE]` token - a run of two spaces emits two of them, which the model never saw.
///
/// The sentence-ender set includes the CJK forms upstream carries. Omitting them
/// appends a spurious `.` to text that already ends in `。` or `？`.
public func puncNorm(_ text: String) -> String {
    if text.isEmpty {
        return "You need to add some text for me to talk."
    }

    var out = text

    // Capitalise first letter. A no-op by the time MTLTokenizer lowercases, but kept
    // so this function is a faithful, independently testable port.
    if let first = out.first, first.isLowercase {
        out = String(first).uppercased() + String(out.dropFirst())
    }

    // Collapse whitespace runs to single spaces and trim both ends.
    // Matches Python `" ".join(text.split())`, which splits on any whitespace.
    out = out.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")

    // Replace uncommon/LLM punctuation. Order is upstream's and is load-bearing:
    // `...` must be consumed before the single-character forms, and ` ,` runs last so
    // it also catches the commas the earlier rules introduce.
    let replacements: [(String, String)] = [
        ("...", ", "),
        ("\u{2026}", ", "),  // …
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("\u{2014}", "-"),  // —
        ("\u{2013}", "-"),  // –
        (" ,", ","),
        ("\u{201C}", "\""),  // “
        ("\u{201D}", "\""),  // ”
        ("\u{2018}", "'"),  // ‘
        ("\u{2019}", "'"),  // ’
    ]
    for (old, new) in replacements {
        out = out.replacingOccurrences(of: old, with: new)
    }

    // Only spaces are stripped, matching Python's `rstrip(" ")`. The replacements
    // above can leave one behind - `"Hello..."` becomes `"Hello, "`.
    while out.hasSuffix(" ") {
        out.removeLast()
    }

    // Add a full stop when nothing terminates the sentence. Written as a suffix test
    // rather than a last-character test so all-whitespace input yields `"."`, as
    // upstream does, instead of an empty string the tokenizer would choke on.
    let sentenceEnders = [".", "!", "?", "-", ",", "\u{3001}", "\u{FF0C}", "\u{3002}", "\u{FF1F}", "\u{FF01}"]
    if !sentenceEnders.contains(where: { out.hasSuffix($0) }) {
        out += "."
    }

    return out
}
