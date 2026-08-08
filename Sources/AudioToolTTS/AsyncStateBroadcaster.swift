//
//  AsyncStateBroadcaster.swift
//  AudioToolTTS
//
//  Race-free AsyncStream broadcasting for provider state changes
//

import Foundation

/// Synchronously registers state-stream continuations so cancellation cannot
/// overtake an asynchronous actor registration. A private serial queue preserves
/// state ordering; termination cleanup is enqueued asynchronously because
/// `yield` may synchronously trigger the callback while already on that queue.
final class AsyncStateBroadcaster<State: Sendable>: @unchecked Sendable {
    private let queue = DispatchQueue(label: "AudioToolTTS.AsyncStateBroadcaster")
    private var currentState: State
    private var continuations: [UUID: AsyncStream<State>.Continuation] = [:]

    init(initialState: State) {
        currentState = initialState
    }

    func makeStream() -> AsyncStream<State> {
        let pair = AsyncStream<State>.makeStream()
        let id = UUID()
        pair.continuation.onTermination = { [weak self] _ in
            self?.removeContinuation(id: id)
        }

        queue.sync {
            continuations[id] = pair.continuation
            if case .terminated = pair.continuation.yield(currentState) {
                continuations.removeValue(forKey: id)
            }
        }

        return pair.stream
    }

    func send(_ state: State) {
        queue.sync {
            currentState = state
            var terminated: [UUID] = []
            for (id, continuation) in continuations {
                if case .terminated = continuation.yield(state) {
                    terminated.append(id)
                }
            }
            for id in terminated {
                continuations.removeValue(forKey: id)
            }
        }
    }

    var activeSubscriptionCount: Int {
        queue.sync { continuations.count }
    }

    private func removeContinuation(id: UUID) {
        queue.async { [weak self] in
            self?.continuations.removeValue(forKey: id)
        }
    }
}
