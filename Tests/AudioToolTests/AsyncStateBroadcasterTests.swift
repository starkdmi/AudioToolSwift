//
//  AsyncStateBroadcasterTests.swift
//  AudioToolTests
//
//  Tests for TTS provider state-stream lifecycle management
//

import Testing
import AudioToolCore
@testable import AudioToolTTS

@Suite("TTS state stream lifecycle")
struct AsyncStateBroadcasterTests {

    @Test("Cancellation racing with state updates releases every subscription")
    func cancellationDuringUpdatesDoesNotLeakSubscriptions() async {
        let broadcaster = AsyncStateBroadcaster(initialState: ModelState.notLoaded)
        let subscriberCount = 128
        var consumers: [Task<Void, Never>] = []
        consumers.reserveCapacity(subscriberCount)

        for _ in 0..<subscriberCount {
            consumers.append(Task {
                for await _ in broadcaster.makeStream() {
                    await Task.yield()
                }
            })
        }

        let clock = ContinuousClock()
        let registrationDeadline = clock.now.advanced(by: .seconds(2))
        while broadcaster.activeSubscriptionCount < subscriberCount,
              clock.now < registrationDeadline {
            await Task.yield()
        }
        #expect(broadcaster.activeSubscriptionCount == subscriberCount)

        let updater = Task {
            for index in 0..<subscriberCount {
                broadcaster.send(index.isMultiple(of: 2) ? .loading : .notLoaded)
                await Task.yield()
            }
        }
        for consumer in consumers {
            consumer.cancel()
            await Task.yield()
        }
        for consumer in consumers {
            await consumer.value
        }
        await updater.value

        #expect(broadcaster.activeSubscriptionCount == 0)
    }
}
